"""
QLoRA fine-tuning script for Qwen3-8B-Instruct using Unsloth.

This script fine-tunes the model for structured financial entity extraction
from SEC 10-K reports using QLoRA (4-bit quantization + LoRA adapters).

Usage:
    python -m src.train --output_dir outputs/qwen3-8b-financial-lora
    
    # Dry run with 2 steps:
    python -m src.train --max_steps 2 --output_dir outputs/test_run
"""

import os
import argparse
from pathlib import Path
from loguru import logger

# Import unsloth first for optimizations
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

from src.data import load_training_data, QWEN3_CHAT_TEMPLATE


# Default configuration
DEFAULT_MODEL = "unsloth/Qwen3-8B-Instruct-unsloth-bnb-4bit"
DEFAULT_OUTPUT_DIR = "outputs/qwen3-8b-financial-lora"
MAX_SEQ_LENGTH = 8192  # Qwen3 supports up to 32K, but 8K is enough for 10-K excerpts


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-8B-Instruct for financial extraction"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help="Base model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for LoRA adapter"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/train.jsonl",
        help="Path to training data JSONL"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Max training steps (-1 for full epochs)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32
) -> tuple:
    """
    Load Qwen3 model with Unsloth optimizations and configure LoRA.
    
    Args:
        model_name: HuggingFace model name or path
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load model with Unsloth's optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Configure LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory optimization
        random_state=42,
    )
    
    # Set up chat template for Qwen3
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",  # Qwen3 uses same template as Qwen2.5
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded with LoRA: r={lora_r}, alpha={lora_alpha}")
    return model, tokenizer


def formatting_prompts_func(examples: dict, tokenizer) -> list[str]:
    """
    Format examples for training using the tokenizer's chat template.
    
    Args:
        examples: Batch of examples with 'messages' key
        tokenizer: Tokenizer with chat template
        
    Returns:
        List of formatted prompt strings
    """
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return texts


def train(args: argparse.Namespace):
    """Main training function."""
    logger.info("Starting QLoRA fine-tuning for financial extraction")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Load training data
    dataset = load_training_data(args.data_path)
    
    # Determine reporting settings
    if args.use_wandb:
        report_to = "wandb"
        os.environ.setdefault("WANDB_PROJECT", "financial-extraction")
    else:
        report_to = "none"
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure training
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=5,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_epochs if args.max_steps <= 0 else 1,
        learning_rate=args.learning_rate,
        fp16=not model.config.torch_dtype == "bfloat16",
        bf16=model.config.torch_dtype == "bfloat16",
        logging_steps=1,
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to=report_to,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,  # Don't pack sequences for this task
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=lambda ex: formatting_prompts_func(ex, tokenizer),
    )
    
    # Print training info
    logger.info(f"Training dataset size: {len(dataset)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # Train
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    
    # Save the LoRA adapter
    logger.info(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Also save in merged format for easier loading (optional)
    merged_dir = output_dir / "merged"
    if args.max_steps <= 0:  # Only for full training runs
        logger.info(f"Saving merged model to {merged_dir}")
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )
    
    logger.info("Training complete!")
    logger.info(f"Final loss: {trainer_stats.training_loss:.4f}")
    
    return trainer_stats


if __name__ == "__main__":
    args = get_args()
    train(args)

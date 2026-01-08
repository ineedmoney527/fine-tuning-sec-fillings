"""
QLoRA fine-tuning script for Qwen3-8B-Instruct using Unsloth.

This script fine-tunes the model for structured financial entity extraction
from SEC 10-K reports using QLoRA (4-bit quantization + LoRA adapters).

Includes chunking support for training on long documents with limited context.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset
from loguru import logger

from chunking import estimate_tokens, chunk_document


# =================CONFIGURATION=================
DEFAULT_MODEL = "unsloth/Qwen3-8B"
DEFAULT_OUTPUT_DIR = "outputs/qwen3-8b-financial-lora"
DEFAULT_DATA_PATH = "data/train.jsonl"

# Chunking configuration for training data
MAX_TRAIN_TOKENS = 4096  # Max tokens per training example
OVERLAP_TOKENS = 200     # Overlap for context continuity

# Custom loss weights for financial tokens
FINANCIAL_KEYS = {
    "revenue", "net_income", "operating_income", "total_assets", 
    "cash_and_equivalents", "diluted_eps", "value", "unit",
    "total_revenue", "cost_of_revenue", "gross_profit", "income_before_tax",
    "earnings_per_share_basic", "earnings_per_share_diluted",
    "total_current_assets", "total_liabilities", "total_shareholders_equity"
}
WEIGHTS = {
    "json_key": 2.0,
    "number": 1.5,
    "json_structure": 1.2,
    "default": 1.0
}


def load_and_chunk_data(
    data_path: str,
    tokenizer,
    max_tokens: int = MAX_TRAIN_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS
) -> List[Dict[str, Any]]:
    """
    Load training data and chunk long examples.
    
    For examples exceeding max_tokens, creates multiple training examples
    from chunks of the user content.
    
    Args:
        data_path: Path to JSONL training file
        tokenizer: Tokenizer for applying chat template
        max_tokens: Maximum tokens per training example
        overlap_tokens: Overlap between chunks
        
    Returns:
        List of formatted training examples with 'text' field
    """
    data = [json.loads(line) for line in open(data_path) if line.strip()]
    logger.info(f"Loaded {len(data)} raw examples from {data_path}")
    
    formatted_data = []
    chunked_count = 0
    
    for ex in data:
        messages = ex.get("messages", [])
        
        # Find user message (document content)
        user_content = None
        system_content = None
        assistant_content = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]
        
        if not user_content:
            continue
        
        # Check if chunking is needed
        token_estimate = estimate_tokens(user_content)
        
        if token_estimate > max_tokens:
            # Chunk the user content
            chunks = chunk_document(user_content, max_tokens, overlap_tokens)
            chunked_count += 1
            
            for i, chunk in enumerate(chunks):
                # Create training example for each chunk
                # All chunks share the same expected output (full extraction)
                chunk_messages = []
                if system_content:
                    chunk_messages.append({"role": "system", "content": system_content})
                
                # Add /no_think suffix for Qwen3 thinking mode
                chunk_messages.append({
                    "role": "user", 
                    "content": chunk + " /no_think"
                })
                
                if assistant_content:
                    chunk_messages.append({"role": "assistant", "content": assistant_content})
                
                text = tokenizer.apply_chat_template(
                    chunk_messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                formatted_data.append({"text": text, "source": f"{ex.get('source', 'unknown')}_chunk{i}"})
        else:
            # No chunking needed
            formatted_messages = []
            for msg in messages:
                content = msg["content"]
                if msg["role"] == "user":
                    content += " /no_think"
                formatted_messages.append({"role": msg["role"], "content": content})
            
            text = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            formatted_data.append({"text": text, "source": ex.get("source", "unknown")})
    
    logger.info(f"Created {len(formatted_data)} training examples ({chunked_count} documents were chunked)")
    return formatted_data


def create_custom_trainer(SFTTrainer, tokenizer):
    """
    Create custom trainer with token-weighted loss for financial extraction.
    
    Weights:
    - Financial JSON keys: 2.0x
    - Numbers: 1.5x  
    - JSON structure: 1.2x
    - Default: 1.0x
    """
    
    class CustomFinancialTrainer(SFTTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            tok = self.processing_class
            
            # Pre-compute token sets for efficient lookup
            self.key_tokens = set()
            for key in FINANCIAL_KEYS:
                self.key_tokens.update(tok.encode(f'"{key}"', add_special_tokens=False))
                self.key_tokens.update(tok.encode(key, add_special_tokens=False))
            
            self.struct_tokens = set()
            for char in '{}[]:",':
                self.struct_tokens.update(tok.encode(char, add_special_tokens=False))
            
            self.digit_tokens = set()
            for char in '0123456789.-':
                self.digit_tokens.update(tok.encode(char, add_special_tokens=False))
            
            logger.info(f"CustomFinancialTrainer initialized with {len(self.key_tokens)} key tokens")
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels", inputs["input_ids"].clone())
            outputs = model(**inputs)
            logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Build weight tensor
            weights = torch.ones_like(shift_labels, dtype=torch.float32)
            for b in range(shift_labels.shape[0]):
                for i in range(shift_labels.shape[1]):
                    token_id = shift_labels[b, i].item()
                    if token_id == -100:
                        weights[b, i] = 0.0
                    elif token_id in self.key_tokens:
                        weights[b, i] = WEIGHTS["json_key"]
                    elif token_id in self.struct_tokens:
                        weights[b, i] = WEIGHTS["json_structure"]
                    elif token_id in self.digit_tokens:
                        weights[b, i] = WEIGHTS["number"]
            
            weights = weights.to(logits.device)
            
            # Weighted cross-entropy loss
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            
            mask = weights.view(-1) > 0
            if mask.sum() > 0:
                loss = (loss * weights.view(-1))[mask].sum() / weights.view(-1)[mask].sum()
            else:
                loss = loss.sum()
            
            return (loss, outputs) if return_outputs else loss
    
    return CustomFinancialTrainer


def train(
    model_name: str = DEFAULT_MODEL,
    data_path: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_seq_length: int = 8192,
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    use_chunking: bool = True,
    max_train_tokens: int = MAX_TRAIN_TOKENS,
    save_merged: bool = True
):
    """
    Run QLoRA fine-tuning with Unsloth.
    
    Args:
        model_name: Base model to fine-tune
        data_path: Path to training JSONL
        output_dir: Directory to save model
        max_seq_length: Maximum sequence length
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation
        learning_rate: Learning rate
        use_chunking: Whether to chunk long training examples
        max_train_tokens: Max tokens per chunk (if chunking enabled)
        save_merged: Whether to save merged model (not just LoRA)
    """
    # CRITICAL: Enable logits for custom loss (Unsloth disables by default)
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    
    # Import Unsloth (requires GPU)
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError:
        logger.error("Unsloth not installed. Run: pip install unsloth")
        raise
    
    from trl import SFTTrainer, SFTConfig
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Configure LoRA
    logger.info("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Set up tokenizer
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and optionally chunk data
    logger.info(f"Loading training data from {data_path}...")
    if use_chunking:
        formatted_data = load_and_chunk_data(
            data_path, tokenizer, 
            max_tokens=max_train_tokens,
            overlap_tokens=OVERLAP_TOKENS
        )
    else:
        # Load without chunking
        data = [json.loads(line) for line in open(data_path) if line.strip()]
        formatted_data = []
        for ex in data:
            messages = [
                {"role": m["role"], "content": m["content"] + " /no_think" if m["role"] == "user" else m["content"]}
                for m in ex["messages"]
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted_data.append({"text": text})
    
    dataset = Dataset.from_list(formatted_data)
    logger.info(f"Training dataset: {len(dataset)} examples")
    
    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=5,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        max_seq_length=max_seq_length,
        packing=False,
        dataset_text_field="text",
    )
    
    # Create custom trainer
    CustomTrainer = create_custom_trainer(SFTTrainer, tokenizer)
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    logger.success("Training complete!")
    
    # Save model
    logger.info(f"Saving LoRA adapters to {output_dir}")
    trainer.save_model(output_dir)
    
    if save_merged:
        merged_dir = f"{output_dir}-merged"
        logger.info(f"Saving merged model to {merged_dir}")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        logger.success(f"Merged model saved to {merged_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3 for financial extraction with QLoRA"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model name")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Training data path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Max sequence length")
    parser.add_argument("--no-chunking", action="store_true", help="Disable training data chunking")
    parser.add_argument("--max-chunk-tokens", type=int, default=MAX_TRAIN_TOKENS, 
                        help="Max tokens per chunk")
    parser.add_argument("--no-merge", action="store_true", help="Don't save merged model")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        max_seq_length=args.max_seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_chunking=not args.no_chunking,
        max_train_tokens=args.max_chunk_tokens,
        save_merged=not args.no_merge
    )


if __name__ == "__main__":
    main()

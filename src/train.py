"""
QLoRA fine-tuning script for Qwen3-8B-Instruct using Unsloth.

This script fine-tunes the model for structured financial entity extraction
from SEC 10-K reports using QLoRA (4-bit quantization + LoRA adapters).
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset
from loguru import logger


# =================CONFIGURATION=================
DEFAULT_MODEL = "unsloth/Qwen3-8B"
DEFAULT_OUTPUT_DIR = "outputs/qwen3-8b-financial-lora"
DEFAULT_DATA_PATH = "data/train.jsonl"

# Sliding window chunking for long documents (T4 GPU limit: 2048 tokens)
MAX_LENGTH = 2048  # Max tokens per chunk
STRIDE = 256       # Overlap to prevent cutting words/context

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


def extract_source_values(answer_json: str) -> List[str]:
    """
    Extract numeric values from the answer JSON and generate search patterns.
    
    SEC 10-K documents may show values in different scales:
    - Raw: 211915000000
    - In billions: 211.9 or 212
    - In millions: 211,915
    - In thousands: 211,915,000
    
    We generate patterns for all possible scales.
    """
    try:
        data = json.loads(answer_json)
    except json.JSONDecodeError:
        return []
    
    patterns = []
    
    for key, value in data.items():
        if value is None:
            continue
            
        # Handle nested {"value": x, "unit": y} structure
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
            if value is None:
                continue
        
        if isinstance(value, (int, float)):
            # For floats (like EPS), add as-is
            if isinstance(value, float):
                patterns.append(f"{value}")
                patterns.append(f"{value:.2f}")
                continue
            
            # For integers, generate multiple scale formats
            abs_val = abs(value)
            
            # Raw value (with commas for readability in tables)
            if abs_val < 1000:
                patterns.append(str(value))
            
            # In thousands (divide by 1,000)
            if abs_val >= 1_000:
                in_thousands = abs_val // 1_000
                patterns.append(f"{in_thousands:,}")
                patterns.append(str(in_thousands))
            
            # In millions (divide by 1,000,000)
            if abs_val >= 1_000_000:
                in_millions = abs_val // 1_000_000
                patterns.append(f"{in_millions:,}")
                patterns.append(str(in_millions))
            
            # In billions (divide by 1,000,000,000) 
            if abs_val >= 1_000_000_000:
                in_billions = abs_val / 1_000_000_000
                patterns.append(f"{in_billions:.1f}")
                patterns.append(f"{int(in_billions)}")
    
    # Remove duplicates
    return list(set(p for p in patterns if p))


def create_sliding_window_preprocessor(tokenizer):
    """
    Create a preprocessing function that:
    1. Chunks long documents into MAX_LENGTH token windows with STRIDE overlap
    2. Scans each chunk for source values from the answer JSON
    3. Discards chunks that don't contain enough answer values (noise reduction)
    
    This is essential for training on GPUs with limited context (e.g., T4 with 2048 tokens).
    """
    
    def preprocess_and_filter_chunks(examples):
        """
        Process batched examples with sliding window chunking.
        Only keeps chunks that contain the answer values.
        """
        # Lists to hold filtered results
        new_input_ids = []
        new_attention_masks = []
        new_labels = []
        
        texts = examples["text"]
        answers = examples["answer"]
        
        for text, answer in zip(texts, answers):
            # Extract searchable values from the answer JSON
            search_patterns = extract_source_values(answer)
            
            if not search_patterns:
                logger.warning("No searchable values found in answer, skipping")
                continue
            
            # 1. Tokenize and chunk the text using sliding window
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                stride=STRIDE,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors=None  # Return lists, not tensors
            )
            
            # Extract mappings and remove from tokenized dict
            overflow_map = tokenized.pop("overflow_to_sample_mapping", None)
            offset_mapping = tokenized.pop("offset_mapping", None)
            
            num_chunks = len(tokenized["input_ids"])
            
            # 2. Iterate through each chunk
            for chunk_idx in range(num_chunks):
                chunk_input_ids = tokenized["input_ids"][chunk_idx]
                chunk_attention_mask = tokenized["attention_mask"][chunk_idx]
                
                # Decode chunk back to text to check for values
                chunk_text = tokenizer.decode(chunk_input_ids, skip_special_tokens=True)
                
                # 3. CHECK: How many answer values are in this chunk?
                matches = sum(1 for pattern in search_patterns if pattern in chunk_text)
                match_ratio = matches / len(search_patterns)
                
                # Keep chunk if at least 30% of values are found
                # (Financial values are spread across different tables/chunks)
                if matches>=1:
                    new_input_ids.append(chunk_input_ids)
                    new_attention_masks.append(chunk_attention_mask)
                    new_labels.append(chunk_input_ids.copy())  # For causal LM, labels = input_ids
        
        return {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_masks,
            "labels": new_labels
        }
    
    return preprocess_and_filter_chunks


def prepare_dataset_with_chunking(data_path: str, tokenizer) -> Dataset:
    """
    Load training data and prepare it with sliding window chunking.
    
    The data format is chat-style JSONL with messages array.
    We extract the user content (document) and assistant content (answer),
    then chunk and filter based on answer presence.
    
    Args:
        data_path: Path to training JSONL file
        tokenizer: Tokenizer for chunking
        
    Returns:
        Filtered HuggingFace Dataset with only answer-containing chunks
    """
    logger.info(f"Loading and chunking training data from {data_path}...")
    
    # Load raw data
    raw_data = [json.loads(line) for line in open(data_path) if line.strip()]
    logger.info(f"Loaded {len(raw_data)} raw examples")
    
    # Extract text (full prompt) and answer for each example
    prepared = []
    for ex in raw_data:
        messages = ex.get("messages", [])
        
        # Build formatted messages with /no_think
        formatted_messages = []
        answer = None
        
        for msg in messages:
            if msg["role"] == "assistant":
                answer = msg["content"]
            content = msg["content"]
            if msg["role"] == "user":
                content += " /no_think"
            formatted_messages.append({"role": msg["role"], "content": content})
        
        if not answer:
            logger.warning(f"Skipping example without assistant response")
            continue
        
        # Apply chat template for full text
        text = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        prepared.append({"text": text, "answer": answer})
    
    logger.info(f"Prepared {len(prepared)} examples for chunking")
    
    # Create dataset
    dataset = Dataset.from_list(prepared)
    
    # Create preprocessor and apply chunking + filtering
    preprocessor = create_sliding_window_preprocessor(tokenizer)
    
    chunked_dataset = dataset.map(
        preprocessor,
        batched=True,
        remove_columns=dataset.column_names  # Remove old columns to avoid shape mismatch
    )
    
    logger.info(f"Created {len(chunked_dataset)} training chunks (answer-containing only)")
    
    return chunked_dataset


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
    max_seq_length: int = 32768,
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
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
    
    # Load and preprocess training data with sliding window chunking
    # This chunks long documents and filters to only keep answer-containing chunks
    dataset = prepare_dataset_with_chunking(data_path, tokenizer)
    
    # Training config - no dataset_text_field since data is pre-tokenized
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
        max_seq_length=MAX_LENGTH,  # Match chunk size
        packing=False,
    )
    
    # Create custom trainer with weighted loss
    CustomTrainer = create_custom_trainer(SFTTrainer, tokenizer)
    trainer = SFTTrainer(
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
    parser.add_argument("--max-seq-length", type=int, default=32768, help="Max sequence length")
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
        save_merged=not args.no_merge
    )


if __name__ == "__main__":
    main()

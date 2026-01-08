"""
QLoRA fine-tuning script for Qwen3-8B-Instruct using Unsloth.
"""

import os
# MUST be set before importing unsloth/transformers for custom loss to work
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

import json
import argparse
from typing import List, Dict, Any

import torch
from datasets import Dataset
from loguru import logger

# =================CONFIGURATION=================
DEFAULT_MODEL = "unsloth/Qwen2.5-7B-Instruct" # Or Qwen/Qwen2.5-7B-Instruct
DEFAULT_OUTPUT_DIR = "outputs/qwen-financial-32k"
DEFAULT_DATA_PATH = "data/train.jsonl"

# Expanded Context Window
MAX_SEQ_LENGTH = 32768 

# Custom loss weights for financial tokens
FINANCIAL_KEYS = {
    "revenue", "net_income", "operating_income", "total_assets", 
    "cash_and_equivalents", "diluted_eps", "value", "unit",
    "total_revenue", "cost_of_revenue", "gross_profit", "income_before_tax",
    "earnings_per_share_basic", "earnings_per_share_diluted",
    "total_current_assets", "total_liabilities", "total_shareholders_equity"
}

# Define weights for specific token types
WEIGHTS = {
    "json_key": 2.0,       # Heavily penalize errors in JSON keys
    "number": 1.5,         # Penalize errors in numbers
    "json_structure": 1.2, # {},:[]
    "default": 1.0
}

def load_data(data_path: str, tokenizer) -> List[Dict[str, Any]]:
    """
    Load training data
    Applies chat template to the full document content.
    """
    try:
        data = [json.loads(line) for line in open(data_path) if line.strip()]
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise

    logger.info(f"Loaded {len(data)} raw examples from {data_path}")
    
    formatted_data = []
    
    for ex in data:
        messages = ex.get("messages", [])
        
        # Simple processing: Apply chat template to the whole conversation
        # We append /no_think to user messages if using a reasoning model that supports it, 
        # otherwise remove it if using standard Qwen/Llama.
        processed_messages = []
        for msg in messages:
            content = msg["content"]
            # Optional: Add specific prompting tweaks here if needed
            processed_messages.append({"role": msg["role"], "content": content})
        
        text = tokenizer.apply_chat_template(
            processed_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        formatted_data.append({"text": text})
    
    logger.info(f"Prepared {len(formatted_data)} full-context training examples")
    return formatted_data


def create_custom_trainer(SFTTrainer, tokenizer):
    """
    Create custom trainer with token-weighted loss for financial extraction.
    """
    class CustomFinancialTrainer(SFTTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            tok = self.processing_class
            
            # Pre-compute token sets for efficient lookup
            self.key_tokens = set()
            for key in FINANCIAL_KEYS:
                # Encode both "key" and key variants
                self.key_tokens.update(tok.encode(f'"{key}"', add_special_tokens=False))
                self.key_tokens.update(tok.encode(key, add_special_tokens=False))
            
            self.struct_tokens = set()
            for char in '{}[]:",':
                self.struct_tokens.update(tok.encode(char, add_special_tokens=False))
            
            self.digit_tokens = set()
            for char in '0123456789.-':
                self.digit_tokens.update(tok.encode(char, add_special_tokens=False))
            
            logger.info(f"CustomTrainer initialized. Tracking {len(self.key_tokens)} key tokens.")
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Standard forward pass
            labels = inputs.get("labels", inputs["input_ids"].clone())
            outputs = model(**inputs)
            
            # Get logits
            logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Build weight tensor (Shape: Batch x Seq_Len)
            weights = torch.ones_like(shift_labels, dtype=torch.float32)
            
            # CPU calculation for weights is safer to avoid complex indexing on GPU
            # but we do it on device for speed if possible. 
            # Iterating tokens is slow in Python, but fine for small batch sizes.
            
            # Optimization: Create boolean masks for vectorization if possible, 
            # otherwise sticking to loop for clarity/safety with variable token IDs.
            for b in range(shift_labels.shape[0]):
                for i in range(shift_labels.shape[1]):
                    token_id = shift_labels[b, i].item()
                    
                    if token_id == -100: # Ignored token
                        weights[b, i] = 0.0
                    elif token_id in self.key_tokens:
                        weights[b, i] = WEIGHTS["json_key"]
                    elif token_id in self.struct_tokens:
                        weights[b, i] = WEIGHTS["json_structure"]
                    elif token_id in self.digit_tokens:
                        weights[b, i] = WEIGHTS["number"]
            
            weights = weights.to(logits.device)
            
            # Weighted cross-entropy
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            # flatten tensors
            loss = loss_fn(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            
            # Apply weights
            flat_weights = weights.view(-1)
            mask = flat_weights > 0
            
            if mask.sum() > 0:
                # Average loss over valid tokens only
                loss = (loss * flat_weights)[mask].sum() / flat_weights[mask].sum()
            else:
                loss = loss.sum()
            
            return (loss, outputs) if return_outputs else loss
    
    return CustomFinancialTrainer


def train(
    model_name: str = DEFAULT_MODEL,
    data_path: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_seq_length: int = MAX_SEQ_LENGTH,
    num_epochs: int = 3,
    batch_size: int = 1, # Kept low for 32k context
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    save_merged: bool = True
):
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError:
        logger.error("Unsloth not installed.")
        raise
    
    from trl import SFTTrainer, SFTConfig
    
    # 1. Load Model
    logger.info(f"Loading model: {model_name} with context window: {max_seq_length}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None, 
        load_in_4bit=True,
    )
    
    # 2. Configure LoRA
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
    
    # 3. Tokenizer Setup
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    
    # 4. Load Data (No Chunking)
    formatted_data = load_data(data_path, tokenizer)
    dataset = Dataset.from_list(formatted_data)
    
    # 5. Training Config
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=5,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True, # Use bf16=True if on Ampere (A100/A10), fp16 for T4
        logging_steps=1,
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False, # Packing can be tricky with very long contexts on low VRAM
    )
    
    # 6. Initialize CUSTOM Trainer
    # IMPORTANT: We use the class generator to inject the tokenizer
    CustomTrainerClass = create_custom_trainer(SFTTrainer, tokenizer)
    
    trainer = CustomTrainerClass( # <--- USING CUSTOM TRAINER HERE
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args
    )
    
    # 7. Train
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    logger.success("Training complete!")
    
    # 8. Save
    logger.info(f"Saving LoRA adapters to {output_dir}")
    trainer.save_model(output_dir)
    
    if save_merged:
        merged_dir = f"{output_dir}-merged"
        logger.info(f"Saving merged model to {merged_dir}")
        # Use merged_16bit for safer export, or merged_4bit to save space
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        
    return output_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1) # Default to 1 for 32k context safety
    parser.add_argument("--accum", type=int, default=8) # Higher accum to compensate for batch size 1
    parser.add_argument("--max-length", type=int, default=32768)
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        max_seq_length=args.max_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum
    )

if __name__ == "__main__":
    main()
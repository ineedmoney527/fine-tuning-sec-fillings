"""
Data loading utilities for fine-tuning.

This module handles loading and formatting the training data
from JSONL files into HuggingFace Dataset format.
"""

import json
from pathlib import Path
from typing import Optional
from datasets import Dataset
from loguru import logger


# Qwen3 ChatML template
QWEN3_CHAT_TEMPLATE = """{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system
You are a helpful assistant.<|im_end|>
' }}{% endif %}{{'<|im_start|>' + message['role'] + '
' + message['content']}}{% if loop.last %}{{ '<|im_end|>' }}{% else %}{{ '<|im_end|>
' }}{% endif %}{% endfor %}"""


def load_jsonl(filepath: str | Path) -> list[dict]:
    """Load JSONL file as list of dicts."""
    filepath = Path(filepath)
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} examples from {filepath}")
    return data


def format_for_training(example: dict) -> dict:
    """
    Format a single example for SFT training.
    
    Converts ChatML-style messages to the format expected by trl's SFTTrainer.
    
    Args:
        example: Dict with 'messages' key containing conversation turns
        
    Returns:
        Dict with 'messages' in the format expected by trainer
    """
    return {"messages": example["messages"]}


def load_training_data(
    filepath: str | Path = "data/train.jsonl",
    train_ratio: float = 1.0,
    seed: int = 42
) -> Dataset | tuple[Dataset, Dataset]:
    """
    Load training data from JSONL file into HuggingFace Dataset.
    
    Args:
        filepath: Path to JSONL file
        train_ratio: Ratio of data for training (1.0 = no split)
        seed: Random seed for reproducibility
        
    Returns:
        Dataset or tuple of (train_dataset, eval_dataset) if train_ratio < 1.0
    """
    data = load_jsonl(filepath)
    
    # Format for training
    formatted_data = [format_for_training(ex) for ex in data]
    
    # Create HF Dataset
    dataset = Dataset.from_list(formatted_data)
    
    if train_ratio >= 1.0:
        logger.info(f"Created dataset with {len(dataset)} examples (no split)")
        return dataset
    
    # Split dataset
    split = dataset.train_test_split(
        train_size=train_ratio,
        seed=seed
    )
    logger.info(f"Split dataset: {len(split['train'])} train, {len(split['test'])} eval")
    return split['train'], split['test']


def get_system_prompt() -> str:
    """Get the system prompt used for financial extraction."""
    return """You are a financial data extraction engine. You will receive a Markdown formatted financial document (SEC 10-K).

Your goal is to extract specific metrics into a flat JSON object.

### Extraction Rules:
1. **Normalization:** Convert all numbers to their full integer value.
   - Example: If the table header says "(In millions)" and the value is "14,527", output `14527000000`.
   - Example: If the value is "5.61" (EPS), keep it as `5.61`.
2. **Missing Data:** If a specific metric is not explicitly stated, return `null`. Do not calculate fields yourself (e.g., do not subtract expenses from revenue to derive operating income). Extract only what is written.
3. **Negative Values:** Return negative numbers as standard JSON integers (e.g., -500), not strings with parentheses.

### Target Schema:
{
  "revenue": <int or null>,             // Total Net Sales / Revenue
  "operating_income": <int or null>,    // Operating Income / Loss
  "net_income": <int or null>,          // Net Income / Loss
  "total_assets": <int or null>,        // Total Assets
  "cash_and_equivalents": <int or null>,// Ending Cash & Cash Equivalents
  "diluted_eps": <float or null>        // Diluted Earnings Per Share
}
"""

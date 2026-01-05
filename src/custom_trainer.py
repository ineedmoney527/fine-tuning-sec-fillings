"""
Custom SFTTrainer with token-weighted loss for financial extraction.

This module provides a CustomFinancialTrainer that applies higher loss weights
to tokens that are critical for structured financial extraction:
- JSON keys (e.g., "revenue", "net_income")
- Numerical tokens
- JSON structure tokens ({, }, :, ,)

Usage:
    from src.custom_trainer import CustomFinancialTrainer
    
    trainer = CustomFinancialTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
"""

import re
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union, Any
from trl import SFTTrainer


# Financial metric keys to boost
FINANCIAL_KEYS = {
    "revenue", "total_revenue", "net_income", "operating_income",
    "total_assets", "cash_and_equivalents", "cash_and_cash_equivalents",
    "diluted_eps", "earnings_per_share_diluted", "earnings_per_share_basic",
    "gross_profit", "cost_of_revenue", "total_liabilities",
    "total_shareholders_equity", "accounts_payable", "accounts_receivable_net",
    "inventories", "total_current_assets", "total_current_liabilities",
    "property_plant_equipment_net", "net_cash_from_operating_activities",
    "net_cash_from_investing_activities", "net_cash_from_financing_activities",
    "income_before_tax", "weighted_average_shares_diluted",
    "value", "unit"  # Also boost nested keys
}

# Token weight configuration
TOKEN_WEIGHTS = {
    "json_key": 2.0,      # Keys like "revenue":
    "number": 1.5,        # Numeric values
    "json_structure": 1.2, # {, }, [, ], :, ,
    "default": 1.0,       # Regular tokens
}


class CustomFinancialTrainer(SFTTrainer):
    """
    Custom SFTTrainer that applies token-weighted loss for financial extraction.
    
    Higher weights are applied to:
    - JSON keys (schema fields)
    - Numerical tokens (extracted values)
    - JSON structure tokens (validity)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_token_weight_cache()
    
    def _build_token_weight_cache(self):
        """Pre-compute token IDs for weighted categories."""
        tokenizer = self.processing_class  # SFTTrainer uses processing_class
        
        # Cache for JSON key token sequences
        self.key_token_ids = set()
        for key in FINANCIAL_KEYS:
            # Tokenize the key with quotes (as it appears in JSON)
            key_with_quotes = f'"{key}"'
            tokens = tokenizer.encode(key_with_quotes, add_special_tokens=False)
            self.key_token_ids.update(tokens)
            # Also add without quotes
            tokens = tokenizer.encode(key, add_special_tokens=False)
            self.key_token_ids.update(tokens)
        
        # JSON structure tokens
        self.structure_tokens = set()
        for char in ['{', '}', '[', ']', ':', ',', '"']:
            tokens = tokenizer.encode(char, add_special_tokens=False)
            self.structure_tokens.update(tokens)
        
        # Number pattern - we'll detect these dynamically
        self.digit_tokens = set()
        for digit in '0123456789.-':
            tokens = tokenizer.encode(digit, add_special_tokens=False)
            self.digit_tokens.update(tokens)
    
    def _get_token_weights(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token weights based on token type.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Label token IDs [batch_size, seq_len], -100 for masked
            
        Returns:
            Tensor of weights [batch_size, seq_len]
        """
        batch_size, seq_len = labels.shape
        weights = torch.ones_like(labels, dtype=torch.float32)
        
        # Iterate through batch
        for b in range(batch_size):
            for i in range(seq_len):
                label_id = labels[b, i].item()
                
                # Skip masked tokens (labels=-100 means don't compute loss)
                if label_id == -100:
                    weights[b, i] = 0.0
                    continue
                
                # Check token type and assign weight
                if label_id in self.key_token_ids:
                    weights[b, i] = TOKEN_WEIGHTS["json_key"]
                elif label_id in self.structure_tokens:
                    weights[b, i] = TOKEN_WEIGHTS["json_structure"]
                elif label_id in self.digit_tokens:
                    weights[b, i] = TOKEN_WEIGHTS["number"]
                else:
                    weights[b, i] = TOKEN_WEIGHTS["default"]
        
        return weights
    
    def compute_loss(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        """
        Compute weighted cross-entropy loss.
        
        Overrides SFTTrainer.compute_loss to apply token-level weighting.
        """
        # Get labels
        labels = inputs.get("labels")
        if labels is None:
            labels = inputs.get("input_ids").clone()
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift for causal LM (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Get token weights
        weights = self._get_token_weights(
            inputs.get("input_ids")[..., :-1],
            shift_labels
        ).to(shift_logits.device)
        
        # Compute per-token cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Reshape for loss computation
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_weights = weights.view(-1)
        
        # Compute unweighted loss per token
        per_token_loss = loss_fct(flat_logits, flat_labels)
        
        # Apply weights
        weighted_loss = per_token_loss * flat_weights
        
        # Mean over non-zero weights (non-masked tokens)
        non_zero_mask = flat_weights > 0
        if non_zero_mask.sum() > 0:
            loss = weighted_loss[non_zero_mask].sum() / flat_weights[non_zero_mask].sum()
        else:
            loss = weighted_loss.sum()
        
        return (loss, outputs) if return_outputs else loss


class JSONValidityTrainer(CustomFinancialTrainer):
    """
    Extended trainer that adds JSON validity checking during evaluation.
    
    Note: This doesn't add validity loss during training (which would require
    generation), but logs JSON validity metrics during evaluation.
    """
    
    def __init__(self, *args, json_validity_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_validity_weight = json_validity_weight
        self.json_parse_errors = 0
        self.json_parse_total = 0
    
    def evaluate(self, *args, **kwargs):
        """Override evaluate to add JSON validity metrics."""
        # Reset counters
        self.json_parse_errors = 0
        self.json_parse_total = 0
        
        # Run standard evaluation
        metrics = super().evaluate(*args, **kwargs)
        
        # Add JSON validity rate if we tracked any
        if self.json_parse_total > 0:
            validity_rate = 1.0 - (self.json_parse_errors / self.json_parse_total)
            metrics["eval_json_validity_rate"] = validity_rate
        
        return metrics

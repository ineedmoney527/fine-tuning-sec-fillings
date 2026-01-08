"""
Inference script for financial extraction using fine-tuned Qwen3 model.

Uses dynamic prompt generation like in generate_dataset.py to enable
flexible extraction of various financial metrics.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, List

from loguru import logger


# =================CONFIGURATION=================
DEFAULT_MODEL_PATH = "outputs/qwen3-8b-financial-lora"
MAX_NEW_TOKENS = 2048

# Dynamic Metric Pool (same as generate_dataset.py)
METRIC_POOL = [
    # --- Income Statement (Consolidated Statement of Operations) ---
    "total_revenue",                # Look for: "Net sales", "Total net sales", "Revenue"
    "cost_of_revenue",              # Look for: "Cost of sales", "Cost of revenue", "Cost of goods sold"
    "gross_profit",                 # Explicit line item (NOT Gross Margin %)
    "operating_income",             # Look for: "Operating income", "Income from operations"
    "income_before_tax",            # Look for: "Income before provision for income taxes", "Earnings before taxes"
    "net_income",                   # Look for: "Net income", "Net earnings"
    "earnings_per_share_basic",     # Look for: "Basic earnings per share"
    "earnings_per_share_diluted",   # Look for: "Diluted earnings per share"
    "weighted_average_shares_diluted", # Look for: "Weighted-average shares - diluted"
    
    # --- Balance Sheet (Consolidated Balance Sheets) ---
    "cash_and_cash_equivalents",    # Ending balance, not including restricted usually
    "accounts_receivable_net",      # Look for: "Accounts receivable, net"
    "inventories",                  # Look for: "Inventories"
    "total_current_assets",         # Explicit total line
    "property_plant_equipment_net", # Look for: "Property, plant and equipment, net"
    "total_assets",                 # Bottom line asset number
    "accounts_payable",             # Explicit line
    "total_current_liabilities",    # Explicit total line
    "total_liabilities",            # Explicit total line
    "total_shareholders_equity",    # Look for: "Total shareholders' equity", "Total equity"

    # --- Cash Flow Statement (Consolidated Statement of Cash Flows) ---
    # These are highly stable "Category Totals" that rarely vary in naming
    "net_cash_from_operating_activities", 
    "net_cash_from_investing_activities",
    "net_cash_from_financing_activities"
]

# Base system prompt template
BASE_SYSTEM_PROMPT = """You are a financial data extraction engine. You will receive a Markdown formatted financial document (SEC 10-K).

Your task is to extract specific financial line items into a structured JSON object.

### Extraction Scope
Focus ONLY on the primary financial tables:
1. **Consolidated Statements of Operations** (Income Statement)
2. **Consolidated Balance Sheets**
3. **Consolidated Statements of Cash Flows**

### Extraction Rules
0. Do NOT perform any arithmetic operations or infer with your own knowledge, if the item does not exist in the report.
1. **Literal Extraction:** Just extract the numeric value exactly as it appears in the table row.
   - Example: If 'Total Liabilities' is not explicitly written as a line item, do NOT calculate it from Assets - Equity. Return null
2. **Synonym Matching:** Map the requested JSON keys to standard financial reporting terms if the original ones not founded:
   - Example: `total_revenue`: Look for "Net sales", "Total net sales", "Revenue", "Total revenues".
   - Example: `cost_of_revenue`: Look for "Cost of sales", "Cost of goods sold", "Cost of revenue".
3. **Unit Extraction:** You MUST check the table title or top row for scale indicators like '(In millions)' or '(In thousands)'. Extract this exact text. Only return 'ones' if the table strictly implies absolute integers (like share counts)
4. **Negative Values:** Convert values in parentheses `(123)` to negative numbers `-123`.
5. **Strict Nulls:** If a field is not explicitly present as a line item, set the entire object to `null`.

### Target Schema
For each requested metric, output an object with `value` and `unit`.
{
  __DYNAMIC_SCHEMA_PLACEHOLDER__
}
"""


def generate_dynamic_prompt(selected_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generates a unique system prompt with a random subset of metrics.
    
    Args:
        selected_metrics: Optional list of specific metrics to use. 
                         If None, randomly selects 4-8 metrics from METRIC_POOL.
    
    Returns:
        Dict containing the 'prompt' string and the 'keys' list.
    """
    if selected_metrics is None:
        # Randomly select 4 to 8 metrics from the pool
        num_metrics = random.randint(4, 8)
        selected_metrics = random.sample(METRIC_POOL, num_metrics)
    
    # Build nested schema: "key": {"value": <int>, "unit": <str>}
    schema_lines = []
    for metric in selected_metrics:
        schema_lines.append(f'  "{metric}": {{"value": <number>, "unit": <string>}}')
    
    schema_str = ",\n".join(schema_lines)
    
    # Inject into base prompt
    full_prompt = BASE_SYSTEM_PROMPT.replace("__DYNAMIC_SCHEMA_PLACEHOLDER__", schema_str)
    
    return {
        "text": full_prompt,
        "keys": selected_metrics
    }


class FinancialExtractor:
    """
    Inference class for financial data extraction using fine-tuned model.
    """
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, use_mlx: bool = False):
        """
        Initialize the extractor with a fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model (merged or LoRA adapters)
            use_mlx: Whether to use MLX for inference (macOS optimized)
        """
        self.model_path = model_path
        self.use_mlx = use_mlx
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        if self.use_mlx:
            try:
                from mlx_lm import load, generate
                self._mlx_generate = generate
                logger.info(f"Loading model with MLX from {self.model_path}...")
                self.model, self.tokenizer = load(self.model_path)
                logger.success("Model loaded successfully with MLX!")
            except ImportError:
                logger.warning("MLX not available, falling back to transformers...")
                self.use_mlx = False
                self._load_transformers_model()
        else:
            self._load_transformers_model()
    
    def _load_transformers_model(self):
        """Load model using transformers library."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading model with transformers from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            logger.success("Model loaded successfully with transformers!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract(
        self, 
        document: str, 
        metrics: Optional[List[str]] = None,
        max_new_tokens: int = MAX_NEW_TOKENS
    ) -> Dict[str, Any]:
        """
        Extract financial data from a document.
        
        Args:
            document: Markdown-formatted financial document text
            metrics: Optional list of specific metrics to extract.
                    If None, uses random subset from METRIC_POOL.
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dict containing extracted financial data
        """
        # Generate dynamic prompt
        prompt_data = generate_dynamic_prompt(metrics)
        system_prompt = prompt_data["text"]
        target_keys = prompt_data["keys"]
        
        logger.debug(f"Extracting metrics: {target_keys}")
        
        # Build chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": document + " /no_think"}
        ]
        
        # Generate response
        if self.use_mlx:
            response = self._generate_mlx(messages, max_new_tokens)
        else:
            response = self._generate_transformers(messages, max_new_tokens)
        
        # Parse JSON response
        try:
            result = json.loads(response)
            return {
                "success": True,
                "data": result,
                "requested_keys": target_keys
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {
                "success": False,
                "raw_response": response,
                "error": str(e),
                "requested_keys": target_keys
            }
    
    def _generate_mlx(self, messages: List[Dict], max_new_tokens: int) -> str:
        """Generate using MLX."""
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        response = self._mlx_generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt,
            max_tokens=max_new_tokens,
            verbose=False
        )
        
        return response
    
    def _generate_transformers(self, messages: List[Dict], max_new_tokens: int) -> str:
        """Generate using transformers."""
        import torch
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        input_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        return response


def main():
    """Test inference with the first sample from train.jsonl."""
    logger.add("logs/inference.log", rotation="10 MB")
    
    # Load first sample from train.jsonl
    train_file = Path("data/train.jsonl")
    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        return
    
    with open(train_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        sample = json.loads(first_line)
    
    # Extract document content from the sample
    messages = sample.get("messages", [])
    document = None
    expected_output = None
    
    for msg in messages:
        if msg["role"] == "user":
            document = msg["content"]
        elif msg["role"] == "assistant":
            expected_output = msg["content"]
    
    if not document:
        logger.error("No user content found in sample")
        return
    
    logger.info("=" * 60)
    logger.info("Testing inference with first sample from train.jsonl")
    logger.info("=" * 60)
    
    # Use some common metrics for testing
    test_metrics = [
        "total_revenue",
        "operating_income", 
        "net_income",
        "total_assets",
        "cash_and_cash_equivalents",
        "earnings_per_share_diluted"
    ]
    
    logger.info(f"Document length: {len(document)} characters")
    logger.info(f"Target metrics: {test_metrics}")
    
    # Initialize extractor
    try:
        extractor = FinancialExtractor()
        
        # Run extraction
        logger.info("Running extraction...")
        result = extractor.extract(document, metrics=test_metrics)
        
        if result["success"]:
            logger.success("Extraction successful!")
            print("\n=== EXTRACTED DATA ===")
            print(json.dumps(result["data"], indent=2))
        else:
            logger.warning("Extraction returned raw response (not valid JSON)")
            print("\n=== RAW RESPONSE ===")
            print(result.get("raw_response", "No response"))
            print(f"\nError: {result.get('error', 'Unknown')}")
        
        # Show expected output for comparison
        if expected_output:
            print("\n=== EXPECTED OUTPUT (from training data) ===")
            try:
                expected_parsed = json.loads(expected_output)
                print(json.dumps(expected_parsed, indent=2))
            except:
                print(expected_output)
                
    except Exception as e:
        logger.error(f"Failed to run inference: {e}")
        logger.info("Make sure you have a fine-tuned model at: outputs/qwen3-8b-financial-lora")
        logger.info("Or specify a different model path")


if __name__ == "__main__":
    main()

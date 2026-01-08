import os
import json
import time
import random
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =================CONFIGURATION=================
API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/chat/completions"
INPUT_DIR = Path("data/processed")
OUTPUT_FILE = Path("data/train_dynamic.jsonl")
MODEL_NAME = "deepseek-chat"  # V3
MAX_RETRIES = 2
DELAY_BETWEEN_CALLS = 0.5

# =================CONFIGURATION=================
# REFINED "SAFE" METRIC POOL
# Criteria: Must be a distinct, explicit line item in >90% of US GAAP 10-Ks.
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

# Refined System Prompt
# Changes:
# 1. Added "Synonym Mapping" rules to help the model link "Sales" to "Revenue".
# 2. Added "Statement Isolation" to keep it focused on the Big 3 tables.
# 3. Explicitly forbade "Gross Margin" (percentage) vs "Gross Profit" (integer).
# ... (imports and metric pool remain the same) ...

# 1. UPDATED SYSTEM PROMPT
# Changed: Removed "multiply" rule. Added "Literal" rule. Added "Unit" rule.
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

def generate_dynamic_prompt() -> Dict[str, Any]:
    """
    Generates a unique system prompt with a random subset of metrics.
    Returns: Dict containing the 'prompt' string and the 'keys' list.
    """
    # Randomly select 4 to 8 metrics from the pool
    num_metrics = random.randint(4, 8)
    selected_metrics = random.sample(METRIC_POOL, num_metrics)
    
    # 2. UPDATED SCHEMA CONSTRUCTION
    # Build nested schema: "key": {"value": <int>, "unit": <str>}
    schema_lines = []
    for metric in selected_metrics:
        # We explicitly describe the shape we want for each key
        schema_lines.append(f'  "{metric}": {{"value": <number>, "unit": <string>}}')
    
    schema_str = ",\n".join(schema_lines)
    
    # Inject into base prompt
    full_prompt = BASE_SYSTEM_PROMPT.replace("__DYNAMIC_SCHEMA_PLACEHOLDER__", schema_str)
    
    return {
        "text": full_prompt,
        "keys": selected_metrics
    }

# ... (Rest of call_llm, validate_extraction, and main remain the same) ...

def call_llm(content: str, system_prompt: str) -> Optional[Dict[str, Any]]:
    if not API_KEY:
        logger.error("DEEPSEEK_API_KEY missing.")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # Safety Check: Warn if content is massive
    est_tokens = len(content) / 4
    if est_tokens > 30000:
        logger.warning(f"Content is huge (~{int(est_tokens)} tokens). Ensure context length support.")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            
            result = response.json()
            content_str = result['choices'][0]['message']['content']
            
            try:
                data = json.loads(content_str)
                return data
            except json.JSONDecodeError:
                logger.error(f"JSON Decode Error on attempt {attempt+1}")
                continue
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 * (attempt + 1))
            
    return None

def validate_extraction(data: Dict, expected_keys: List[str]) -> bool:
    """Ensure the LLM returned the exact keys we asked for."""
    if not data:
        return False
        
    extracted_keys = set(data.keys())
    required_keys = set(expected_keys)
    
    # Check if we got all keys (it's okay if LLM returns nulls, but keys must exist)
    missing = required_keys - extracted_keys
    if missing:
        logger.warning(f"Validation Warning: LLM missed keys: {missing}")
        # We can still accept partials, but ideally we want strict adherence
        return False
    return True

def main():
    logger.add("logs/gen_dynamic_dataset.log", rotation="10 MB")
    
    if not INPUT_DIR.exists():
        logger.error(f"Directory {INPUT_DIR} not found.")
        return

    # processed_files = list(INPUT_DIR.glob("ADBE_2022.md"))
    processed_files = list(INPUT_DIR.glob("*.md"))
    logger.info(f"Found {len(processed_files)} files.")
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        for file_path in processed_files:
            logger.info(f"Processing {file_path.name}...")
            
            try:
                text = file_path.read_text(encoding="utf-8")
                
                if len(text) < 500:
                    logger.warning(f"Skipping {file_path.name} - too short.")
                    continue

                # 1. Generate Dynamic Prompt
                prompt_data = generate_dynamic_prompt()
                current_system_prompt = prompt_data["text"]
                target_keys = prompt_data["keys"]
                
                logger.debug(f"Requesting keys: {target_keys}")

                # 2. Call LLM
                extracted_data = call_llm(text, current_system_prompt)
                
                # Check if extraction succeeded
                if not extracted_data:
                    logger.error(f"Extraction returned no data: {file_path.name}")
                    continue
                
                # Check if the extraction resulted in ALL nulls
                non_null_count = sum(1 for v in extracted_data.values() if v is not None)

                # 2. Logic: Drop if completely empty (or save to a separate 'negatives' file)
                if non_null_count == 0:
                    logger.warning(f"Skipping {file_path.name}: All extracted values are NULL (Empty Sample).")
                    continue # Do not write to train.jsonl

                # 3. (Optional) Keep a small percentage of negatives for robustness
                if non_null_count == 0 and random.random() > 0.1: continue
                
                # 3. Save if valid
                if extracted_data and validate_extraction(extracted_data, target_keys):
                    training_example = {
                        "messages": [
                            {"role": "system", "content": current_system_prompt},
                            {"role": "user", "content": text},
                            {"role": "assistant", "content": json.dumps(extracted_data)}
                        ]
                    }
                    
                    f_out.write(json.dumps(training_example) + "\n")
                    f_out.flush()
                    logger.success(f"Saved: {file_path.name}")
                else:
                    logger.error(f"Failed or Invalid extraction: {file_path.name}")
            
            except Exception as e:
                logger.exception(f"Critical error on {file_path.name}")
            
            time.sleep(DELAY_BETWEEN_CALLS)

if __name__ == "__main__":
    main()
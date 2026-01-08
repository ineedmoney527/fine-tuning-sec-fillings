"""
Inference script for Qwen3-8B Financial Extraction.
Loads the fine-tuned model and runs a test on the first example from the training set.
"""

import json
import torch
import argparse
from pathlib import Path
from loguru import logger
from unsloth import FastLanguageModel

# =================CONFIGURATION=================
DEFAULT_MODEL_PATH = "outputs/qwen-financial-32k" # Path to your saved LoRA or merged model
DATA_PATH = "data/train.jsonl"
MAX_SEQ_LENGTH = 32768
LOAD_IN_4BIT = True

def load_model(model_path):
    """
    Load the model and tokenizer. 
    Unsloth handles loading both base model + adapters automatically if pointing to adapter dir.
    """
    logger.info(f"Loading model from: {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def get_first_training_sample(file_path):
    """Reads the first line of the JSONL file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line:
                return None
            return json.loads(first_line)
    except FileNotFoundError:
        logger.error(f"Could not find data file: {file_path}")
        return None

def run_inference(model, tokenizer, sample):
    """
    Reconstructs the prompt from the sample and generates output.
    """
    messages = sample["messages"]
    
    # Extract the parts we need
    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
    expected_output = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    
    logger.info("Preparing prompt...")
    
    # Construct the chat template for inference
    # Note: We do NOT include the assistant's response here, because we want the model to generate it.
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # Apply template with add_generation_prompt=True to signal the model to start speaking
    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    logger.info("Generating response (this may take a moment for long docs)...")
    
    # Generate
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=1024,      # Enough for a JSON object
        use_cache=True,
        temperature=0.1,          # Low temp for factual extraction
        top_p=0.9
    )
    
    # Decode only the new tokens (the response)
    # outputs[0] contains the full sequence (prompt + response). We slice to get just the response.
    response_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    
    return response_text, expected_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to fine-tuned model/adapter")
    parser.add_argument("--data", default=DATA_PATH, help="Path to training data")
    args = parser.parse_args()

    # 1. Load Data
    sample = get_first_training_sample(args.data)
    if not sample:
        logger.error("No training data found to test.")
        return

    # 2. Load Model
    model, tokenizer = load_model(args.model)

    # 3. Run Inference
    logger.info("--- INPUT METADATA ---")
    # Parse the system prompt to see what keys were requested in this specific sample
    sys_prompt = sample['messages'][0]['content']
    requested_keys = [line.split('"')[1] for line in sys_prompt.split('\n') if '": {"value":' in line]
    logger.info(f"Requested Keys in this sample: {requested_keys}")

    generated_json_str, expected_json_str = run_inference(model, tokenizer, sample)

    # 4. Compare Results
    print("\n" + "="*50)
    print("🤖 MODEL PREDICTION:")
    print("="*50)
    print(generated_json_str)
    
    print("\n" + "="*50)
    print("✅ GROUND TRUTH:")
    print("="*50)
    print(expected_json_str)

    # Optional: Basic JSON validation
    try:
        gen_json = json.loads(generated_json_str)
        logger.success("Prediction is valid JSON.")
    except json.JSONDecodeError:
        logger.error("Prediction is NOT valid JSON.")

if __name__ == "__main__":
    main()
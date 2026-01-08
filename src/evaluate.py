"""
Evaluate fine-tuned model on training examples.

This script loads the fine-tuned LoRA adapter and evaluates it against
real examples from the training data to verify extraction accuracy.

Usage:
    # Using MLX on Apple Silicon (local):
    python src/evaluate.py --model_path outputs/qwen3-8b-financial-lora
    
    # Using Unsloth/CUDA:
    python src/evaluate.py --model_path outputs/qwen3-8b-financial-lora --backend unsloth
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


def load_test_examples(data_path: str = "data/train.jsonl", num_examples: int = 3) -> list:
    """Load N examples from training data for evaluation."""
    examples = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            if line.strip():
                examples.append(json.loads(line))
    logger.info(f"Loaded {len(examples)} examples for evaluation")
    return examples


def extract_expected_output(example: Dict) -> Dict[str, Any]:
    """Extract the expected JSON output from a training example."""
    for msg in example["messages"]:
        if msg["role"] == "assistant":
            return json.loads(msg["content"])
    return {}


def extract_user_content(example: Dict) -> str:
    """Extract the user message (10-K content) from a training example."""
    for msg in example["messages"]:
        if msg["role"] == "user":
            return msg["content"]
    return ""


def extract_system_prompt(example: Dict) -> str:
    """Extract the system prompt from a training example."""
    for msg in example["messages"]:
        if msg["role"] == "system":
            return msg["content"]
    return ""


def compare_outputs(expected: Dict, actual: Dict) -> Dict[str, Any]:
    """
    Compare expected vs actual extraction results.
    
    Returns:
        Dict with comparison metrics
    """
    results = {
        "total_keys": 0,
        "matched_keys": 0,
        "value_matches": 0,
        "value_mismatches": [],
        "missing_keys": [],
        "extra_keys": [],
    }
    
    expected_keys = set(expected.keys())
    actual_keys = set(actual.keys())
    
    results["total_keys"] = len(expected_keys)
    results["matched_keys"] = len(expected_keys & actual_keys)
    results["missing_keys"] = list(expected_keys - actual_keys)
    results["extra_keys"] = list(actual_keys - expected_keys)
    
    # Compare values for matching keys
    for key in expected_keys & actual_keys:
        exp_val = expected[key]
        act_val = actual.get(key)
        
        # Handle nested value/unit structure
        if isinstance(exp_val, dict) and isinstance(act_val, dict):
            exp_v = exp_val.get("value")
            act_v = act_val.get("value")
        else:
            exp_v = exp_val
            act_v = act_val
        
        if exp_v == act_v:
            results["value_matches"] += 1
        else:
            results["value_mismatches"].append({
                "key": key,
                "expected": exp_v,
                "actual": act_v
            })
    
    # Calculate accuracy
    if results["total_keys"] > 0:
        results["key_accuracy"] = results["matched_keys"] / results["total_keys"]
        if results["matched_keys"] > 0:
            results["value_accuracy"] = results["value_matches"] / results["matched_keys"]
        else:
            results["value_accuracy"] = 0.0
    else:
        results["key_accuracy"] = 0.0
        results["value_accuracy"] = 0.0
    
    return results


def run_inference_mlx(model_path: str, messages: list) -> str:
    """Run inference using MLX-LM (for Apple Silicon).
    
    For LoRA adapters, loads the base Qwen3 model and applies the adapter.
    """
    try:
        from mlx_lm import load, generate
        from pathlib import Path
        
        model_path = Path(model_path)
        
        # Check if this is a LoRA adapter (has adapter_config.json but no config.json)
        is_lora_adapter = (
            (model_path / "adapter_config.json").exists() and
            not (model_path / "config.json").exists()
        )
        
        if is_lora_adapter:
            # Load adapter config to get base model name
            import json
            with open(model_path / "adapter_config.json", 'r') as f:
                adapter_config = json.load(f)
            
            base_model = adapter_config.get("base_model_name_or_path", "Qwen/Qwen3-8B-Instruct")
            logger.info(f"Loading base model: {base_model}")
            logger.info(f"Applying LoRA adapter from: {model_path}")
            
            # Load base model with adapter
            model, tokenizer = load(base_model, adapter_path=str(model_path,))
        else:
            # Load merged model directly
            logger.info(f"Loading merged model: {model_path}")
            model, tokenizer = load(str(model_path))
        
        # Format messages for Qwen
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt,
        )
        return response
    except ImportError:
        logger.error("mlx-lm not installed. Install with: pip install mlx-lm")
        raise


def run_inference_unsloth(model_path: str, messages: list) -> str:
    """Run inference using Unsloth (for CUDA GPUs)."""
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=8192,
            load_in_4bit=True,
        )
        
        FastLanguageModel.for_inference(model)
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            # temperature=0.1,
        )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response
    except ImportError:
        logger.error("unsloth not installed")
        raise


def parse_json_response(response: str) -> Optional[Dict]:
    """Parse JSON from model response."""
    import re
    
    # Try direct parse
    try:
        return json.loads(response.strip())
    except:
        pass
    
    # Try to find JSON in markdown code blocks
    patterns = [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```', r'\{[\s\S]*\}']
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str.strip())
            except:
                continue
    
    return None


def evaluate(
    model_path: str,
    data_path: str = "data/train.jsonl",
    num_examples: int = 3,
    backend: str = "mlx"
) -> Dict[str, Any]:
    """
    Evaluate the fine-tuned model on training examples.
    
    Args:
        model_path: Path to LoRA adapter or merged model
        data_path: Path to training JSONL
        num_examples: Number of examples to evaluate
        backend: "mlx" or "unsloth"
        
    Returns:
        Evaluation results dict
    """
    logger.info(f"Evaluating model: {model_path}")
    logger.info(f"Backend: {backend}")
    
    # Load examples
    examples = load_test_examples(data_path, num_examples)
    
    # Select inference function
    if backend == "mlx":
        run_inference = run_inference_mlx
    else:
        run_inference = run_inference_unsloth
    
    all_results = []
    
    for i, example in enumerate(examples):
        logger.info(f"\n{'='*60}")
        logger.info(f"Example {i+1}/{len(examples)}")
        logger.info(f"{'='*60}")
        
        # Extract components
        system_prompt = extract_system_prompt(example)
        user_content = extract_user_content(example)
        expected = extract_expected_output(example)
        
        logger.info(f"Expected keys: {list(expected.keys())}")
        
        # Build messages with /no_think
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content + " /no_think"}
        ]
        
        # Run inference
        logger.info("Running inference...")
        try:
            response = run_inference(model_path, messages)
            logger.info(f"Raw response (first 500 chars): {response[:500]}")
            
            # Parse response
            actual = parse_json_response(response)
            
            if actual is None:
                logger.error("Failed to parse JSON from response")
                result = {
                    "example_id": i,
                    "success": False,
                    "error": "JSON parse failed",
                    "raw_response": response[:1000]
                }
            else:
                # Compare
                comparison = compare_outputs(expected, actual)
                result = {
                    "example_id": i,
                    "success": True,
                    "expected": expected,
                    "actual": actual,
                    "comparison": comparison
                }
                
                logger.info(f"Key accuracy: {comparison['key_accuracy']:.1%}")
                logger.info(f"Value accuracy: {comparison['value_accuracy']:.1%}")
                
                if comparison["value_mismatches"]:
                    logger.warning(f"Mismatches: {comparison['value_mismatches']}")
                    
        except Exception as e:
            logger.exception(f"Inference failed: {e}")
            result = {
                "example_id": i,
                "success": False,
                "error": str(e)
            }
        
        all_results.append(result)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    
    successful = [r for r in all_results if r.get("success")]
    logger.info(f"Successful extractions: {len(successful)}/{len(all_results)}")
    
    if successful:
        avg_key_acc = sum(r["comparison"]["key_accuracy"] for r in successful) / len(successful)
        avg_val_acc = sum(r["comparison"]["value_accuracy"] for r in successful) / len(successful)
        logger.info(f"Average key accuracy: {avg_key_acc:.1%}")
        logger.info(f"Average value accuracy: {avg_val_acc:.1%}")
    
    return {
        "model_path": model_path,
        "num_examples": len(examples),
        "results": all_results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned financial extraction model")
    parser.add_argument("--model_path", type=str, default="outputs/qwen3-8b-financial-lora",
                        help="Path to fine-tuned model")
    parser.add_argument("--data_path", type=str, default="data/train.jsonl",
                        help="Path to evaluation data")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of examples to evaluate")
    parser.add_argument("--backend", type=str, default="mlx", choices=["mlx", "unsloth"],
                        help="Inference backend")
    parser.add_argument("--output", type=str, default="logs/eval_results.json",
                        help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/evaluate_{time}.log", rotation="10 MB")
    
    results = evaluate(
        model_path=args.model_path,
        data_path=args.data_path,
        num_examples=args.num_examples,
        backend=args.backend
    )
    
    # Always save detailed results
    output_path = args.output
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Detailed results saved to {output_path}")
    
    # Also save human-readable comparison
    comparison_path = output_path.replace('.json', '_comparison.txt')
    with open(comparison_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL OUTPUT vs GROUND TRUTH COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        for r in results["results"]:
            f.write(f"Example {r['example_id'] + 1}\n")
            f.write("-" * 40 + "\n")
            
            if r.get("success"):
                f.write("STATUS: SUCCESS\n\n")
                f.write("GROUND TRUTH:\n")
                f.write(json.dumps(r["expected"], indent=2) + "\n\n")
                f.write("MODEL OUTPUT:\n")
                f.write(json.dumps(r["actual"], indent=2) + "\n\n")
                
                comp = r["comparison"]
                f.write(f"Key Accuracy: {comp['key_accuracy']:.1%}\n")
                f.write(f"Value Accuracy: {comp['value_accuracy']:.1%}\n")
                
                if comp["value_mismatches"]:
                    f.write("\nMISMATCHES:\n")
                    for m in comp["value_mismatches"]:
                        f.write(f"  {m['key']}: expected={m['expected']}, got={m['actual']}\n")
            else:
                f.write(f"STATUS: FAILED\n")
                f.write(f"ERROR: {r.get('error', 'Unknown')}\n")
                if "raw_response" in r:
                    f.write(f"\nRAW RESPONSE:\n{r['raw_response'][:2000]}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    logger.info(f"Human-readable comparison saved to {comparison_path}")


if __name__ == "__main__":
    main()

"""
Evaluation module for fine-tuned financial extraction model.

Compares model predictions against ground truth from the training set.

Usage:
    python -m src.evaluate --model-path ./outputs/lora_adapters --num-samples 3
"""

import json
import argparse
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.schema import FinancialMetrics, parse_json_from_response


# 3 samples from training set for quick evaluation
EVAL_SAMPLES = [
    # Sample 1: Microsoft FY2023
    {
        "input": """ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
INCOME STATEMENTS
| (In millions, except per share amounts) |  |  |  |
| Year Ended June 30, | 2023 | 2022 | 2021 |
| Total revenue | 211,915 | 198,270 | 168,088 |
| Operating income | 88,523 | 83,383 | 69,916 |
| Net income | $72,361 | $72,738 | $61,271 |
| Diluted | $9.68 | $9.65 | $8.05 |

BALANCE SHEETS
| (In millions) |  |  |
| June 30, | 2023 | 2022 |
| Cash and cash equivalents | $34,704 | $13,931 |
| Total assets | $411,976 | $364,840 |""",
        "expected": {
            "revenue": 211915000000,
            "operating_income": 88523000000,
            "net_income": 72361000000,
            "total_assets": 411976000000,
            "cash_and_equivalents": 34704000000,
            "diluted_eps": 9.68,
        },
        "company": "Microsoft",
        "fiscal_year": "FY2023",
    },
    # Sample 2: Apple FY2024
    {
        "input": """Apple Inc. CONSOLIDATED STATEMENTS OF OPERATIONS (In millions, except per-share amounts)
| Years ended | September 28, 2024 | September 30, 2023 |
| Total net sales | 391,035 | 383,285 |
| Operating income | 123,216 | 114,301 |
| Net income | $93,736 | $96,995 |
| Diluted | $6.08 | $6.13 |

CONSOLIDATED BALANCE SHEETS (In millions)
| September 28, 2024 | September 30, 2023 |
| Cash and cash equivalents | $29,943 | $29,965 |
| Total assets | $364,980 | $352,583 |""",
        "expected": {
            "revenue": 391035000000,
            "operating_income": 123216000000,
            "net_income": 93736000000,
            "total_assets": 364980000000,
            "cash_and_equivalents": 29943000000,
            "diluted_eps": 6.08,
        },
        "company": "Apple",
        "fiscal_year": "FY2024",
    },
    # Sample 3: Tesla FY2022
    {
        "input": """Tesla, Inc. Consolidated Statements of Operations (in millions, except per share data)
| Year Ended December 31, | 2022 | 2021 | 2020 |
| Total revenues | 81,462 | 53,823 | 31,536 |
| Income from operations | 13,656 | 6,523 | 1,994 |
| Net income | 12,587 | 5,644 | 862 |
| Diluted | $3.62 | $1.63 | $0.21 |

Consolidated Balance Sheets (in millions)
| December 31, | 2022 | 2021 |
| Cash and cash equivalents | $16,253 | $17,576 |
| Total assets | $82,338 | $62,131 |""",
        "expected": {
            "revenue": 81462000000,
            "operating_income": 13656000000,
            "net_income": 12587000000,
            "total_assets": 82338000000,
            "cash_and_equivalents": 16253000000,
            "diluted_eps": 3.62,
        },
        "company": "Tesla",
        "fiscal_year": "FY2022",
    },
]


# System prompt from training
SYSTEM_PROMPT = """You are a financial data extraction engine. You will receive a Markdown formatted financial document (SEC 10-K).

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


def calculate_metrics(prediction: dict, expected: dict) -> dict:
    """
    Calculate evaluation metrics for a single prediction.
    
    Args:
        prediction: Model's predicted values
        expected: Ground truth values
        
    Returns:
        Dict with per-field accuracy and error metrics
    """
    fields = ["revenue", "operating_income", "net_income", 
              "total_assets", "cash_and_equivalents", "diluted_eps"]
    
    results = {
        "correct_fields": 0,
        "total_fields": len(fields),
        "field_details": {},
    }
    
    for field in fields:
        pred_val = prediction.get(field)
        exp_val = expected.get(field)
        
        # Check exact match (with tolerance for floats)
        if pred_val is None and exp_val is None:
            is_correct = True
            error_pct = 0.0
        elif pred_val is None or exp_val is None:
            is_correct = False
            error_pct = 100.0  # Missing value
        elif isinstance(exp_val, float):
            # Float comparison with tolerance
            is_correct = abs(pred_val - exp_val) < 0.01
            error_pct = abs(pred_val - exp_val) / abs(exp_val) * 100 if exp_val != 0 else 0
        else:
            # Integer comparison - exact match required
            is_correct = pred_val == exp_val
            error_pct = abs(pred_val - exp_val) / abs(exp_val) * 100 if exp_val != 0 else 0
        
        results["field_details"][field] = {
            "predicted": pred_val,
            "expected": exp_val,
            "correct": is_correct,
            "error_pct": round(error_pct, 2),
        }
        
        if is_correct:
            results["correct_fields"] += 1
    
    results["accuracy"] = results["correct_fields"] / results["total_fields"]
    return results


def evaluate_with_model(
    model_path: str,
    model_type: str = "lora",
    samples: list = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate model on samples.
    
    Args:
        model_path: Path to model/adapters
        model_type: Type of model (lora, merged, mlx)
        samples: List of samples to evaluate (default: EVAL_SAMPLES)
        verbose: Print detailed results
        
    Returns:
        Evaluation results dict
    """
    from src.inference import FinancialExtractor
    
    if samples is None:
        samples = EVAL_SAMPLES
    
    # Load model
    extractor = FinancialExtractor(
        model_path=model_path,
        model_type=model_type,
    )
    
    results = {
        "samples": [],
        "overall_accuracy": 0.0,
        "per_field_accuracy": {},
    }
    
    total_correct = 0
    total_fields = 0
    field_correct_counts = {}
    
    for i, sample in enumerate(samples):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Sample {i+1}: {sample['company']} {sample['fiscal_year']}")
            print(f"{'='*60}")
        
        # Get prediction
        raw_dict, validated = extractor.extract(sample["input"])
        
        # Use validated if available, else raw
        prediction = validated.model_dump() if validated else raw_dict
        
        # Calculate metrics
        metrics = calculate_metrics(prediction, sample["expected"])
        
        sample_result = {
            "company": sample["company"],
            "fiscal_year": sample["fiscal_year"],
            "prediction": prediction,
            "expected": sample["expected"],
            "metrics": metrics,
        }
        results["samples"].append(sample_result)
        
        # Aggregate
        total_correct += metrics["correct_fields"]
        total_fields += metrics["total_fields"]
        
        for field, details in metrics["field_details"].items():
            if field not in field_correct_counts:
                field_correct_counts[field] = {"correct": 0, "total": 0}
            field_correct_counts[field]["total"] += 1
            if details["correct"]:
                field_correct_counts[field]["correct"] += 1
        
        if verbose:
            print(f"Accuracy: {metrics['accuracy']*100:.1f}% ({metrics['correct_fields']}/{metrics['total_fields']} fields)")
            print("\nField Details:")
            for field, details in metrics["field_details"].items():
                status = "✓" if details["correct"] else "✗"
                print(f"  {status} {field}:")
                print(f"      Predicted: {details['predicted']}")
                print(f"      Expected:  {details['expected']}")
                if not details["correct"]:
                    print(f"      Error: {details['error_pct']}%")
    
    # Overall metrics
    results["overall_accuracy"] = total_correct / total_fields if total_fields > 0 else 0
    results["per_field_accuracy"] = {
        field: counts["correct"] / counts["total"]
        for field, counts in field_correct_counts.items()
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("OVERALL RESULTS")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {results['overall_accuracy']*100:.1f}% ({total_correct}/{total_fields} fields)")
        print("\nPer-field Accuracy:")
        for field, acc in results["per_field_accuracy"].items():
            print(f"  {field}: {acc*100:.1f}%")
    
    return results


def evaluate_without_model(verbose: bool = True) -> None:
    """
    Display evaluation samples without running inference.
    Useful when model is not available locally.
    """
    print("\n" + "="*60)
    print("EVALUATION SAMPLES (without model inference)")
    print("="*60)
    print("\nThese 3 samples can be used to evaluate your fine-tuned model:")
    
    for i, sample in enumerate(EVAL_SAMPLES, 1):
        print(f"\n{'─'*60}")
        print(f"Sample {i}: {sample['company']} {sample['fiscal_year']}")
        print(f"{'─'*60}")
        print("\nInput (truncated):")
        input_preview = sample["input"][:500] + "..." if len(sample["input"]) > 500 else sample["input"]
        print(input_preview)
        print("\nExpected Output:")
        print(json.dumps(sample["expected"], indent=2))
    
    print("\n" + "="*60)
    print("To run evaluation with your model, use:")
    print("  python -m src.evaluate --model-path ./outputs/lora_adapters")
    print("="*60)


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate financial extraction model")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to saved model/adapters (omit to just show samples)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lora", "merged", "mlx"],
        default="lora",
        help="Type of saved model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to evaluate (max 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for results JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    if not args.model_path:
        # Just show samples without running model
        evaluate_without_model()
        return 0
    
    # Select samples
    samples = EVAL_SAMPLES[:min(args.num_samples, len(EVAL_SAMPLES))]
    
    # Run evaluation
    results = evaluate_with_model(
        model_path=args.model_path,
        model_type=args.model_type,
        samples=samples,
        verbose=not args.quiet,
    )
    
    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())

"""
Inference module for fine-tuned financial extraction model.

Supports loading:
1. LoRA adapters with PEFT/Unsloth
2. Merged 16-bit model for direct inference
3. MLX-converted model for Apple Silicon

Usage:
    python -m src.inference --model-path ./outputs/lora_adapters --input "Your financial text..."
"""

import json
import argparse
from typing import Optional
from pathlib import Path

from pydantic import ValidationError

from src.schema import FinancialMetrics, parse_json_from_response


# System prompt used during training
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


class FinancialExtractor:
    """Wrapper for financial extraction inference."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "lora",  # lora, merged, or mlx
        max_new_tokens: int = 512,
        device: str = "auto",
    ):
        """
        Initialize the extractor with a fine-tuned model.
        
        Args:
            model_path: Path to saved model/adapters
            model_type: Type of saved model (lora, merged, mlx)
            max_new_tokens: Maximum tokens to generate
            device: Device to run on (auto, cuda, mps, cpu)
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.device = device
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model based on type."""
        if self.model_type == "mlx":
            self._load_mlx_model()
        elif self.model_type == "lora":
            self._load_lora_model()
        else:  # merged
            self._load_merged_model()
    
    def _load_lora_model(self):
        """Load LoRA adapters with Unsloth/PEFT."""
        try:
            from unsloth import FastLanguageModel
            
            # Load base model with adapters
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.model_path),
                max_seq_length=8192,
                dtype=None,
                load_in_4bit=True,
            )
            # Set to inference mode
            FastLanguageModel.for_inference(self.model)
            print(f"Loaded LoRA model from {self.model_path}")
            
        except ImportError:
            # Fallback to PEFT
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            # Read adapter config to get base model
            adapter_config_path = self.model_path / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path) as f:
                    config = json.load(f)
                base_model_name = config.get("base_model_name_or_path")
            else:
                base_model_name = "Qwen/Qwen3-8B-Instruct"
            
            print(f"Loading base model: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype="auto",
                device_map=self.device,
                trust_remote_code=True,
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )
            
            self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
            self.model.eval()
            print(f"Loaded LoRA adapters from {self.model_path}")
    
    def _load_merged_model(self):
        """Load merged 16-bit model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype="auto",
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Loaded merged model from {self.model_path}")
    
    def _load_mlx_model(self):
        """Load MLX-converted model for Apple Silicon."""
        try:
            import mlx_lm
            self.model, self.tokenizer = mlx_lm.load(str(self.model_path))
            print(f"Loaded MLX model from {self.model_path}")
        except ImportError:
            raise ImportError("mlx-lm is required for MLX models: pip install mlx-lm")
    
    def _generate_mlx(self, prompt: str) -> str:
        """Generate with MLX model."""
        import mlx_lm
        response = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            verbose=False,
        )
        return response
    
    def _generate_transformers(self, prompt: str) -> str:
        """Generate with Transformers/Unsloth model."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # Greedy for deterministic extraction
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response
    
    def extract(
        self,
        financial_text: str,
        validate: bool = True,
    ) -> tuple[dict, Optional[FinancialMetrics]]:
        """
        Extract financial metrics from text.
        
        Args:
            financial_text: Raw financial document text (markdown)
            validate: Whether to validate against Pydantic schema
            
        Returns:
            Tuple of (raw_dict, validated_metrics or None)
        """
        # Build chat messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": financial_text + " /no_think"},
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Generate
        if self.model_type == "mlx":
            response = self._generate_mlx(prompt)
        else:
            response = self._generate_transformers(prompt)
        
        # Parse JSON from response
        try:
            raw_dict = parse_json_from_response(response)
        except ValueError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw response: {response}")
            return {"error": str(e), "raw_response": response}, None
        
        # Optionally validate
        validated = None
        if validate:
            try:
                validated = FinancialMetrics.model_validate(raw_dict)
            except ValidationError as e:
                print(f"Validation warning: {e}")
        
        return raw_dict, validated


def main():
    """CLI entry point for inference."""
    parser = argparse.ArgumentParser(description="Financial extraction inference")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved model/adapters",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lora", "merged", "mlx"],
        default="lora",
        help="Type of saved model",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input financial text (or use --input-file)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to input file containing financial text",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: stdout)",
    )
    
    args = parser.parse_args()
    
    # Get input text
    if args.input:
        text = args.input
    elif args.input_file:
        with open(args.input_file) as f:
            text = f.read()
    else:
        print("Error: Must provide --input or --input-file")
        return 1
    
    # Initialize extractor
    extractor = FinancialExtractor(
        model_path=args.model_path,
        model_type=args.model_type,
    )
    
    # Extract
    raw_dict, validated = extractor.extract(text)
    
    # Output
    result = {
        "raw_extraction": raw_dict,
        "validated": validated.model_dump() if validated else None,
    }
    
    output_json = json.dumps(result, indent=2)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Results saved to {args.output}")
    else:
        print(output_json)
    
    return 0


if __name__ == "__main__":
    exit(main())

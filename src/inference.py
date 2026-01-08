"""
Inference module for fine-tuned financial extraction model.

This module handles inference with chunking support for long documents,
designed for use with smaller fine-tuned models that have limited context windows.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

from chunking import estimate_tokens, chunk_document, merge_extraction_results
from data import get_system_prompt
from schema import FinancialMetrics, parse_json_from_response


# Default configuration for fine-tuned model inference
DEFAULT_MAX_TOKENS = 2048  # Smaller context for fine-tuned models
DEFAULT_OVERLAP_TOKENS = 200


class FinancialExtractor:
    """
    Financial data extractor using a fine-tuned model.
    
    Automatically chunks long documents and merges extraction results.
    """
    
    def __init__(
        self,
        model_path: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        device: str = "auto"
    ):
        """
        Initialize the extractor with a fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model (LoRA adapter or merged)
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks for context continuity
            device: Device to run inference on ("auto", "cuda", "mps", "cpu")
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.device = device
        
        self.model = None
        self.tokenizer = None
        self.system_prompt = get_system_prompt()
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading model from {self.model_path}...")
            
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    device_map = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device_map = "mps"
                else:
                    device_map = "cpu"
            else:
                device_map = self.device
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if device_map != "cpu" else torch.float32,
                device_map=device_map
            )
            
            logger.info(f"Model loaded on {device_map}")
            
        except ImportError:
            logger.error("transformers package not installed. Run: pip install transformers torch")
            raise
    
    def _generate_response(self, text: str) -> str:
        """
        Generate extraction response for a single chunk.
        
        Args:
            text: Document text (chunk or full document)
            
        Returns:
            Raw model response string
        """
        if self.model is None:
            self.load_model()
        
        # Format as chat messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode response (only the new tokens)
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract financial metrics from document text.
        
        Automatically chunks long documents and merges results.
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary of extracted financial metrics
        """
        token_estimate = estimate_tokens(text)
        
        if token_estimate > self.max_tokens:
            # Document needs chunking
            logger.info(f"Document ~{token_estimate} tokens, chunking into pieces...")
            chunks = chunk_document(text, self.max_tokens, self.overlap_tokens)
            
            chunk_results = []
            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}...")
                try:
                    response = self._generate_response(chunk)
                    result = parse_json_from_response(response)
                    chunk_results.append(result)
                except Exception as e:
                    logger.warning(f"Chunk {i+1} extraction failed: {e}")
                    continue
            
            if chunk_results:
                merged = merge_extraction_results(chunk_results)
                logger.info(f"Merged {len(chunk_results)} chunk results")
                return merged
            else:
                logger.error("All chunks failed extraction")
                return {}
        else:
            # Single extraction
            response = self._generate_response(text)
            return parse_json_from_response(response)
    
    def extract_validated(self, text: str) -> FinancialMetrics:
        """
        Extract and validate financial metrics.
        
        Args:
            text: Full document text
            
        Returns:
            Validated FinancialMetrics object
        """
        raw_result = self.extract(text)
        return FinancialMetrics.model_validate(raw_result)


def extract_from_file(
    file_path: str,
    model_path: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
) -> Dict[str, Any]:
    """
    Convenience function to extract metrics from a file.
    
    Args:
        file_path: Path to the document file
        model_path: Path to the fine-tuned model
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
        
    Returns:
        Dictionary of extracted financial metrics
    """
    extractor = FinancialExtractor(
        model_path=model_path,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens
    )
    
    text = Path(file_path).read_text(encoding="utf-8")
    return extractor.extract(text)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract financial metrics from documents")
    parser.add_argument("file", help="Path to document file")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Maximum tokens per chunk")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_TOKENS,
                        help="Overlap tokens between chunks")
    
    args = parser.parse_args()
    
    result = extract_from_file(
        args.file,
        args.model,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap
    )
    
    print(json.dumps(result, indent=2))

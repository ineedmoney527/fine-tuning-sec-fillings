"""
Inference module with Pydantic validation and retry logic.

This module provides the FinancialExtractor class that loads a fine-tuned
model, extracts financial metrics from text, validates the output against
a Pydantic schema, and retries with error feedback on validation failures.

Usage:
    from src.inference import FinancialExtractor
    
    extractor = FinancialExtractor("outputs/qwen3-8b-financial-lora")
    result = extractor.extract(markdown_content)
    print(result.metrics)
"""

import json
from pathlib import Path
from typing import Optional
from loguru import logger

from pydantic import ValidationError

from src.schema import (
    FinancialMetrics, 
    ExtractionResult, 
    parse_json_from_response,
    SCHEMA_STRING
)
from src.data import get_system_prompt


class FinancialExtractor:
    """
    Financial metrics extractor with Pydantic validation and retry logic.
    
    This class:
    1. Loads the fine-tuned Qwen3 model with LoRA adapter
    2. Generates structured JSON from 10-K markdown content
    3. Validates output against Pydantic schema
    4. Retries with error feedback if validation fails
    """
    
    def __init__(
        self,
        model_path: str | Path,
        max_retries: int = 3,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        """
        Initialize the extractor.
        
        Args:
            model_path: Path to saved LoRA adapter or merged model
            max_retries: Maximum retry attempts on validation failure
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (low = more deterministic)
        """
        self.model_path = Path(model_path)
        self.max_retries = max_retries
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model."""
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        
        logger.info(f"Loading model from {self.model_path}")
        
        # Check if this is a merged model or LoRA adapter
        merged_path = self.model_path / "merged"
        if merged_path.exists():
            load_path = merged_path
            logger.info("Loading merged model")
        else:
            load_path = self.model_path
            logger.info("Loading LoRA adapter")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(load_path),
            max_seq_length=8192,
            dtype=None,
            load_in_4bit=True,
        )
        
        # Enable inference mode for faster generation
        FastLanguageModel.for_inference(self.model)
        
        # Set up chat template
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen-2.5",
        )
        
        logger.info("Model loaded successfully")
    
    def _build_messages(
        self, 
        content: str, 
        error_feedback: Optional[str] = None
    ) -> list[dict]:
        """
        Build chat messages for the model.
        
        Args:
            content: 10-K markdown content
            error_feedback: Optional error message from previous attempt
            
        Returns:
            List of message dicts in ChatML format
        """
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": content}
        ]
        
        if error_feedback:
            # Add error context for retry
            messages.append({
                "role": "assistant",
                "content": error_feedback
            })
            messages.append({
                "role": "user", 
                "content": (
                    "The previous response was invalid. "
                    "Please output ONLY valid JSON matching the schema. "
                    "Do not include any explanation or markdown formatting."
                )
            })
        
        return messages
    
    def _generate(self, messages: list[dict]) -> str:
        """
        Generate response from the model.
        
        Args:
            messages: Chat messages
            
        Returns:
            Generated text response
        """
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Decode only the generated part
        generated = outputs[0][inputs.shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return response.strip()
    
    def extract(self, content: str) -> ExtractionResult:
        """
        Extract financial metrics from 10-K content.
        
        This method attempts extraction, validates against Pydantic schema,
        and retries with error feedback if validation fails.
        
        Args:
            content: 10-K markdown content
            
        Returns:
            ExtractionResult with validated metrics
            
        Raises:
            ValueError: If extraction fails after max_retries attempts
        """
        errors = []
        last_response = ""
        
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Extraction attempt {attempt}/{self.max_retries}")
            
            # Build messages with error feedback if retrying
            error_feedback = errors[-1] if errors else None
            messages = self._build_messages(content, error_feedback)
            
            # Generate response
            response = self._generate(messages)
            last_response = response
            logger.debug(f"Raw response: {response[:200]}...")
            
            try:
                # Parse JSON from response
                data = parse_json_from_response(response)
                
                # Validate against Pydantic schema
                metrics = FinancialMetrics.model_validate(data)
                
                logger.info(f"Extraction successful on attempt {attempt}")
                return ExtractionResult(
                    metrics=metrics,
                    raw_response=response,
                    attempts=attempt,
                    validation_errors=errors
                )
                
            except ValueError as e:
                # JSON parsing failed
                error_msg = f"JSON parsing error: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Attempt {attempt} failed: {error_msg}")
                
            except ValidationError as e:
                # Pydantic validation failed
                error_msg = f"Validation error: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Attempt {attempt} failed: {error_msg}")
        
        # All retries exhausted
        raise ValueError(
            f"Extraction failed after {self.max_retries} attempts. "
            f"Errors: {errors}. Last response: {last_response[:500]}"
        )
    
    def extract_batch(
        self, 
        contents: list[str],
        continue_on_error: bool = True
    ) -> list[ExtractionResult | Exception]:
        """
        Extract financial metrics from multiple documents.
        
        Args:
            contents: List of 10-K markdown contents
            continue_on_error: If True, continue processing on failures
            
        Returns:
            List of ExtractionResults or Exceptions if continue_on_error=True
        """
        results = []
        for i, content in enumerate(contents):
            logger.info(f"Processing document {i+1}/{len(contents)}")
            try:
                result = self.extract(content)
                results.append(result)
            except Exception as e:
                if continue_on_error:
                    logger.error(f"Document {i+1} failed: {e}")
                    results.append(e)
                else:
                    raise
        return results


def quick_extract(
    content: str,
    model_path: str = "outputs/qwen3-8b-financial-lora",
    max_retries: int = 3
) -> FinancialMetrics:
    """
    Convenience function for one-off extraction.
    
    Args:
        content: 10-K markdown content
        model_path: Path to fine-tuned model
        max_retries: Maximum retry attempts
        
    Returns:
        Validated FinancialMetrics
    """
    extractor = FinancialExtractor(model_path, max_retries=max_retries)
    result = extractor.extract(content)
    return result.metrics

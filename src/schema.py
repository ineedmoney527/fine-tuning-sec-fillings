"""
Pydantic schema for financial extraction output validation.

This module provides structured data models for validating
LLM-extracted financial metrics from SEC 10-K reports.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
import json
import re


class FinancialMetrics(BaseModel):
    """
    Pydantic model for validated financial extraction output.
    
    All monetary values are expected as full integers (not in millions/billions).
    EPS is a float value.
    """
    revenue: Optional[int] = Field(
        default=None,
        description="Total Net Sales / Revenue in full integer value"
    )
    operating_income: Optional[int] = Field(
        default=None,
        description="Operating Income / Loss in full integer value"
    )
    net_income: Optional[int] = Field(
        default=None,
        description="Net Income / Loss in full integer value"
    )
    total_assets: Optional[int] = Field(
        default=None,
        description="Total Assets in full integer value"
    )
    cash_and_equivalents: Optional[int] = Field(
        default=None,
        description="Ending Cash & Cash Equivalents in full integer value"
    )
    diluted_eps: Optional[float] = Field(
        default=None,
        description="Diluted Earnings Per Share"
    )

    @field_validator('revenue', 'operating_income', 'net_income', 
                     'total_assets', 'cash_and_equivalents', mode='before')
    @classmethod
    def coerce_to_int(cls, v):
        """Coerce numeric values to integers."""
        if v is None:
            return None
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            # Remove commas and try to parse
            cleaned = v.replace(',', '').strip()
            if cleaned.lower() in ('null', 'none', ''):
                return None
            return int(float(cleaned))
        return v

    @field_validator('diluted_eps', mode='before')
    @classmethod
    def coerce_to_float(cls, v):
        """Coerce EPS to float."""
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.replace(',', '').strip()
            if cleaned.lower() in ('null', 'none', ''):
                return None
            return float(cleaned)
        return float(v) if v is not None else None


class ExtractionResult(BaseModel):
    """Wrapper for extraction result with metadata."""
    metrics: FinancialMetrics
    raw_response: str
    attempts: int = 1
    validation_errors: list[str] = Field(default_factory=list)


def parse_json_from_response(response: str) -> dict:
    """
    Extract JSON from LLM response, handling various formats.
    
    Args:
        response: Raw LLM output that may contain JSON
        
    Returns:
        Parsed JSON as dict
        
    Raises:
        ValueError: If no valid JSON found
    """
    # Try direct parse first
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in markdown code blocks
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}'
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                continue
    
    raise ValueError(f"Could not extract valid JSON from response: {response[:200]}...")


def validate_extraction(response: str) -> tuple[FinancialMetrics, list[str]]:
    """
    Parse and validate LLM response against FinancialMetrics schema.
    
    Args:
        response: Raw LLM output containing JSON
        
    Returns:
        Tuple of (validated FinancialMetrics, list of error messages)
        
    Raises:
        ValueError: If JSON parsing fails
        ValidationError: If Pydantic validation fails
    """
    errors = []
    
    # Parse JSON
    try:
        data = parse_json_from_response(response)
    except ValueError as e:
        errors.append(str(e))
        raise
    
    # Validate against schema
    metrics = FinancialMetrics.model_validate(data)
    
    return metrics, errors


# Schema representation for prompt injection
SCHEMA_STRING = """{
  "revenue": <int or null>,             // Total Net Sales / Revenue
  "operating_income": <int or null>,    // Operating Income / Loss
  "net_income": <int or null>,          // Net Income / Loss
  "total_assets": <int or null>,        // Total Assets
  "cash_and_equivalents": <int or null>,// Ending Cash & Cash Equivalents
  "diluted_eps": <float or null>        // Diluted Earnings Per Share
}"""

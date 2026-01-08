"""
Document chunking and result merging for long financial documents.

This module handles splitting long 10-K documents into chunks with sliding
window overlap, and merging JSON extraction results from multiple chunks.
"""

import re
from typing import Dict, List, Any, Optional
from loguru import logger


# Default configuration
DEFAULT_MAX_TOKENS = 4096
DEFAULT_OVERLAP_TOKENS = 200
CHARS_PER_TOKEN = 4  # Rough heuristic for token estimation


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using character-based heuristic.
    
    Args:
        text: Input text to estimate tokens for
        
    Returns:
        Estimated token count (approximately len(text) / 4)
    """
    return len(text) // CHARS_PER_TOKEN


def find_split_point(text: str, target_pos: int, search_range: int = 200) -> int:
    """
    Find a natural split point (paragraph boundary) near target position.
    
    Searches for double newlines or section headers near the target position
    to avoid splitting in the middle of sentences or tables.
    
    Args:
        text: Full text to search in
        target_pos: Target character position to split near
        search_range: How far to search before/after target (in chars)
        
    Returns:
        Best split position (character index)
    """
    # Search window around target
    start = max(0, target_pos - search_range)
    end = min(len(text), target_pos + search_range)
    window = text[start:end]
    
    # Priority 1: Double newline (paragraph break)
    double_newline = window.rfind('\n\n')
    if double_newline != -1:
        return start + double_newline + 2  # After the double newline
    
    # Priority 2: Section header (# followed by text)
    header_match = re.search(r'\n#+\s', window)
    if header_match:
        return start + header_match.start() + 1  # Before the header
    
    # Priority 3: Single newline
    single_newline = window.rfind('\n')
    if single_newline != -1:
        return start + single_newline + 1
    
    # Fallback: Just use target position
    return target_pos


def chunk_document(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
) -> List[str]:
    """
    Split document into chunks with sliding window overlap.
    
    Splits at natural boundaries (paragraphs, headers) when possible.
    The overlap ensures context continuity between chunks.
    
    Args:
        text: Full document text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    total_tokens = estimate_tokens(text)
    
    # No chunking needed
    if total_tokens <= max_tokens:
        logger.debug(f"Document under {max_tokens} tokens, no chunking needed")
        return [text]
    
    # Convert token limits to character positions
    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    
    chunks = []
    current_pos = 0
    chunk_num = 0
    
    while current_pos < len(text):
        chunk_num += 1
        
        # Calculate end position for this chunk
        chunk_end = current_pos + max_chars
        
        if chunk_end >= len(text):
            # Last chunk - take everything remaining
            chunks.append(text[current_pos:])
            logger.debug(f"Chunk {chunk_num}: chars {current_pos}-{len(text)} (final)")
            break
        
        # Find natural split point
        split_pos = find_split_point(text, chunk_end)
        
        # Extract chunk
        chunk_text = text[current_pos:split_pos]
        chunks.append(chunk_text)
        
        logger.debug(f"Chunk {chunk_num}: chars {current_pos}-{split_pos} (~{estimate_tokens(chunk_text)} tokens)")
        
        # Move position forward, but keep overlap
        current_pos = split_pos - overlap_chars
        # Ensure we don't go backwards
        current_pos = max(current_pos, split_pos - overlap_chars)
        # Ensure forward progress
        if current_pos <= 0 or (chunks and current_pos < len(text) - len(chunks[-1])):
            current_pos = split_pos
    
    logger.info(f"Split document into {len(chunks)} chunks (overlap: {overlap_tokens} tokens)")
    return chunks


def merge_extraction_results(
    results: List[Dict[str, Any]],
    strategy: str = "first_non_null"
) -> Dict[str, Any]:
    """
    Merge multiple JSON extraction results into a single result.
    
    Uses "first non-null" strategy by default: for each key, take the first
    non-null value encountered across all chunks.
    
    Args:
        results: List of extraction result dictionaries
        strategy: Merging strategy ("first_non_null" or "most_confident")
        
    Returns:
        Merged dictionary with best values from all chunks
    """
    if not results:
        return {}
    
    if len(results) == 1:
        return results[0]
    
    merged = {}
    
    # Track which chunk each value came from (for logging)
    value_sources: Dict[str, int] = {}
    
    # Collect all unique keys
    all_keys = set()
    for result in results:
        if result:
            all_keys.update(result.keys())
    
    if strategy == "first_non_null":
        # For each key, find first non-null value
        for key in all_keys:
            for chunk_idx, result in enumerate(results):
                if result and key in result:
                    value = result[key]
                    # Check if value is non-null (handle nested dict format)
                    if _is_non_null(value):
                        merged[key] = value
                        value_sources[key] = chunk_idx
                        break
            else:
                # No non-null value found, use null
                merged[key] = None
    
    elif strategy == "most_confident":
        # For each key, prefer value from chunk with most non-null extractions
        chunk_scores = []
        for result in results:
            if result:
                score = sum(1 for v in result.values() if _is_non_null(v))
                chunk_scores.append(score)
            else:
                chunk_scores.append(0)
        
        # Sort chunks by score (highest first)
        sorted_indices = sorted(range(len(results)), key=lambda i: chunk_scores[i], reverse=True)
        
        for key in all_keys:
            for chunk_idx in sorted_indices:
                result = results[chunk_idx]
                if result and key in result and _is_non_null(result[key]):
                    merged[key] = result[key]
                    value_sources[key] = chunk_idx
                    break
            else:
                merged[key] = None
    
    # Log merge summary
    logger.debug(f"Merged {len(results)} chunks: {len([k for k, v in merged.items() if _is_non_null(v)])} non-null values")
    
    return merged


def _is_non_null(value: Any) -> bool:
    """
    Check if a value is non-null (handles nested dict format).
    
    Values can be:
    - Simple: 123 or None
    - Nested: {"value": 123, "unit": "millions"} or {"value": null, ...}
    """
    if value is None:
        return False
    
    if isinstance(value, dict):
        # Nested format - check the "value" key
        inner_value = value.get("value")
        return inner_value is not None
    
    return True

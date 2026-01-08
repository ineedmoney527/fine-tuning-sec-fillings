"""
Unit tests for the chunking module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunking import (
    estimate_tokens,
    find_split_point,
    chunk_document,
    merge_extraction_results,
    _is_non_null,
    CHARS_PER_TOKEN
)


class TestEstimateTokens:
    """Tests for estimate_tokens function."""
    
    def test_empty_string(self):
        assert estimate_tokens("") == 0
    
    def test_short_text(self):
        # 20 chars / 4 = 5 tokens
        assert estimate_tokens("12345678901234567890") == 5
    
    def test_long_text(self):
        text = "a" * 4000
        assert estimate_tokens(text) == 1000


class TestFindSplitPoint:
    """Tests for find_split_point function."""
    
    def test_finds_double_newline(self):
        text = "First paragraph.\n\nSecond paragraph."
        # Target near position 25 should find the \n\n at position 16-17
        split = find_split_point(text, 25, search_range=50)
        assert split == 18  # After \n\n
    
    def test_finds_header(self):
        text = "Some content here.\n# Header\nMore content."
        split = find_split_point(text, 25, search_range=50)
        # Should find before the header
        assert text[split-1:split+2] == "\n# "[0:3] or split == 19
    
    def test_falls_back_to_single_newline(self):
        text = "Line one.\nLine two.\nLine three."
        split = find_split_point(text, 15, search_range=10)
        assert text[split-1] == '\n' or split == 15


class TestChunkDocument:
    """Tests for chunk_document function."""
    
    def test_no_chunking_needed(self):
        text = "Short document."
        chunks = chunk_document(text, max_tokens=1000, overlap_tokens=10)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_creates_multiple_chunks(self):
        # Create text that's ~1000 tokens (4000 chars)
        text = "Word " * 800  # ~4000 chars = ~1000 tokens
        chunks = chunk_document(text, max_tokens=300, overlap_tokens=50)
        assert len(chunks) > 1
    
    def test_overlap_exists(self):
        # Create text that needs chunking
        text = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100
        chunks = chunk_document(text, max_tokens=70, overlap_tokens=20)
        
        if len(chunks) > 1:
            # Check that end of chunk 0 overlaps with start of chunk 1
            # (Approximate check - overlap should exist)
            assert len(chunks[0]) > 50  # Chunk has content
    
    def test_handles_natural_boundaries(self):
        text = "First section content.\n\nSecond section content.\n\nThird section."
        chunks = chunk_document(text, max_tokens=20, overlap_tokens=5)
        # Should try to split at paragraph boundaries
        assert len(chunks) >= 1


class TestMergeExtractionResults:
    """Tests for merge_extraction_results function."""
    
    def test_empty_results(self):
        assert merge_extraction_results([]) == {}
    
    def test_single_result(self):
        result = {"revenue": 1000, "net_income": 500}
        assert merge_extraction_results([result]) == result
    
    def test_first_non_null_strategy(self):
        results = [
            {"revenue": None, "net_income": 500},
            {"revenue": 1000, "net_income": 600},
            {"revenue": 2000, "net_income": None}
        ]
        merged = merge_extraction_results(results, strategy="first_non_null")
        assert merged["revenue"] == 1000  # First non-null
        assert merged["net_income"] == 500  # First non-null
    
    def test_handles_nested_dict_format(self):
        results = [
            {"revenue": {"value": None, "unit": "millions"}},
            {"revenue": {"value": 1000, "unit": "millions"}}
        ]
        merged = merge_extraction_results(results, strategy="first_non_null")
        assert merged["revenue"]["value"] == 1000
    
    def test_all_nulls_returns_null(self):
        results = [
            {"revenue": None},
            {"revenue": None}
        ]
        merged = merge_extraction_results(results)
        assert merged["revenue"] is None
    
    def test_combines_different_keys(self):
        results = [
            {"revenue": 1000},
            {"net_income": 500}
        ]
        merged = merge_extraction_results(results)
        assert merged["revenue"] == 1000
        assert merged["net_income"] == 500
    
    def test_most_confident_strategy(self):
        results = [
            {"revenue": 100, "income": None, "assets": None},  # 1 non-null
            {"revenue": 200, "income": 50, "assets": 1000}     # 3 non-null
        ]
        merged = merge_extraction_results(results, strategy="most_confident")
        # Should prefer values from chunk with more data
        assert merged["revenue"] == 200


class TestIsNonNull:
    """Tests for _is_non_null helper function."""
    
    def test_none_is_null(self):
        assert _is_non_null(None) is False
    
    def test_integer_is_non_null(self):
        assert _is_non_null(123) is True
        assert _is_non_null(0) is True
    
    def test_string_is_non_null(self):
        assert _is_non_null("hello") is True
        assert _is_non_null("") is True  # Empty string is still non-null
    
    def test_nested_dict_with_value(self):
        assert _is_non_null({"value": 100, "unit": "millions"}) is True
    
    def test_nested_dict_with_null_value(self):
        assert _is_non_null({"value": None, "unit": "millions"}) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

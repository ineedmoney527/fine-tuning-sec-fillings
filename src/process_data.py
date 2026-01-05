#!/usr/bin/env python3
"""
Process raw HTML 10-K filings into clean Markdown format.

This script converts HTML files from data/raw/ to clean Markdown using docling,
preserving table structures and removing noise.
"""

import re
from pathlib import Path
from typing import Optional, List

from loguru import logger

try:
    from docling.document_converter import DocumentConverter
    USING_DOCLING = True
except ImportError:
    logger.warning("docling not available, falling back to markdownify")
    from markdownify import markdownify as md
    USING_DOCLING = False


# Configuration
INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
MIN_WORDS_PER_LINE = 5  # Remove lines with fewer words (noise removal)


def setup_logging() -> None:
    """Configure loguru logger."""
    logger.add(
        "logs/process_data_{time}.log",
        rotation="10 MB",
        retention="30 days",
        level="INFO",
    )


def convert_html_to_markdown_docling(html_content: str) -> Optional[str]:
    """
    Convert HTML to Markdown using docling.
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Markdown string, or None if conversion failed
    """
    try:
        logger.info("Converting HTML to Markdown using docling...")
        
        # Initialize DocumentConverter
        converter = DocumentConverter()
        
        # Convert HTML to markdown
        # docling expects file path or file-like object
        # We'll save temporarily and convert
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp:
            tmp.write(html_content)
            tmp_path = tmp.name
        
        try:
            result = converter.convert(tmp_path)
            markdown = result.document.export_to_markdown()
            return markdown
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"Error converting with docling: {e}")
        return None


def convert_html_to_markdown_markdownify(html_content: str) -> Optional[str]:
    """
    Convert HTML to Markdown using markdownify (fallback).
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Markdown string, or None if conversion failed
    """
    try:
        logger.info("Converting HTML to Markdown using markdownify...")
        
        # Convert with table preservation
        markdown = md(
            html_content,
            heading_style="ATX",  # Use # for headings
            bullets="-",  # Use - for bullet lists
            strip=['script', 'style'],  # Remove script/style tags
        )
        
        return markdown
        
    except Exception as e:
        logger.error(f"Error converting with markdownify: {e}")
        return None


def parse_markdown_table(table_lines: List[str]) -> List[List[str]]:
    """Parse markdown table lines into a matrix."""
    matrix = []
    for line in table_lines:
        # Split by pipe, handle edge cases roughly
        cells = [c.strip() for c in line.strip('|').split('|')]
        matrix.append(cells)
    return matrix


def reconstruct_markdown_table(matrix: List[List[str]]) -> List[str]:
    """Reconstruct markdown table from matrix."""
    lines = []
    for row in matrix:
        lines.append('| ' + ' | '.join(row) + ' |')
    return lines


def is_decorator(text: str) -> bool:
    """Check if text is just a decorator (symbols, spaces)."""
    if not text:
        return True
    return all(c in "$%() ,.-" for c in text)


def optimize_table(table_lines: List[str]) -> List[str]:
    """
    Remove redundant/duplicate/empty columns by merging them.
    Includes row-level structural cleanup (decorator merging and deduplication).
    """
    if not table_lines:
        return []
        
    matrix = parse_markdown_table(table_lines)
    if not matrix:
        return []

    # Normalize matrix width
    max_cols = max(len(row) for row in matrix)
    for row in matrix:
        while len(row) < max_cols:
            row.append("")

    num_rows = len(matrix)
    num_cols = max_cols
    
    # 1. Row-level cleanup: Handle decorators and identical neighbors
    for r in range(num_rows):
        # Skip separator rows
        is_sep = set(matrix[r][0]).issubset({'-', ':', ' '}) and '-' in matrix[r][0]
        if is_sep:
            continue
            
        for c in range(num_cols):
            val = matrix[r][c]
            
            # Merge decorator columns (e.g. $) into next data column
            if c < num_cols - 1 and is_decorator(val) and val.strip():
                next_val = matrix[r][c+1]
                clean_next = next_val.strip()
                is_financial_number = (
                    any(char.isdigit() for char in clean_next) or 
                    clean_next == '-' or 
                    clean_next == '—' or  # em-dash
                    (clean_next.startswith('(') and clean_next.endswith(')'))
                )

                if is_financial_number:
                    matrix[r][c+1] = f"{val.strip()}{clean_next}"
                    matrix[r][c] = ""
            
            # Suppress internal row duplicates (e.g. Label | Label)
            if c > 0 and matrix[r][c] == matrix[r][c-1] and matrix[r][c]:
                matrix[r][c] = ""

    # 2. Global Column Merging
    deleted_cols = set()
    
    # Try to merge column j into some column i < j
    for j in range(1, num_cols):
        for i in range(j):
            if i in deleted_cols: continue
            
            # check if mergeable
            can_merge = True
            for r in range(num_rows):
                val_i = matrix[r][i]
                val_j = matrix[r][j]
                
                # Check for separator rows (e.g., ---)
                is_sep_i = set(val_i).issubset({'-', ':', ' '}) and '-' in val_i
                is_sep_j = set(val_j).issubset({'-', ':', ' '}) and '-' in val_j
                
                if is_sep_i and is_sep_j:
                    continue 
                
                if val_i and val_j and val_i != val_j:
                    can_merge = False
                    break
            
            if can_merge:
                # Merge j into i
                for r in range(num_rows):
                    if not matrix[r][i]:
                        matrix[r][i] = matrix[r][j]
                deleted_cols.add(j)
                break 
                
    new_matrix = []
    for r in range(num_rows):
        new_row = [matrix[r][c] for c in range(num_cols) if c not in deleted_cols]
        new_matrix.append(new_row)
        
    return reconstruct_markdown_table(new_matrix)

def strip_markdown_links(text: str) -> str:
    """
    Removes markdown links [Text](url) -> Text.
    Also cleans standard HTML artifacts.
    """
    # Remove link structure, keep label: [Label](url) -> Label
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove anchor tags: <a ...>Label</a> -> Label
    text = re.sub(r'<a[^>]*>(.*?)</a>', r'\1', text)
    return text.strip()

def is_narrative_noise(line: str) -> bool:
    """
    Determines if a line is likely narrative text (sentences) 
    that should be removed to save tokens.
    """
    # 1. Structural checks
    if not line.strip(): return False
    if line.startswith('|'): return False # Never delete table rows here
    if line.startswith('#'): return False # Keep headers
    
    # 2. Content checks
    words = line.split()
    
    # If it's long (> 15 words), it's a paragraph. 
    # Financial headers are rarely this long.
    if len(words) > 15:
        return True
        
    # 3. Detect "See accompanying notes" or "Table of Contents" garbage
    lower_line = line.lower()
    noise_phrases = [
        "see accompanying notes",
        "table of contents",
        "index to consolidated",
        "click here",
        "return to top"
    ]
    if any(phrase in lower_line for phrase in noise_phrases):
        return True

    return False

def clean_markdown(markdown: str) -> str:
    """
    Densify markdown content:
    1. Strip links and artifacts first.
    2. Filter long narrative text.
    3. Optimize tables.
    """
    # Initial cleanup of the whole block
    markdown = strip_markdown_links(markdown)
    
    lines = markdown.split('\n')
    cleaned_lines: List[str] = []
    
    in_table = False
    table_buffer: List[str] = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check if line looks like a table row
        # Must have at least one internal pipe to be a real table row
        # (Avoids false positives on lines that just start with | for styling)
        is_table_row = stripped.startswith('|') and stripped.count('|') > 1
        
        if is_table_row:
            if not in_table:
                in_table = True
                table_buffer = []
            table_buffer.append(stripped)
        else:
            if in_table:
                # Flush and optimize the buffered table
                if table_buffer:
                    optimized = optimize_table(table_buffer)
                    # Only keep tables that have actual content (more than just headers)
                    if len(optimized) > 2: 
                        cleaned_lines.extend(optimized)
                        cleaned_lines.append("") # Spacing
                in_table = False
                table_buffer = []
            
            # --- AGGRESSIVE FILTERING LOGIC ---
            if is_narrative_noise(stripped):
                # logger.debug(f"Dropped narrative: {stripped[:30]}...")
                continue
            
            # If we survived the noise check, keep it
            if stripped:
                cleaned_lines.append(stripped)
    
    # Flush trailing table if file ends with a table
    if in_table and table_buffer:
        optimized = optimize_table(table_buffer)
        if len(optimized) > 2:
            cleaned_lines.extend(optimized)
    
    # Final whitespace cleanup
    cleaned = '\n'.join(cleaned_lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()

def process_html_file(
    input_path: Path, 
    output_path: Path,
    use_docling: bool = USING_DOCLING
) -> bool:
    """
    Process a single HTML file to Markdown.
    
    Args:
        input_path: Path to input HTML file
        output_path: Path to output Markdown file
        use_docling: Whether to use docling (or markdownify)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing {input_path.name}...")
        
        # Read HTML
        html_content = input_path.read_text(encoding='utf-8')
        
        # Convert to Markdown
        if use_docling:
            markdown = convert_html_to_markdown_docling(html_content)
        else:
            markdown = convert_html_to_markdown_markdownify(html_content)
        
        if not markdown:
            logger.error(f"Failed to convert {input_path.name}")
            return False
        
        # Clean markdown
        cleaned_markdown = clean_markdown(markdown)
        
        # Save to output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned_markdown, encoding='utf-8')
        
        logger.success(
            f"Processed {input_path.name} -> {output_path.name} "
            f"({len(cleaned_markdown)} chars)"
        )
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_path.name}: {e}")
        return False


def process_all_files(input_dir: Path, output_dir: Path) -> dict:
    """
    Process all HTML files in input directory.
    
    Args:
        input_dir: Directory containing raw HTML files
        output_dir: Directory to save processed Markdown files
        
    Returns:
        Dictionary with processing statistics
    """
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return {"attempted": 0, "successful": 0, "failed": 0}
    
    # Find all HTML files
    html_files = list(input_dir.glob("**/*.html"))

    
    if not html_files:
        logger.warning(f"No HTML files found in {input_dir}")
        return {"attempted": 0, "successful": 0, "failed": 0}
    
    logger.info(f"Found {len(html_files)} HTML files to process")
    
    stats = {"attempted": 0, "successful": 0, "failed": 0}
    
    for html_file in html_files:
        stats["attempted"] += 1
        
        # Generate output filename (e.g., AAPL_2024.html -> AAPL_2024.md)
        output_file = output_dir / html_file.with_suffix('.md').name
        
        if process_html_file(html_file, output_file):
            stats["successful"] += 1
        else:
            stats["failed"] += 1
    
    return stats


def main() -> None:
    """Main entry point."""
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("HTML to Markdown Processor")
    logger.info("=" * 60)
    logger.info(f"Using converter: {'docling' if USING_DOCLING else 'markdownify'}")
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Min words per line: {MIN_WORDS_PER_LINE}")
    logger.info("")
    
    # Process all files
    stats = process_all_files(INPUT_DIR, OUTPUT_DIR)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"Attempted: {stats['attempted']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    if stats["attempted"] > 0:
        logger.info(
            f"Success rate: {stats['successful']/stats['attempted']*100:.1f}%"
        )


if __name__ == "__main__":
    main()

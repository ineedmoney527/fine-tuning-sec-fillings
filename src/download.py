
#!/usr/bin/env python3
"""
Download SEC 10-K filings Item 8 (Financial Statements) for specified tickers.

This script uses the sec-api QueryApi to fetch 10-K filings for target companies
and extracts specifically Item 8 (Financial Statements and Supplementary Data).
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from loguru import logger
from sec_api import QueryApi, ExtractorApi


# Configuration
API_KEY = os.getenv("SEC_API_KEY", "")  # Set via environment variable
TICKERS =["PLD", "ADI"]






YEARS = [2022, 2023, 2024]
OUTPUT_DIR = Path("data/raw")
RATE_LIMIT_DELAY = 0.3  # seconds between API calls


def setup_logging() -> None:
    """Configure loguru logger."""
    logger.add(
        "logs/download_filings_{time}.log",
        rotation="10 MB",
        retention="30 days",
        level="INFO",
    )


def validate_api_key(api_key: str) -> bool:
    """Validate that SEC API key is set."""
    if not api_key:
        logger.error("SEC_API_KEY environment variable not set!")
        logger.info("Set it with: export SEC_API_KEY='your-api-key-here'")
        return False
    return True


def query_filings(
    query_api: QueryApi, ticker: str, year: int
) -> Optional[List[Dict[str, Any]]]:
    """
    Query SEC API for 10-K filings for a specific ticker and year.

    Args:
        query_api: Initialized QueryApi instance
        ticker: Company ticker symbol
        year: Filing year

    Returns:
        List of filing metadata dicts, or None if error
    """
    # Build query for 10-K filings
    query = {
        "query": {
            "query_string": {
                "query": f'ticker:"{ticker}" AND formType:"10-K" AND filedAt:[{year}-01-01 TO {year}-12-31]'
            }
        },
        "from": 0,
        "size": 10,  # Should typically only be 1 per year
        "sort": [{"filedAt": {"order": "desc"}}],
    }

    try:
        logger.info(f"Querying 10-K for {ticker} in {year}...")
        response = query_api.get_filings(query)
        filings = response.get("filings", [])

        if not filings:
            logger.warning(f"No 10-K found for {ticker} in {year}")
            return None

        logger.success(f"Found {len(filings)} filing(s) for {ticker} in {year}")
        return filings

    except Exception as e:
        logger.error(f"Error querying {ticker} {year}: {e}")
        return None


def extract_item_8(
    extractor_api: ExtractorApi, filing_url: str, ticker: str, year: int
) -> Optional[str]:
    """
    Extract Item 8 (Financial Statements) from a 10-K filing.

    Args:
        extractor_api: Initialized ExtractorApi instance
        filing_url: URL to the full 10-K filing
        ticker: Company ticker symbol
        year: Filing year

    Returns:
        HTML content of Item 8, or None if error
    """
    try:
        logger.info(f"Extracting Item 8 for {ticker} {year}...")

        # Extract Item 8 section
        section_html = extractor_api.get_section(
            filing_url=filing_url,
            section="8",  # Item 8
            return_type="html",
        )

        if not section_html or len(section_html.strip()) < 100:
            logger.warning(f"Item 8 appears empty for {ticker} {year}")
            return None

        logger.success(
            f"Extracted Item 8 for {ticker} {year} ({len(section_html)} chars)"
        )
        return section_html

    except Exception as e:
        logger.error(f"Error extracting Item 8 for {ticker} {year}: {e}")
        return None


def save_html(content: str, ticker: str, year: int, output_dir: Path) -> bool:
    """
    Save HTML content to file.

    Args:
        content: HTML content to save
        ticker: Company ticker symbol
        year: Filing year
        output_dir: Output directory path

    Returns:
        True if saved successfully, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{ticker}_{year}.html"

    try:
        output_file.write_text(content, encoding="utf-8")
        logger.success(f"Saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving {output_file}: {e}")
        return False


def download_filings(
    tickers: List[str], years: List[int], output_dir: Path, api_key: str
) -> Dict[str, int]:
    """
    Download Item 8 from 10-K filings for specified tickers and years.

    Args:
        tickers: List of ticker symbols
        years: List of years to download
        output_dir: Directory to save HTML files
        api_key: SEC API key

    Returns:
        Dictionary with download statistics
    """
    query_api = QueryApi(api_key=api_key)
    extractor_api = ExtractorApi(api_key=api_key)

    stats = {"attempted": 0, "successful": 0, "failed": 0}

    for ticker in tickers:
        for year in years:
            stats["attempted"] += 1

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

            # Query for filing
            filings = query_filings(query_api, ticker, year)
            if not filings:
                stats["failed"] += 1
                continue

            # Get the most recent filing (should be only one)
            filing = filings[0]
            filing_url = filing.get("linkToFilingDetails")

            if not filing_url:
                logger.error(f"No filing URL found for {ticker} {year}")
                stats["failed"] += 1
                continue

            # Extract Item 8
            time.sleep(RATE_LIMIT_DELAY)  # Rate limit extraction API too
            item_8_html = extract_item_8(extractor_api, filing_url, ticker, year)

            # Check if Item 8 is too short (likely a reference/placeholder)
            # NVDA and some others put financials in Item 15 or split them
            if item_8_html and len(item_8_html.strip()) < 2000:
                logger.warning(f"Item 8 is remarkably short ({len(item_8_html)} chars). Attempting fallback to Item 15...")
                
                try:
                    time.sleep(RATE_LIMIT_DELAY)
                    item_15_html = extractor_api.get_section(
                        filing_url=filing_url,
                        section="15",
                        return_type="html"
                    )
                    
                    if item_15_html and len(item_15_html.strip()) > 1000:
                        logger.success(f"Found substantial content in Item 15 ({len(item_15_html)} chars). Appending.")
                        # Append Item 15 to Item 8
                        item_8_html += "\n<hr>\n" + item_15_html
                    else:
                        logger.warning("Item 15 was also empty or missing.")
                        
                except Exception as e:
                    logger.error(f"Fallback extraction failed: {e}")

            if not item_8_html:
                stats["failed"] += 1
                continue

            # Save to file
            if save_html(item_8_html, ticker, year, output_dir):
                stats["successful"] += 1
            else:
                stats["failed"] += 1

    return stats


def main() -> None:
    """Main entry point."""
    setup_logging()

    logger.info("=" * 60)
    logger.info("SEC 10-K Item 8 Downloader")
    logger.info("=" * 60)

    # Validate API key
    if not validate_api_key(API_KEY):
        return

    logger.info(f"Tickers: {', '.join(TICKERS)}")
    logger.info(f"Years: {', '.join(map(str, YEARS))}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("")

    # Download filings
    stats = download_filings(TICKERS, YEARS, OUTPUT_DIR, API_KEY)

    # Print summary
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)
    logger.info(f"Attempted: {stats['attempted']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(
        f"Success rate: {stats['successful']/stats['attempted']*100:.1f}%"
        if stats["attempted"] > 0
        else "N/A"
    )


if __name__ == "__main__":
    main()
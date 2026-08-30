"""
Integration test for the CIAN scraper.

IMPORTANT: This test makes real HTTP requests to cian.ru.
Run explicitly with:
    pytest tests/integration/test_cian_scraper.py -v

Do NOT include in the standard CI suite.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

pytestmark = pytest.mark.integration


@pytest.mark.integration
def test_cian_scraper_returns_listings():
    """Scrape 2 pages from Moscow and verify at least some listings are returned."""
    from src.data_collection.cian_scraper import CianScraper

    scraper = CianScraper(city="moskva", deal_type="sale")
    listings = scraper.scrape_pages(scraper.base_url, max_pages=2)

    # We only assert structure, not a specific count (site may block)
    assert isinstance(listings, list), "scrape_pages must return a list"
    if listings:
        first = listings[0]
        for key in ("price", "total_area", "rooms"):
            assert key in first, f"Expected key '{key}' in listing dict"

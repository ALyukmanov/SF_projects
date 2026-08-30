"""
DEPRECATED: This file is a manual integration script, not a pytest test.

Use the proper integration test instead:
    pytest tests/integration/test_cian_scraper.py -v

Or run this script directly:
    python test_cian_scraper.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging

from src.data_collection.cian_scraper import CianScraper

logging.basicConfig(level=logging.INFO)


def test_cian():
    """Test CIAN scraper with minimal pages"""
    print("=" * 70)
    print("TESTING CIAN SCRAPER")
    print("=" * 70)

    # Create scraper
    scraper = CianScraper(city="moskva", deal_type="sale")
    print(f"\nBase URL: {scraper.base_url}")
    print(f"Delay range: {scraper.delay_range[0]}-{scraper.delay_range[1]} seconds")

    # Try to scrape 2 pages
    print("\nScraping 2 pages from Moscow...")
    listings = scraper.scrape_pages(scraper.base_url, max_pages=2)

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Total listings collected: {len(listings)}")

    if len(listings) > 0:
        print("\n✅ CIAN scraper is WORKING!")

        # Show first listing
        print("\nFirst listing sample:")
        first = listings[0]
        for key, value in first.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        # Check unique addresses
        addresses = [item.get("address", "") for item in listings if item.get("address")]
        unique_addresses = set(addresses)
        print("\nAddress uniqueness:")
        print(f"  Total addresses: {len(addresses)}")
        print(f"  Unique addresses: {len(unique_addresses)}")
        print(f"  Uniqueness rate: {len(unique_addresses) / len(addresses) * 100:.1f}%")

        # Check real dates
        if "scraped_at" in first:
            print(f"\nScraped timestamp: {first['scraped_at']}")
            print("  (This should be current date/time - January 2026)")

        print("\n✅ Ready for full collection!")
        return True
    else:
        print("\n❌ No listings collected!")
        print("Possible issues:")
        print("  - CIAN might be blocking requests (HTTP 429)")
        print("  - CSS selectors might be outdated")
        print("  - Network connectivity issues")
        return False


if __name__ == "__main__":
    try:
        success = test_cian()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

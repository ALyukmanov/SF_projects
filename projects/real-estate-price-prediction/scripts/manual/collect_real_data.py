"""
Full-scale real data collection from CIAN
Collects data from 5 cities using Selenium scraper
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
from datetime import datetime
from typing import Dict

import pandas as pd

from src.data_collection.cian_selenium_scraper import CianSeleniumScraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

# Collection configuration
CITIES = {
    "moskva": {"name": "Москва", "pages": 20},
    "spb": {"name": "Санкт-Петербург", "pages": 15},
    "ekaterinburg": {"name": "Екатеринбург", "pages": 10},
    "novosibirsk": {"name": "Новосибирск", "pages": 10},
    "kazan": {"name": "Казань", "pages": 10},
}


def collect_city_data(city_code: str, config: Dict) -> pd.DataFrame:
    """
    Collect data for one city

    Args:
        city_code: City code (e.g., 'moskva')
        config: City configuration

    Returns:
        DataFrame with listings
    """
    logger.info("=" * 70)
    logger.info(f"COLLECTING: {config['name']} ({config['pages']} pages)")
    logger.info("=" * 70)

    try:
        scraper = CianSeleniumScraper(city=city_code, headless=True)
        listings = scraper.scrape_pages(max_pages=config["pages"])
        scraper.close()

        df = pd.DataFrame(listings)
        df["city"] = config["name"]

        logger.info(f"✅ {config['name']}: {len(df)} listings collected")

        return df

    except Exception as e:
        logger.error(f"❌ {config['name']} failed: {e}")
        return pd.DataFrame()


def main():
    """Main collection function"""
    print("\n" + "=" * 70)
    print("REAL DATA COLLECTION FROM CIAN")
    print("=" * 70)

    start_time = datetime.now()
    all_data = []

    # Collect from each city
    for city_code, config in CITIES.items():
        df = collect_city_data(city_code, config)

        if len(df) > 0:
            all_data.append(df)

            # Save intermediate results
            temp_path = Path("data/raw") / f"cian_{city_code}_temp.csv"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(temp_path, index=False, encoding="utf-8-sig")
            logger.info(f"   Saved temp: {temp_path}")

    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Save combined file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("data/raw") / f"cian_listings_real_{timestamp}.csv"
        combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        # Statistics
        print("\n" + "=" * 70)
        print("COLLECTION SUMMARY")
        print("=" * 70)
        print(f"\nTotal listings collected: {len(combined_df):,}")
        print(f"Unique URLs: {combined_df['url'].nunique():,}")
        print(f"Time elapsed: {(datetime.now() - start_time).total_seconds():.1f} seconds")

        print("\nBy city:")
        city_stats = combined_df.groupby("city").agg({"url": "count", "price": "median"}).round(0)
        city_stats.columns = ["Count", "Median Price"]
        print(city_stats)

        print(f"\nSaved to: {output_path}")
        print("=" * 70)

        # Validation
        print("\nDATA VALIDATION:")
        print(f"  Total records: {len(combined_df):,}")
        print(
            f"  Unique URLs: {combined_df['url'].nunique():,} ({combined_df['url'].nunique()/len(combined_df)*100:.1f}%)"
        )

        if "price" in combined_df.columns:
            valid_prices = combined_df["price"].notna().sum()
            print(f"  Valid prices: {valid_prices:,} ({valid_prices/len(combined_df)*100:.1f}%)")

        if "rooms" in combined_df.columns:
            valid_rooms = combined_df["rooms"].notna().sum()
            print(f"  Valid rooms: {valid_rooms:,} ({valid_rooms/len(combined_df)*100:.1f}%)")

        print("\n✅ COLLECTION COMPLETED!")

        return 0

    else:
        print("\n❌ No data collected")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

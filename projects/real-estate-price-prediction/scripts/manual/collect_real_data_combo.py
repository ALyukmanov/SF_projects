"""
COMBO Data Collection from CIAN
5 main cities (30 pages) + 3 new cities (15 pages)
Target: 5,000+ real listings
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

# COMBO CONFIGURATION
CITIES = {
    # Main cities - 30 pages each
    "moskva": {"name": "Москва", "pages": 30},
    "spb": {"name": "Санкт-Петербург", "pages": 30},
    "ekaterinburg": {"name": "Екатеринбург", "pages": 30},
    "novosibirsk": {"name": "Новосибирск", "pages": 30},
    "kazan": {"name": "Казань", "pages": 30},
    # New cities - 15 pages each
    "nizhniy-novgorod": {"name": "Нижний Новгород", "pages": 15},
    "samara": {"name": "Самара", "pages": 15},
    "krasnodar": {"name": "Краснодар", "pages": 15},
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
    print("COMBO DATA COLLECTION FROM CIAN")
    print("=" * 70)
    print("Target: 5,000+ listings from 8 cities")
    print("=" * 70)

    start_time = datetime.now()
    all_data = []

    # Expected totals
    total_pages = sum(c["pages"] for c in CITIES.values())
    expected_listings = total_pages * 28  # ~28 per page

    print("\nConfiguration:")
    print(f"  Cities: {len(CITIES)}")
    print(f"  Total pages: {total_pages}")
    print(f"  Expected listings: ~{expected_listings:,}")
    print(f"  Estimated time: ~{total_pages * 15 / 60:.0f} minutes")
    print()

    # Collect from each city
    for idx, (city_code, config) in enumerate(CITIES.items(), 1):
        print(f"\n[{idx}/{len(CITIES)}] {config['name']}")

        df = collect_city_data(city_code, config)

        if len(df) > 0:
            all_data.append(df)

            # Save intermediate results
            temp_path = Path("data/raw") / f"cian_{city_code}_combo.csv"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(temp_path, index=False, encoding="utf-8-sig")
            logger.info(f"   Saved temp: {temp_path}")

    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates by URL
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=["url"], keep="first")
        duplicates_removed = initial_count - len(combined_df)

        # Save combined file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("data/raw") / f"cian_listings_combo_{timestamp}.csv"
        combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        # Statistics
        print("\n" + "=" * 70)
        print("COMBO COLLECTION SUMMARY")
        print("=" * 70)
        print(f"\nTotal listings collected: {initial_count:,}")
        print(f"Duplicates removed: {duplicates_removed}")
        print(f"Final unique listings: {len(combined_df):,}")
        print(
            f"Unique URLs: {combined_df['url'].nunique():,} ({combined_df['url'].nunique()/len(combined_df)*100:.1f}%)"
        )
        print(f"Time elapsed: {(datetime.now() - start_time).total_seconds():.1f} seconds")
        print(
            f"Collection rate: {len(combined_df) / (datetime.now() - start_time).total_seconds():.1f} listings/sec"
        )

        print("\nBy city:")
        city_stats = combined_df.groupby("city").agg({"url": "count", "price": "median"}).round(0)
        city_stats.columns = ["Count", "Median Price"]
        city_stats = city_stats.sort_values("Count", ascending=False)
        print(city_stats.to_string())

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
            print(
                f"  Price range: {combined_df['price'].min():,.0f} - {combined_df['price'].max():,.0f} rub"
            )
            print(f"  Median price: {combined_df['price'].median():,.0f} rub")

        if "rooms" in combined_df.columns:
            valid_rooms = combined_df["rooms"].notna().sum()
            print(f"  Valid rooms: {valid_rooms:,} ({valid_rooms/len(combined_df)*100:.1f}%)")

        print("\n✅ COMBO COLLECTION COMPLETED!")
        print("Next step: Clean data and run feature engineering")

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

"""
Add latitude / longitude to the scraped restate.ru listings.

The search-results pages the scraper uses only give a city-level address, no
coordinates. Each individual listing page
(``https://www.restate.ru/base/<id>.html``) does carry a point, in the
``data-lat`` / ``data-lng`` attributes of its map ``<div>``. This script
walks the listings we already have, fetches each listing page once over
plain HTTP with a polite delay, parses the coordinates
(``src.data_collection.parsing_utils.parse_listing_coordinates``) and writes
a copy with the coordinates filled in to
``data/interim/restate_<city>_listings_geo.csv``. The original
``restate_<city>_listings.csv`` is left untouched.

It is safe to stop (Ctrl+C) and re-run: every fetched page is cached to
``data/raw/restate_details/coordinates.jsonl`` and already-cached URLs are
skipped. Re-running with nothing left to fetch just rebuilds the ``_geo.csv``
copies from the cache.

Usage:
    python scripts/enrich_coordinates.py                 # all cities
    python scripts/enrich_coordinates.py --city moscow
    python scripts/enrich_coordinates.py --limit 50      # try 50 pages then stop
    python scripts/enrich_coordinates.py --retry-missing # re-fetch pages that
                                                         # gave no coordinates
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data_collection.network import (  # noqa: E402
    build_session,
    fetch_with_retry,
    rotate_user_agent,
)
from src.data_collection.parsing_utils import parse_listing_coordinates  # noqa: E402
from src.data_collection.persistence import JsonlCheckpoint  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("enrich_coordinates")

INTERIM_DIR = _PROJECT_ROOT / "data" / "interim"
CACHE_PATH = _PROJECT_ROOT / "data" / "raw" / "restate_details" / "coordinates.jsonl"

_CITY_FILES = {
    "moscow": "restate_moscow_listings.csv",
    "petersburg": "restate_petersburg_listings.csv",
}


def _target_csvs(city: str) -> list[Path]:
    if city == "all":
        return sorted(
            p for p in INTERIM_DIR.glob("restate_*_listings.csv") if not p.stem.endswith("_geo")
        )
    path = INTERIM_DIR / _CITY_FILES[city]
    return [path] if path.exists() else []


def _geo_path(source_csv: Path) -> Path:
    return source_csv.with_name(source_csv.stem + "_geo.csv")


def _load_cache() -> dict[str, dict]:
    """Return {url: cached_row} from the coordinates cache."""
    cache: dict[str, dict] = {}
    for row in JsonlCheckpoint(CACHE_PATH).read_all():
        url = row.get("url")
        if url:
            cache[url] = row
    return cache


def _fetch_missing(
    urls: list[str],
    cache: dict[str, dict],
    *,
    delay: float,
    limit: int | None,
    retry_missing: bool,
) -> None:
    def needs_fetch(u: str) -> bool:
        if u not in cache:
            return True
        return retry_missing and cache[u].get("latitude") is None

    todo = [u for u in urls if needs_fetch(u)]
    if not todo:
        logger.info("Nothing to fetch — every listing page is already cached.")
        return
    if limit is not None:
        todo = todo[:limit]

    logger.info(
        "Fetching %d listing page(s) (delay ~%.1fs each, ~%.0f min total).",
        len(todo),
        delay,
        len(todo) * (delay + 0.5) / 60,
    )
    checkpoint = JsonlCheckpoint(CACHE_PATH)
    session = build_session()
    found = 0

    for i, url in enumerate(todo, start=1):
        result = fetch_with_retry(session, url, max_retries=2, base_delay=5.0, timeout=20.0)
        coords = parse_listing_coordinates(result.text or "") if result.ok else None
        row = {
            "url": url,
            "latitude": coords[0] if coords else None,
            "longitude": coords[1] if coords else None,
            "http_status": result.status_code,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        checkpoint.append([row])
        cache[url] = row
        if coords:
            found += 1

        if i % 25 == 0 or i == len(todo):
            logger.info("  %d/%d done, %d with coordinates so far.", i, len(todo), found)

        if i < len(todo):
            time.sleep(delay + random.uniform(0.0, 1.0))
            if i % 50 == 0:
                rotate_user_agent(session)

    logger.info("Fetch loop finished: %d/%d pages yielded coordinates.", found, len(todo))


def _write_geo_csvs(csv_paths: list[Path], cache: dict[str, dict]) -> None:
    for path in csv_paths:
        df = pd.read_csv(path)
        if "url" not in df.columns:
            logger.warning("%s has no 'url' column — skipped.", path.name)
            continue

        df["latitude"] = df["url"].map(lambda u: cache.get(u, {}).get("latitude"))
        df["longitude"] = df["url"].map(lambda u: cache.get(u, {}).get("longitude"))
        out_path = _geo_path(path)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        have = df["latitude"].notna().sum()
        logger.info(
            "%s: %d/%d rows have coordinates (%.1f%%).",
            out_path.name,
            have,
            len(df),
            100.0 * have / len(df) if len(df) else 0.0,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--city", choices=["all", "moscow", "petersburg"], default="all")
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Seconds between requests (default: 2.0)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many pages.")
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="Also re-fetch listings that were fetched before but gave no coordinates.",
    )
    args = parser.parse_args()

    csv_paths = _target_csvs(args.city)
    if not csv_paths:
        print(
            f"No interim CSV found for --city {args.city} in {INTERIM_DIR}.\n"
            "Collect listings first:  python -m src.data_collection scrape --source restate --city moscow",
            file=sys.stderr,
        )
        return 2

    urls: list[str] = []
    for path in csv_paths:
        urls.extend(pd.read_csv(path, usecols=["url"])["url"].dropna().tolist())
    urls = list(dict.fromkeys(urls))  # de-dupe, keep order
    logger.info("%d unique listing URL(s) across %d file(s).", len(urls), len(csv_paths))

    cache = _load_cache()
    logger.info("%d listing page(s) already in the cache at %s.", len(cache), CACHE_PATH)

    try:
        _fetch_missing(
            urls,
            cache,
            delay=args.delay,
            limit=args.limit,
            retry_missing=args.retry_missing,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted — writing what was fetched so far.")

    _write_geo_csvs(csv_paths, cache)

    with_coords = sum(1 for u in urls if cache.get(u, {}).get("latitude") is not None)
    print("\n" + "=" * 60)
    print("COORDINATE ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"Listings total     : {len(urls)}")
    print(f"With coordinates   : {with_coords} ({100.0 * with_coords / len(urls):.1f}%)")
    print(f"Cache file         : {CACHE_PATH}")
    print(f"Output CSVs        : {', '.join(_geo_path(p).name for p in csv_paths)}")
    remaining = len(urls) - len([u for u in urls if u in cache])
    if remaining:
        print(f"\nNot fetched yet    : {remaining} — re-run this script to continue.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

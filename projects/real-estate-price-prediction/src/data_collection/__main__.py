"""
CLI entrypoint for the project's real-estate data-ingestion pipeline.

    python -m src.data_collection probe --source restate --city moscow
    python -m src.data_collection probe --source restate --city petersburg
    python -m src.data_collection probe --source cian
    python -m src.data_collection scrape --source restate --city moscow --limit 50
    python -m src.data_collection scrape --source restate --city moscow --resume <run_dir>
    python -m src.data_collection scrape --source restate --city moscow --incremental
    python -m src.data_collection scrape --source restate --city moscow \
        --categories 1_rooms_flats_sale,2_rooms_flats_sale \
        --sort-orders default,price_asc,price_desc --max-pages-per-category 24

See src/data_collection/README.md for the full command reference and the
reasoning behind each source's status (restate = primary/active,
cian = dormant/CAPTCHA-blocked, kept for reference only).
"""

from __future__ import annotations

import argparse
import json
import sys

# Windows consoles often default to a non-UTF-8 codepage (cp1251/cp866),
# which crashes on Cyrillic listing text (addresses, descriptions) the
# moment it's printed. Reconfigure stdout/stderr to UTF-8 defensively so
# this CLI works from a stock `cmd.exe`/PowerShell without requiring the
# caller to set PYTHONIOENCODING themselves.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.schema import find_near_duplicate_candidates, validate_listings_df
from src.data_collection.history import new_current_table, update_history
from src.data_collection.normalize import normalize_restate_records
from src.data_collection.persistence import JsonlCheckpoint
from src.data_collection.sources.restate import (
    RESTATE_CITIES,
    RESTATE_SALE_CATEGORIES,
    RESTATE_SORT_ORDERS,
    CategorySpec,
    RestateScraper,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_HISTORY = PROJECT_ROOT / "data" / "history"

_CATEGORY_BY_SLUG: Dict[str, CategorySpec] = {c.slug: c for c in RESTATE_SALE_CATEGORIES}


def _resolve_categories(categories_csv: Optional[str]) -> Optional[List[CategorySpec]]:
    """Turn a ``--categories`` comma-separated slug list into `CategorySpec`
    objects. Returns ``None`` (all default categories) when unset."""
    if not categories_csv:
        return None
    slugs = [s.strip() for s in categories_csv.split(",") if s.strip()]
    unknown = [s for s in slugs if s not in _CATEGORY_BY_SLUG]
    if unknown:
        raise SystemExit(f"Unknown category slug(s): {unknown}. Known: {sorted(_CATEGORY_BY_SLUG)}")
    return [_CATEGORY_BY_SLUG[s] for s in slugs]


def _resolve_sort_orders(sort_orders_csv: Optional[str]) -> Optional[List[str]]:
    if not sort_orders_csv:
        return None
    names = [s.strip() for s in sort_orders_csv.split(",") if s.strip()]
    unknown = [s for s in names if s not in RESTATE_SORT_ORDERS]
    if unknown:
        raise SystemExit(f"Unknown sort order(s): {unknown}. Known: {sorted(RESTATE_SORT_ORDERS)}")
    return names


def _cmd_probe(args: argparse.Namespace) -> int:
    if args.source == "restate":
        scraper = RestateScraper(city=args.city)
        category = scraper.categories[0]
        url = scraper.category_url(category.slug, page=1)
        print(f"PROBE restate: GET {url}")
        items, ok, reason = scraper._fetch_and_parse(url, category)  # single, sparing request
        print(f"  ok={ok} reason={reason} items_found={len(items)}")
        if items:
            print(f"  sample item url: {items[0]['url']}")
            print(f"  sample item name: {items[0]['raw_item'].get('name')}")
        return 0 if ok and items else 1

    if args.source == "cian":
        from src.data_collection.cian_scraper import CianScraper

        scraper = CianScraper(city="moskva")
        url = scraper.base_url
        print(f"PROBE cian: GET {url}")
        listings = scraper.scrape_page(url)
        print(f"  items_found={len(listings)}")
        print(
            "  NOTE: CIAN is a known dead end (CAPTCHA redirect) as of 2026-07-29 and "
            "2026-08-24 -- this probe exists so the status can be re-checked later "
            "without writing new code, not because a bypass is planned."
        )
        return 0 if listings else 1

    print(f"Unknown source: {args.source}", file=sys.stderr)
    return 2


def _run_dir_for(source: str, city: str, resume: Optional[str]) -> Path:
    if resume:
        d = Path(resume)
        if not d.exists():
            raise SystemExit(f"--resume path does not exist: {d}")
        return d
    date_str = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%H%M%S")
    d = DATA_RAW / source / date_str / f"{city}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_previous_run_dir(source: str, city: str, exclude: Path) -> Optional[Path]:
    base = DATA_RAW / source
    if not base.exists():
        return None
    candidates = []
    for date_dir in base.iterdir():
        if not date_dir.is_dir():
            continue
        for run_dir in date_dir.iterdir():
            if not run_dir.is_dir() or run_dir.resolve() == exclude.resolve():
                continue
            if run_dir.name.startswith(city + "_") and (run_dir / "listings_raw.jsonl").exists():
                candidates.append(run_dir)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _price_by_url(raw_records: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for r in raw_records:
        item = r.get("raw_item", {}) if isinstance(r.get("raw_item"), dict) else {}
        offers = item.get("offers", {}) if isinstance(item.get("offers"), dict) else {}
        try:
            out[r["url"]] = float(offers.get("price"))
        except (TypeError, ValueError):
            out[r["url"]] = None
    return out


def _history_dir_for(source: str, city: str) -> Path:
    return DATA_HISTORY / source / city


def _update_history_table(
    source: str, city: str, normalized: List[Dict[str, Any]], run_dir_name: str
) -> None:
    """Fold this run's normalized records into the persistent
    ``listings_current``/``listing_events`` tables for *source*/*city* (see
    src/data_collection/history.py). Safe to call repeatedly across separate
    days' runs -- state carries forward, nothing is overwritten wholesale.
    """
    hist_dir = _history_dir_for(source, city)
    hist_dir.mkdir(parents=True, exist_ok=True)
    current_path = hist_dir / "listings_current.parquet"
    events_path = hist_dir / "listing_events.jsonl"

    current_df = pd.read_parquet(current_path) if current_path.exists() else new_current_table()
    run_at = datetime.now(timezone.utc).isoformat()
    updated_df, events_df = update_history(
        current_df, normalized, run_id=run_dir_name, run_at=run_at
    )

    updated_df.to_parquet(current_path, index=False)
    if not events_df.empty:
        with events_path.open("a", encoding="utf-8") as f:
            for _, row in events_df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False, default=str) + "\n")

    status_counts = updated_df["status"].value_counts().to_dict() if not updated_df.empty else {}
    print(f"\nHistory updated: {current_path}")
    print(f"  Status counts (all-time tracked listings): {status_counts}")
    if not events_df.empty:
        print(f"  Events this run: {len(events_df)} -> {events_path}")


def _cmd_scrape(args: argparse.Namespace) -> int:
    if args.source != "restate":
        print("Only --source restate is currently wired for 'scrape'.", file=sys.stderr)
        return 2

    run_dir = _run_dir_for(args.source, args.city, args.resume)
    checkpoint_path = run_dir / "listings_raw.jsonl"
    print(f"Run directory: {run_dir}")
    print(f"Raw checkpoint: {checkpoint_path}")

    categories = _resolve_categories(args.categories)
    sort_orders = _resolve_sort_orders(args.sort_orders)
    try:
        scraper = RestateScraper(
            city=args.city,
            categories=categories,
            sort_orders=sort_orders,
            checkpoint_path=str(checkpoint_path),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    raw_records, stats = scraper.scrape(
        max_pages_per_category=args.max_pages_per_category, limit=args.limit
    )

    run_metadata = {
        "source": args.source,
        "city": args.city,
        "categories": [c.slug for c in scraper.categories],
        "sort_orders": scraper.sort_orders,
        "max_pages_per_category": args.max_pages_per_category,
        "limit": args.limit,
        "invoked_at": datetime.now(timezone.utc).isoformat(),
        "resumed_from": args.resume,
        "stats_this_invocation": {
            "pages_fetched": stats.pages_fetched,
            "pages_failed": stats.pages_failed,
            "items_seen_on_pages": stats.items_seen_on_pages,
            "items_new": stats.items_new,
            "items_duplicate": stats.items_duplicate,
        },
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if stats.errors:
        with (run_dir / "errors.jsonl").open("a", encoding="utf-8") as f:
            for err in stats.errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")

    print(f"Pages fetched: {stats.pages_fetched} (failed: {stats.pages_failed})")
    print(f"Items seen on pages: {stats.items_seen_on_pages}")
    print(
        f"New this invocation: {stats.items_new}  Duplicate this invocation: {stats.items_duplicate}"
    )

    checkpoint = JsonlCheckpoint(checkpoint_path)
    all_raw = list(checkpoint.read_all())
    city_name_ru = RESTATE_CITIES[args.city]["name_ru"]
    normalized = normalize_restate_records(
        all_raw, city_name=city_name_ru, region_name=city_name_ru
    )
    print(f"Normalized records (full run dir content): {len(normalized)} (from {len(all_raw)} raw)")

    df = pd.DataFrame(normalized)
    report = validate_listings_df(df, strict=False)
    print(str(report))

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_INTERIM / f"restate_{args.city}_listings.csv"
    parquet_path = DATA_INTERIM / f"restate_{args.city}_listings.parquet"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_parquet(parquet_path, index=False)
    print(f"Interim CSV:     {csv_path} ({len(df)} rows)")
    print(f"Interim Parquet: {parquet_path}")

    print("\nField coverage:")
    for col in [
        "price",
        "total_area",
        "living_area",
        "rooms",
        "floor",
        "floors_total",
        "address",
        "published_at",
        "latitude",
        "year_built",
        "seller_type",
    ]:
        if col in df.columns:
            pct = 100.0 * df[col].notna().mean() if len(df) else 0.0
            print(f"  {col:15s}: {pct:5.1f}%")

    near_dupes = find_near_duplicate_candidates(df)
    if not near_dupes.empty:
        print(
            f"\nNear-duplicate candidates: {near_dupes['url'].nunique()} row(s) across "
            f"{near_dupes['group_id'].nunique()} group(s) -- NOT auto-removed, see "
            f"{run_dir / 'near_duplicate_candidates.csv'}"
        )
        near_dupes.to_csv(
            run_dir / "near_duplicate_candidates.csv", index=False, encoding="utf-8-sig"
        )

    if args.track_history:
        _update_history_table(args.source, args.city, normalized, run_dir_name=run_dir.name)

    if args.incremental:
        prev_dir = _find_previous_run_dir(args.source, args.city, exclude=run_dir)
        if prev_dir is None:
            print(
                "\n--incremental requested but no previous completed run found for this source/city."
            )
        else:
            print(f"\nIncremental comparison vs previous run: {prev_dir}")
            prev_raw = list(JsonlCheckpoint(prev_dir / "listings_raw.jsonl").read_all())
            prev_by_id = _price_by_url(prev_raw)
            cur_by_id = _price_by_url(all_raw)

            new_ids = set(cur_by_id) - set(prev_by_id)
            missing_ids = set(prev_by_id) - set(cur_by_id)
            common_ids = set(cur_by_id) & set(prev_by_id)
            updated_ids = {i for i in common_ids if cur_by_id[i] != prev_by_id[i]}
            unchanged_ids = common_ids - updated_ids

            print(f"  Seen previously: {len(prev_by_id)}")
            print(f"  New:             {len(new_ids)}")
            print(f"  Unchanged:       {len(unchanged_ids)}")
            print(f"  Updated (price): {len(updated_ids)}")
            print(f"  Missing (gone):  {len(missing_ids)}")

            incremental_report = {
                "compared_to": str(prev_dir),
                "seen_previously": len(prev_by_id),
                "new": len(new_ids),
                "unchanged": len(unchanged_ids),
                "updated": len(updated_ids),
                "missing": len(missing_ids),
                "updated_examples": [
                    {"url": i, "old_price": prev_by_id[i], "new_price": cur_by_id[i]}
                    for i in list(updated_ids)[:10]
                ],
            }
            (run_dir / "incremental_report.json").write_text(
                json.dumps(incremental_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.data_collection")
    sub = p.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="One sparing request to check a source is reachable.")
    probe.add_argument("--source", required=True, choices=["restate", "cian"])
    probe.add_argument("--city", default="moscow", choices=sorted(RESTATE_CITIES))
    probe.set_defaults(func=_cmd_probe)

    scrape = sub.add_parser("scrape", help="Collect listings from a source.")
    scrape.add_argument("--source", required=True, choices=["restate"])
    scrape.add_argument("--city", default="moscow", choices=sorted(RESTATE_CITIES))
    scrape.add_argument(
        "--categories",
        default=None,
        help=f"Comma-separated category slugs (default: all). Known: {sorted(_CATEGORY_BY_SLUG)}",
    )
    scrape.add_argument(
        "--sort-orders",
        default=None,
        help=f"Comma-separated sort orders to sweep per category (default: 'default' only). "
        f"Known: {sorted(RESTATE_SORT_ORDERS)}. Sweeping more than one accesses more of a "
        f"large category's inventory past the site's ~24-page crawl-depth cap.",
    )
    scrape.add_argument(
        "--limit", type=int, default=None, help="Stop after this many new listings."
    )
    scrape.add_argument("--max-pages-per-category", type=int, default=3)
    scrape.add_argument(
        "--resume", default=None, help="Path to an existing run directory to resume into."
    )
    scrape.add_argument(
        "--incremental", action="store_true", help="Compare against the most recent previous run."
    )
    scrape.add_argument(
        "--track-history",
        action="store_true",
        help=(
            "Fold this run into a persistent cross-run listings_current/listing_events "
            "table under data/history/<source>/<city>/ (NEW/SEEN_UNCHANGED/UPDATED/"
            "NOT_SEEN_THIS_RUN/CONFIRMED_REMOVED — see src/data_collection/history.py). "
            "Unlike --incremental (a one-off two-run diff), this accumulates across every "
            "run this flag is passed on."
        ),
    )
    scrape.set_defaults(func=_cmd_scrape)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

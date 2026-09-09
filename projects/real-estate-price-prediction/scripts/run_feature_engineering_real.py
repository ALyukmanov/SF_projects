"""
Feature engineering pipeline for real listings data.
Usage: python scripts/run_feature_engineering_real.py [--input path] [--output path]

Primary real-data source (2026-08-25+): all ``restate_*_listings.csv`` files
in ``data/interim/`` (one per city, written by
``python -m src.data_collection scrape --source restate``) are loaded and
concatenated -- restate.ru is the project's active source, see
src/data_collection/README.md. Falls back to the legacy single latest
``cian_listings*.csv`` in ``data/raw/`` (CIAN is dormant/CAPTCHA-blocked but
its normalized schema is still compatible, kept for reference only) if no
restate data is found. Either path then runs the same cleaning, economic
feature engineering and feature engineering steps, and saves the enriched
DataFrame to data/processed/real_estate_engineered.csv.

NOTE (known gap): each ``restate_<city>_listings.csv``
reflects only the most recent single ``scrape`` invocation's own run
directory, not an accumulation across separate collection runs on different
days -- there is no cross-run merge/history layer yet. The full raw archives
under data/raw/restate/<date>/<city>_<time>/ do persist per run, but this
script currently ingests only the latest interim CSV per city.

FAIL-CLOSED BY DEFAULT: if no raw CSV exists in data/raw/, this script exits
with an error instead of silently generating synthetic data. Pass
``--allow-synthetic`` (or run ``make demo-data``) to explicitly opt into
200 rows of clearly-labelled synthetic demo data — the output is then
written to ``real_estate_engineered.synthetic.csv`` (not the "real" output
filename), with an ``is_synthetic=True`` column embedded in the data itself
and a ``*.synthetic.meta.json`` sidecar recording generation parameters.
See DATA_CARD.md for why this
matters: earlier versions of this script silently produced synthetic data
whenever no real CSV was found, which is how the committed model artefact
ended up trained on synthetic data without that being obvious from the
pipeline itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable regardless of CWD
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent  # .../scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent  # .../final_project/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Now safe to import project modules
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd

from src.features.feature_engineering import FeatureEngineer
from src.features.geo_features import GeoFeatureBuilder, coverage_report
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.economic_features import EconomicFeatureEngineer
from src.preprocessing.imputer import GroupMedianImputer
from src.utils.logger import get_logger

logger = get_logger("run_feature_engineering_real")

# ---------------------------------------------------------------------------
# Cities used in synthetic demo data
# ---------------------------------------------------------------------------
_CITIES = [
    "москва",
    "санкт-петербург",
    "екатеринбург",
    "новосибирск",
    "казань",
    "нижний новгород",
    "самара",
    "краснодар",
]

_CITY_BASE_PPM = {
    "москва": 280_000,
    "санкт-петербург": 180_000,
    "екатеринбург": 90_000,
    "новосибирск": 85_000,
    "казань": 80_000,
    "нижний новгород": 75_000,
    "самара": 72_000,
    "краснодар": 78_000,
}

_BUILDING_TYPES = ["panel", "brick", "monolith", "block"]


def _generate_synthetic_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame of *n* synthetic real-estate listings."""
    rng = np.random.default_rng(seed)

    cities = rng.choice(_CITIES, size=n)
    rooms = rng.integers(0, 5, size=n)  # 0 = studio
    total_area = rng.uniform(20, 150, size=n).round(1)
    floors_total = rng.integers(5, 25, size=n)
    floor = np.array([rng.integers(1, ft + 1) for ft in floors_total])
    year_built = rng.integers(1960, 2024, size=n)

    base_ppm = np.array([_CITY_BASE_PPM[c] for c in cities], dtype=float)
    # Add noise ±20 %
    price = (base_ppm * total_area * rng.uniform(0.80, 1.20, size=n)).round(-3)

    building_type = rng.choice(_BUILDING_TYPES, size=n)

    # Dates spread over 2023-2025
    days = rng.integers(0, 365 * 2, size=n)
    base_date = pd.Timestamp("2023-01-01")
    date_published = [str(base_date + pd.Timedelta(days=int(d)))[:10] for d in days]
    scraped_at = date_published  # same for synthetic data

    urls = [f"https://cian.ru/sale/flat/{100_000 + i}/" for i in range(n)]
    addresses = [f"ул. Примерная, д. {rng.integers(1, 200)}, {c}" for i, c in enumerate(cities)]

    return pd.DataFrame(
        {
            "price": price,
            "rooms": rooms,
            "total_area": total_area,
            "floor": floor,
            "floors_total": floors_total,
            "address": addresses,
            "city": cities,
            "year_built": year_built,
            "building_type": building_type,
            "url": urls,
            "scraped_at": scraped_at,
            "date_published": date_published,
            # Explicit provenance flag embedded IN the data itself (not just the
            # filename/logs) so any downstream consumer of this DataFrame can
            # detect synthetic rows programmatically. See DATA_CARD.md.
            "is_synthetic": True,
        }
    )


def _find_latest_csv(directory: Path, pattern: str = "cian_listings*.csv") -> Path | None:
    """Return the most recently modified CSV matching *pattern* in *directory*."""
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_restate_data(interim_dir: Path) -> pd.DataFrame | None:
    """Load and concatenate every ``restate_*_listings.csv`` in *interim_dir*
    (one file per city, see ``src/data_collection/__main__.py::_cmd_scrape``).

    Returns ``None`` if no such file exists. Rows are deduplicated by ``url``
    later, in :class:`~src.preprocessing.cleaner.DataCleaner`, so a listing
    appearing in more than one city file (should not normally happen, since
    restate.ru listing URLs are per-listing, not per-city) is not double
    counted.
    """
    if not interim_dir.exists():
        return None

    # Prefer the coordinate-enriched copies (restate_<city>_listings_geo.csv,
    # written by scripts/enrich_coordinates.py) over the plain ones, per city,
    # so latitude/longitude reach the training pipeline. Falls back to the
    # plain file for any city that has not been through coordinate enrichment.
    plain = sorted(interim_dir.glob("restate_*_listings.csv"))
    if not plain:
        return None
    csv_paths = []
    for p in plain:
        geo = p.with_name(p.stem + "_geo.csv")
        csv_paths.append(geo if geo.is_file() else p)

    frames = []
    for p in csv_paths:
        city_df = pd.read_csv(p)
        logger.info("Loaded %d row(s) from %s", len(city_df), p)
        frames.append(city_df)
    df = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined restate real-data: %d row(s) total across %d file(s).", len(df), len(csv_paths)
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature engineering pipeline for CIAN real-estate data."
    )
    parser.add_argument(
        "--input",
        default=str(_PROJECT_ROOT / "data" / "raw"),
        help="Directory containing raw CSV files (default: data/raw/)",
    )
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "data" / "processed"),
        help="Directory to write the engineered CSV (default: data/processed/)",
    )
    parser.add_argument(
        "--skip-geo",
        action="store_true",
        help=(
            "Skip the OpenStreetMap geo-feature step (nearest-POI distances / "
            "counts from data/external/osm_poi.csv). Geo features are added "
            "automatically when that file exists; use this to reproduce the "
            "pre-geo feature set."
        ),
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help=(
            "Explicitly allow generating 200 rows of synthetic demo data when "
            "no real cian_listings*.csv is found in --input. Without this "
            "flag, the script fails closed with an error instead of silently "
            "fabricating data. Equivalent to `make demo-data`."
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    interim_dir = _PROJECT_ROOT / "data" / "interim"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load data (fail closed if no real data and no explicit opt-in)
    # Priority: restate.ru (active source) > CIAN (dormant, legacy) > synthetic.
    # ------------------------------------------------------------------
    is_synthetic = False
    data_source = None
    df = _load_restate_data(interim_dir)
    if df is not None:
        data_source = "restate"
        logger.info("Loaded %d rows, %d columns from restate.ru interim data.", *df.shape)
        is_synthetic = bool(df.get("is_synthetic", pd.Series([False])).any())
    else:
        csv_path = _find_latest_csv(input_dir)
        if csv_path is not None:
            data_source = "cian"
            logger.info("No restate data found; loading legacy CIAN data from %s", csv_path)
            df = pd.read_csv(csv_path)
            logger.info("Loaded %d rows, %d columns.", *df.shape)
            is_synthetic = bool(df.get("is_synthetic", pd.Series([False])).any())
            if is_synthetic:
                logger.warning(
                    "Input CSV '%s' is itself flagged is_synthetic=True. "
                    "Downstream artefacts trained from this run are NOT real-data artefacts.",
                    csv_path,
                )
        elif args.allow_synthetic:
            data_source = "synthetic"
            logger.warning(
                "No restate_*_listings.csv found in '%s' and no cian_listings*.csv found in "
                "'%s'. --allow-synthetic was passed: generating SYNTHETIC demo data (200 rows).",
                interim_dir,
                input_dir,
            )
            df = _generate_synthetic_data(n=200)
            is_synthetic = True
            logger.info("Synthetic data generated: %d rows, %d columns.", *df.shape)
        else:
            print(
                "\nERROR: no real data found and synthetic generation was not requested.\n"
                f"  Looked for '{interim_dir}/restate_*_listings.csv' — none found.\n"
                f"  Looked for '{input_dir}/cian_listings*.csv' — none found.\n\n"
                "This script fails closed by design (see DATA_CARD.md): it will not\n"
                "silently fabricate training data.\n\n"
                "To proceed, either:\n"
                "  1. Collect real data first, e.g.:\n"
                "       python -m src.data_collection scrape --source restate --city moscow\n"
                "     (see README.md 'Reproducing the data'), or\n"
                "  2. Explicitly request synthetic demo data:\n"
                "       python scripts/run_feature_engineering_real.py --allow-synthetic\n"
                "     or:\n"
                "       make demo-data\n",
                file=sys.stderr,
            )
            raise SystemExit(2)

    # ------------------------------------------------------------------
    # Step 2: Clean (structural only — dedup/cast/range-filter. Does NOT
    # impute rooms/total_area/floor/floors_total: that must happen AFTER a
    # train/test split, fit on train only, or it leaks holdout information
    # into train — see src/preprocessing/imputer.py.)
    # ------------------------------------------------------------------
    logger.info("Running DataCleaner...")
    cleaner = DataCleaner()
    df = cleaner.clean(df)
    logger.info("After cleaning: %d rows, %d columns.", *df.shape)

    # ------------------------------------------------------------------
    # Step 3: Economic features (year-keyed lookup/fallback table, not a
    # dataset statistic — safe to compute before any split).
    # ------------------------------------------------------------------
    logger.info("Running EconomicFeatureEngineer (use_api=False)...")
    eco_eng = EconomicFeatureEngineer(use_api=False)
    df = eco_eng.add_economic_features(df)
    logger.info("Economic features added.")

    # ------------------------------------------------------------------
    # Step 3b: Geo features from OpenStreetMap POI (nearest-POI distances +
    # counts within a radius, per city). Like the economic features these are
    # an external reference join with nothing fitted from the listings, so
    # computing them here — before any split — is leakage-free. Listings
    # without coordinates keep the columns with sentinel distances / zero
    # counts and has_coordinates=0 (never dropped). Skipped automatically if
    # data/external/osm_poi.csv is absent, or with --skip-geo.
    # ------------------------------------------------------------------
    geo_builder = None if args.skip_geo else GeoFeatureBuilder()
    if geo_builder is not None and geo_builder.available:
        logger.info("Adding OSM geo features | deduped POI counts=%s", geo_builder.poi_summary())
        df = geo_builder.transform(df)
        logger.info(
            "Geo feature coverage:\n%s",
            coverage_report(df).to_string(index=False),
        )
    elif args.skip_geo:
        logger.info("--skip-geo: OSM geo features not added.")
    else:
        logger.warning(
            "data/external/osm_poi.csv not found — geo features not added. "
            "Run scripts/prepare_osm_poi.py first if you want them."
        )

    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = is_synthetic
    if "data_source" not in df.columns:
        df["data_source"] = data_source

    # ------------------------------------------------------------------
    # Step 4: Save the CLEANED dataset — the canonical input for training
    # (scripts/run_model_training_real.py) and leakage verification
    # (scripts/verify_split_leakage.py). May still contain NaN in
    # rooms/total_area/floor/floors_total for genuinely missing source rows
    # — those are filled by a GroupMedianImputer fit on the train split only,
    # inside src/data/split_pipeline.py, never here.
    # ------------------------------------------------------------------
    if is_synthetic:
        cleaned_output_path = output_dir / "real_estate_cleaned.synthetic.csv"
    else:
        cleaned_output_path = output_dir / "real_estate_cleaned.csv"
    df.to_csv(cleaned_output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved cleaned data to %s (is_synthetic=%s)", cleaned_output_path, is_synthetic)

    # ------------------------------------------------------------------
    # Step 5: ALSO save a fully-featurized, fully-imputed snapshot for EDA /
    # the Streamlit dashboard's analytics tab — NOT for model training or
    # evaluation. This fits the imputer on the WHOLE dataset (there is no
    # train/test split at this stage), which is fine for descriptive
    # analytics but would be leakage if used to build a train/holdout split
    # afterwards. Any script measuring model quality must instead load the
    # cleaned CSV above and go through split_impute_featurize().
    # ------------------------------------------------------------------
    eda_imputer = GroupMedianImputer().fit(df)
    df_eda = eda_imputer.transform(df)
    feat_eng = FeatureEngineer()
    df_eda = feat_eng.create_features(df_eda)

    if is_synthetic:
        output_path = output_dir / "real_estate_engineered.synthetic.csv"
    else:
        output_path = output_dir / "real_estate_engineered.csv"
    df_eda.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(
        "Saved fully-imputed EDA-only snapshot to %s (is_synthetic=%s) — "
        "NOT leakage-safe for train/test evaluation, see step 4's cleaned CSV for that.",
        output_path,
        is_synthetic,
    )
    df = df_eda

    if is_synthetic:
        meta_path = output_dir / "real_estate_engineered.synthetic.meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "is_synthetic": True,
                    "generator": "_generate_synthetic_data (scripts/run_feature_engineering_real.py)",
                    "n_rows": int(len(df)),
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "note": (
                        "Demo data ONLY. Not scraped from CIAN. Do not present "
                        "results derived from this file as real-market findings."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote synthetic-data metadata sidecar to %s", meta_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY" + (" [SYNTHETIC/DEMO DATA]" if is_synthetic else ""))
    print("=" * 60)
    print(f"Data source: {data_source}")
    print(f"Rows   : {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Output : {output_path}")
    print("\nFeature list:")
    for i, col in enumerate(df.columns, 1):
        dtype_str = str(df[col].dtype)
        null_count = df[col].isna().sum()
        print(f"  {i:3d}. {col:<30} {dtype_str:<12} nulls={null_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()

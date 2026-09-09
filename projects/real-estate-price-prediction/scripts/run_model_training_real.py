"""
Model training pipeline for Real Estate Price Prediction.
Usage: python scripts/run_model_training_real.py [--model linear_regression] [--input path]
(--model defaults to linear_regression, the current best per
reports/model_comparison.csv; pass --model xgboost/random_forest/etc. to
train a different model type for comparison.)

Finds the latest engineered CSV in data/processed/, trains the requested
model, evaluates it, and saves the artefact to models/.

FAIL-CLOSED BY DEFAULT: if no engineered CSV exists in data/processed/, this
script exits with an error instead of silently generating synthetic training
data. Pass ``--allow-synthetic`` (or run ``make train-demo``) to explicitly
opt into training on synthetic demo data. Whenever training data is
synthetic (either freshly generated, or loaded from an existing
``*.synthetic.csv`` / any file with an ``is_synthetic=True`` column), the
saved model artefact filename is tagged ``..._synthetic_...pkl`` and its
metadata carries ``is_synthetic=True``, ``data_source``, a dataset SHA-256
hash, row/column counts and library versions — see DATA_CARD.md for why
this matters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable regardless of CWD
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import pandas as pd
import sklearn
import xgboost

from src.data.split_pipeline import split_impute_featurize
from src.features.feature_engineering import FEATURE_SCHEMA_VERSION, FeatureEngineer
from src.models.evaluator import ModelEvaluator
from src.models.trainer import ModelTrainer
from src.utils.logger import get_logger

logger = get_logger("run_model_training_real")

_SUPPORTED_MODELS = (
    "xgboost",
    "random_forest",
    "lightgbm",
    "catboost",
    "linear_regression",
    "ridge",
)


def _find_latest_csv(directory: Path, pattern: str = "*.csv") -> Path | None:
    """Return the most recently modified CSV matching *pattern* in *directory*."""
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_data_source(is_synthetic: bool, df: pd.DataFrame) -> str:
    """Determine the provenance label recorded in the trained artefact's metadata.

    Real data path prefers the ``data_source`` column written by
    ``run_feature_engineering_real.py`` (2026-08-25+) over guessing. If that
    column is present but contains MORE THAN ONE distinct value (e.g. an
    accidental restate+cian concatenation), this is a genuinely mixed
    dataset -- picking the majority label silently would misrepresent the
    other source's rows, so a warning is logged naming every value found,
    and the label itself is set to "mixed:<sorted values>" rather than
    picking one and hiding the rest.
    """
    if is_synthetic:
        return "synthetic_demo_generator"
    if "data_source" in df.columns and df["data_source"].notna().any():
        values = sorted(df["data_source"].dropna().unique().tolist())
        if len(values) > 1:
            logger.warning(
                "Dataset has MIXED data_source values %s (not a single clean source) -- "
                "recording as 'mixed:...' rather than silently picking the majority label.",
                values,
            )
            return "mixed:" + ",".join(values)
        return str(values[0])
    # Engineered CSV predates the data_source column (pre-2026-08-25) --
    # every real source at that time was CIAN.
    return "cian_scraper_manual_run"


def _git_commit_short() -> str | None:
    """Best-effort short git commit hash; never raises, returns None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _geo_lineage(df: pd.DataFrame, feature_names: list[str]) -> dict:
    """Geo-enrichment provenance for the artefact metadata.

    ``geo_enabled`` is driven by the model's actual ``feature_names`` (the
    ground truth), not just by columns being present in the frame.
    ``osm_poi_manifest`` echoes data/external/osm_poi_manifest.json's
    generation timestamp + source extracts so a served model records which
    OSM snapshot its geo features came from.
    """
    from src.features.geo_features import GEO_FEATURE_COLUMNS

    geo_enabled = any(c in feature_names for c in GEO_FEATURE_COLUMNS)
    coverage = None
    if "has_coordinates" in df.columns:
        n = int(len(df))
        hc = pd.to_numeric(df["has_coordinates"], errors="coerce")
        with_coords = int((hc == 1).sum())
        coverage = {
            "rows": n,
            "with_coordinates": with_coords,
            "coordinate_coverage_pct": round(100.0 * with_coords / n, 2) if n else None,
        }
    manifest_path = _PROJECT_ROOT / "data" / "external" / "osm_poi_manifest.json"
    osm_manifest = None
    if manifest_path.is_file():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            osm_manifest = {
                "generated_at": m.get("generated_at"),
                "row_count": m.get("row_count"),
                "counts_by_category": m.get("counts_by_category"),
                "sources": [
                    {"city": s.get("city"), "file": s.get("file"), "sha256": s.get("sha256")}
                    for s in m.get("sources", [])
                ],
            }
        except Exception:  # noqa: BLE001
            osm_manifest = {"error": "osm_poi_manifest.json present but unreadable"}
    return {
        "geo_enabled": geo_enabled,
        "coordinate_coverage": coverage,
        "osm_poi_manifest": osm_manifest,
    }


def _build_lineage_metadata(
    df: pd.DataFrame,
    resolved_data_source: str,
    is_synthetic: bool,
    dataset_path_repo_relative: str,
    dataset_sha256: str | None,
    dataset_rows: int,
    dataset_columns: int,
    training_start: datetime,
    training_end: datetime,
    split_strategy: str,
    trainer_params: dict,
    params_from_study: str | None,
    feature_names: "list[str] | tuple[str, ...]" = (),
) -> dict:
    """Assemble the full lineage/provenance dict embedded in a saved artefact.

    Covers version/timestamp/git revision, dataset fingerprint and coverage,
    target definition, split method, seed, hyperparameters, and library
    versions -- see scripts/promote_model.py's validation checks for why
    each field exists.
    (feature_names and metrics themselves are attached separately by
    ModelTrainer.save()/fit(), not duplicated here.)
    """
    city_coverage = df["city"].value_counts().to_dict() if "city" in df.columns else None
    category_coverage = (
        df["source_category"].value_counts().to_dict() if "source_category" in df.columns else None
    )
    return {
        "data_source": resolved_data_source,
        "is_synthetic": is_synthetic,
        "dataset_path": dataset_path_repo_relative,
        "dataset_sha256": dataset_sha256,
        "dataset_rows": int(dataset_rows),
        "dataset_columns": int(dataset_columns),
        "city_coverage": city_coverage,
        "category_coverage": category_coverage,
        "target": "price",
        "target_transform": "log1p",
        "training_start": training_start.isoformat(),
        "training_end": training_end.isoformat(),
        "split_strategy": split_strategy,
        "random_seed": 42,
        "hyperparameters": trainer_params or "library_defaults",
        "tuned_from_study": params_from_study,
        "library_versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "git_commit": _git_commit_short(),
        **_geo_lineage(df, feature_names),
    }


def _generate_synthetic_engineered_data(n: int = 400) -> pd.DataFrame:
    """Explicitly-opted-in synthetic data path (used only with --allow-synthetic
    when no engineered CSV exists yet). Mirrors run_feature_engineering_real.py.
    """
    import importlib.util

    from src.preprocessing.cleaner import DataCleaner
    from src.preprocessing.economic_features import EconomicFeatureEngineer

    fe_script = _SCRIPT_DIR / "run_feature_engineering_real.py"
    spec = importlib.util.spec_from_file_location("fe_real", fe_script)
    fe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe_mod)

    df = fe_mod._generate_synthetic_data(n=n)

    cleaner = DataCleaner()
    df = cleaner.clean(df)

    eco_eng = EconomicFeatureEngineer(use_api=False)
    df = eco_eng.add_economic_features(df)

    feat_eng = FeatureEngineer()
    df = feat_eng.create_features(df)
    df["is_synthetic"] = True

    # Persist for future runs — synthetic filename, never the "real" name.
    processed_dir = _PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "real_estate_engineered.synthetic.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.warning("SYNTHETIC engineered data saved to %s (NOT real CIAN data).", out_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a price prediction model on engineered CIAN data."
    )
    parser.add_argument(
        "--model",
        choices=list(_SUPPORTED_MODELS),
        default="linear_regression",
        help=(
            "Model type to train (default: linear_regression — the model "
            "reports/model_comparison.csv's 5-fold CV + holdout comparison "
            "found best on this dataset, MAE ~31%% lower than xgboost with a "
            "far simpler model; NOT xgboost, to avoid silently regressing "
            "the production artefact back to a worse model if this script is "
            "re-run without an explicit --model. See MODEL_CARD.md "
            "'Ограничения' for the drift risk this default is guarding "
            "against, and its own limits — this is a manual default, not an "
            "automated 'pick whatever the comparison says is best' guard.)"
        ),
    )
    parser.add_argument(
        "--input",
        default=str(_PROJECT_ROOT / "data" / "processed"),
        help="Directory containing engineered CSV files (default: data/processed/)",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help=(
            "Explicitly allow training on synthetic demo data — either an "
            "existing *.synthetic.csv / is_synthetic=True file, or (if no "
            "engineered CSV exists at all) freshly generated synthetic data. "
            "Without this flag, the script fails closed. Equivalent to "
            "`make train-demo`."
        ),
    )
    parser.add_argument(
        "--split-strategy",
        choices=["random", "group", "location"],
        default="location",
        help=(
            "'location' (default): GroupShuffleSplit on building-level groups "
            "(src.data.schema.build_location_groups) -- every listing sharing a "
            "rounded coordinate stays on one side. This is the honest evaluation "
            "for the geo model and the strategy behind the current promotion "
            "candidate. 'group': only near-duplicate listings isolated "
            "(src.data.schema.build_split_groups) -- more optimistic once "
            "coordinates exist. 'random': plain 80/20, diagnostic only."
        ),
    )
    parser.add_argument(
        "--params-from-study",
        default=None,
        metavar="MODEL_NAME",
        help=(
            "Load tuned hyperparameters for MODEL_NAME from "
            "reports/tuning_study_real.json (produced by scripts/tune_models_real.py) "
            "instead of using --model's untuned defaults. MODEL_NAME must match both "
            "--model and a key in the study file (e.g. --model xgboost "
            "--params-from-study xgboost)."
        ),
    )
    parser.add_argument(
        "--params-from-manifest",
        action="store_true",
        help=(
            "Reuse the hyperparameters recorded in models/current_model.json. Use this "
            "when the geo-feature evaluation showed the currently-promoted "
            "hyperparameters still generalise best and re-tuning did not beat them "
            "(see reports/geo_final_candidates.json). Mutually exclusive with "
            "--params-from-study."
        ),
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Update models/current_model.json to point at this newly trained "
            "artefact, so API/dashboard start serving it after a restart. "
            "For REAL (non-synthetic) data this defaults to OFF: a freshly "
            "trained real-data model is saved to disk either way, but does "
            "NOT become the serving model until someone deliberately reviews "
            "its metrics and passes --promote (or hand-edits "
            "models/current_model.json) — training alone is not validation. "
            "Synthetic/demo runs (--allow-synthetic) keep the historical "
            "always-promote behaviour, since there is nothing to review there."
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    models_dir = _PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    training_start = datetime.now()

    # ------------------------------------------------------------------
    # Step 1: Load data — fail closed if none exists and synthetic
    # generation was not explicitly requested. Prefers the "cleaned" CSV
    # (structural cleaning only, NaNs preserved in rooms/total_area/floor/
    # floors_total — see run_feature_engineering_real.py) over the legacy
    # fully-imputed "engineered" CSV, so imputation happens train-only,
    # after the split, via split_impute_featurize() below. Falling back to
    # "*.csv" (any engineered file) only covers legacy/synthetic runs that
    # predate the cleaned-CSV split (those files have zero missing values in
    # the four imputed columns already, so re-running them through the same
    # split/impute/featurize path is a safe no-op, not a re-introduced leak).
    # ------------------------------------------------------------------
    csv_path = _find_latest_csv(input_dir, "real_estate_cleaned*.csv") or _find_latest_csv(
        input_dir, "*.csv"
    )
    if csv_path is not None:
        logger.info("Loading data from %s", csv_path)
        df = pd.read_csv(csv_path)
        logger.info("Loaded %d rows, %d columns.", *df.shape)
        is_synthetic = bool(df.get("is_synthetic", pd.Series([False])).any()) or (
            "synthetic" in csv_path.name
        )
        if not is_synthetic and "cleaned" not in csv_path.name:
            logger.warning(
                "'%s' is not a real_estate_cleaned*.csv file -- it may be the fully-imputed, "
                "EDA-only real_estate_engineered.csv (imputed on the WHOLE dataset before any "
                "split, see run_feature_engineering_real.py). split_impute_featurize() below "
                "will find no missing values left to impute train-only on such a file, which is "
                "NOT the same leakage-safety guarantee as training against the real cleaned CSV. "
                "Run scripts/run_feature_engineering_real.py first to regenerate it.",
                csv_path,
            )
        if is_synthetic and not args.allow_synthetic:
            print(
                f"\nERROR: '{csv_path}' is synthetic/demo data "
                "(is_synthetic=True or filename contains 'synthetic'), but "
                "--allow-synthetic was not passed.\n"
                "This script fails closed by design: training a model on "
                "synthetic data without explicit acknowledgement would let a "
                "synthetic-data artefact silently pass as a real one.\n\n"
                "Re-run with:  python scripts/run_model_training_real.py --allow-synthetic\n"
                "or:            make train-demo\n",
                file=sys.stderr,
            )
            raise SystemExit(2)
    elif args.allow_synthetic:
        df = _generate_synthetic_engineered_data(n=400)
        is_synthetic = True
        logger.warning("Feature engineering produced %d SYNTHETIC rows.", len(df))
        csv_path = _PROJECT_ROOT / "data" / "processed" / "real_estate_engineered.synthetic.csv"
    else:
        print(
            "\nERROR: no engineered CSV found in "
            f"'{input_dir}' and synthetic generation was not requested.\n\n"
            "This script fails closed by design (see DATA_CARD.md).\n"
            "To proceed, either:\n"
            "  1. Run feature engineering on real data first:\n"
            "       python scripts/run_feature_engineering_real.py\n"
            "  2. Explicitly request synthetic demo training data:\n"
            "       python scripts/run_model_training_real.py --allow-synthetic\n"
            "     or:\n"
            "       make train-demo\n",
            file=sys.stderr,
        )
        raise SystemExit(2)

    dataset_sha256 = _sha256_of_file(csv_path) if csv_path.exists() else None
    dataset_rows, dataset_columns = df.shape

    # ------------------------------------------------------------------
    # Step 2+3: Split FIRST, then fit imputation/feature engineering on
    # train only (see src/data/split_pipeline.py). Replaces an earlier
    # "featurize the whole dataset, then split" order, which fit rooms/
    # total_area/floor/floors_total medians on data spanning both splits.
    # ------------------------------------------------------------------
    if "price" not in df.columns:
        raise ValueError(
            "Target column 'price' not found. Ensure the input CSV contains price data."
        )

    logger.info(
        "Splitting (%s) then fitting imputer/features on train only...", args.split_strategy
    )
    split_result = split_impute_featurize(df, split_strategy=args.split_strategy)
    X_train, X_test = split_result.X_train, split_result.X_test
    y_train, y_test = split_result.y_train, split_result.y_test
    X = pd.concat([X_train, X_test])
    split_strategy = split_result.split_strategy
    logger.info(
        "Data split (%s): %d train / %d test.",
        args.split_strategy,
        len(X_train),
        len(X_test),
    )

    # ------------------------------------------------------------------
    # Step 3b: Optionally load tuned hyperparameters from a prior study
    # ------------------------------------------------------------------
    trainer_params: dict = {}
    if args.params_from_study and args.params_from_manifest:
        print(
            "ERROR: pass only one of --params-from-study / --params-from-manifest.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.params_from_manifest:
        manifest_path = models_dir / "current_model.json"
        if not manifest_path.is_file():
            print(
                f"ERROR: --params-from-manifest passed but {manifest_path} does not exist.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trainer_params = dict(manifest.get("hyperparameters") or {})
        if not trainer_params:
            print(
                f"ERROR: {manifest_path} has no 'hyperparameters' to reuse.", file=sys.stderr
            )
            raise SystemExit(2)
        logger.info("Reusing hyperparameters from %s: %s", manifest_path, trainer_params)
    if args.params_from_study:
        study_path = _PROJECT_ROOT / "reports" / "tuning_study_real.json"
        if not study_path.exists():
            print(
                f"ERROR: --params-from-study was passed but {study_path} does not exist. "
                "Run scripts/tune_models_real.py first.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        study = json.loads(study_path.read_text(encoding="utf-8"))
        if args.params_from_study not in study:
            print(
                f"ERROR: '{args.params_from_study}' not found in {study_path}. "
                f"Available: {sorted(study)}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        trainer_params = study[args.params_from_study]["best_params"]
        logger.info(
            "Loaded tuned params for '%s' from %s: %s",
            args.params_from_study,
            study_path,
            trainer_params,
        )

    # ------------------------------------------------------------------
    # Step 4: Fit model on X_train; evaluate on held-out X_test
    # ------------------------------------------------------------------
    logger.info("Training %s model...", args.model)
    trainer = ModelTrainer(
        model_type=args.model, config={"params": trainer_params} if trainer_params else None
    )
    eval_metrics = trainer.fit(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names=split_result.feature_names,
    )
    evaluator = ModelEvaluator()

    # ------------------------------------------------------------------
    # Step 5: Assemble provenance metadata (required for traceability) and save
    # ------------------------------------------------------------------
    training_end = datetime.now()
    resolved_data_source = _resolve_data_source(is_synthetic, df)

    try:
        dataset_path_repo_relative = str(csv_path.resolve().relative_to(_PROJECT_ROOT))
    except ValueError:
        # csv_path isn't under the project root (e.g. an explicit --input
        # pointing elsewhere) -- fall back to just the filename rather than
        # embedding a full absolute path (which would leak the local
        # machine's username on Windows, e.g. C:\Users\<name>\...).
        dataset_path_repo_relative = csv_path.name

    metadata = _build_lineage_metadata(
        df=df,
        resolved_data_source=resolved_data_source,
        is_synthetic=is_synthetic,
        dataset_path_repo_relative=dataset_path_repo_relative,
        dataset_sha256=dataset_sha256,
        dataset_rows=dataset_rows,
        dataset_columns=dataset_columns,
        training_start=training_start,
        training_end=training_end,
        split_strategy=split_strategy,
        trainer_params=trainer_params,
        params_from_study=(
            args.params_from_study
            or ("current_model.json (manifest reuse)" if args.params_from_manifest else None)
        ),
        feature_names=split_result.feature_names,
    )
    # Persist the train-fit imputer so inference/re-evaluation can reuse the
    # exact same fill values instead of recomputing them from whatever data
    # is at hand (see src/preprocessing/imputer.py).
    metadata["imputer"] = split_result.imputer.to_dict()
    set_as_current = is_synthetic or args.promote
    model_path = trainer.save(
        str(models_dir),
        metadata=metadata,
        tag="synthetic" if is_synthetic else None,
        set_as_current=set_as_current,
    )
    logger.info(
        "Model saved to %s (is_synthetic=%s, promoted_to_current=%s)",
        model_path,
        is_synthetic,
        set_as_current,
    )
    if not set_as_current:
        logger.warning(
            "Real-data model NOT promoted to current_model.json (pass --promote after "
            "reviewing its metrics to make API/dashboard serve it). "
            "models/current_model.json still points at the previously reviewed artefact."
        )

    # ------------------------------------------------------------------
    # Step 6: Print summary report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL TRAINING SUMMARY" + (" [SYNTHETIC/DEMO DATA]" if is_synthetic else ""))
    print("=" * 60)
    print(f"Model type      : {args.model}")
    print(f"Data source     : {metadata['data_source']}")
    print(
        f"Dataset         : {csv_path} (sha256={dataset_sha256[:12] if dataset_sha256 else 'N/A'}...)"
    )
    print(f"Total samples   : {len(X):,}")
    print(f"Train / Test    : {len(X_train):,} / {len(X_test):,}")
    print(f"Feature count   : {X.shape[1]}")
    print(f"Features used   : {list(X.columns)}")
    print(f"Model saved to  : {model_path}")
    print(
        f"Promoted to current_model.json : {set_as_current}"
        + ("" if set_as_current else "  (pass --promote after review to serve this model)")
    )
    print()
    evaluator.print_report(eval_metrics)
    interval = trainer._prediction_interval
    if interval:
        print(
            f"Estimated range ({interval['coverage_target']:.0%} target coverage): "
            f"[{interval['lower_offset']:+,.0f}, {interval['upper_offset']:+,.0f}] RUB offset "
            f"| observed coverage on holdout: {interval['observed_coverage']:.1%}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Honest model comparison on real (restate.ru) engineered data.
Usage: python scripts/compare_models_real.py [--input path] [--split-strategy group|random]

Loads the latest engineered CSV from data/processed/ and compares a
baseline-through-boosting model set:

  - dummy_median       -- sklearn DummyRegressor(strategy='median'), the
                           honest floor every other model must beat
  - linear_regression  -- unregularized linear baseline
  - ridge              -- L2-regularized linear baseline
  - elastic_net        -- L1+L2-regularized linear baseline
  - random_forest      -- bagged tree ensemble
  - extra_trees        -- bagged tree ensemble, extra-randomized splits
  - hist_gradient_boosting -- sklearn's native histogram-based boosting (no extra dependency)
  - xgboost            -- gradient boosting
  - catboost           -- gradient boosting with native categorical support
                           (interesting here specifically because this dataset
                           has real categorical features -- city, building_type)

--split-strategy (default: group) chooses the train/test split:
  - group  -- GroupShuffleSplit, grouping near-duplicate listings (see
              src.data.schema.build_split_groups) so no near-duplicate pair
              is ever split across train/test. Adopted as the PRIMARY
              methodology after a leakage audit quantified 37.5% of
              near-duplicate groups leaking across a plain random split.
  - random -- plain 80/20 train_test_split(random_state=42), kept only as a
              reference/continuity number against the original random-split
              comparison, never as the primary metric.

Both use test_size=0.2 and random_state=42 for reproducibility.

For the best model (lowest test MAE), additionally reports:
  - per-price-segment (quartile) MAE breakdown
  - per-city MAE breakdown (when >1 city present)
  - residual summary statistics

Saves reports/model_comparison_real.csv. Does NOT save any model artefact or
touch models/current_model.json -- this script is comparison/evaluation
only. Use `scripts/run_model_training_real.py --model <winner> [--promote]`
to actually train+save (and, after review, promote) a chosen model.

FAIL-CLOSED: refuses to run on synthetic-flagged data (comparing model
quality on synthetic data would not be an honest real-market comparison),
unless --allow-synthetic is passed (e.g. to smoke-test this script itself).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.schema import build_split_groups
from src.features.feature_engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.utils.logger import get_logger

logger = get_logger("compare_models_real")

# Models ModelTrainer already builds/scales/fits (production-supported types
# -- whichever wins here can be reproduced exactly via
# `run_model_training_real.py --model <name>`).
_TRAINER_BACKED_MODELS = ["linear_regression", "ridge", "random_forest", "xgboost", "catboost"]

# Extra benchmark-only models: sklearn estimators with no ModelTrainer
# support (not part of the production artefact contract, comparison only).
# Each factory takes no arguments and returns a fresh unfitted estimator.
_SKLEARN_ONLY_MODELS = {
    "dummy_median": lambda: DummyRegressor(strategy="median"),
    # ElasticNetCV picks alpha via internal 5-fold CV on the training data
    # only (never touches the held-out test set) -- a fixed alpha=1.0
    # (sklearn's ElasticNet default) was tried first and zeroed all 27
    # coefficients on this dataset (the log1p(price) target's scale is
    # small, std~0.7, so alpha=1.0 is wildly too strong), collapsing to an
    # intercept-only prediction indistinguishable from the dummy baseline --
    # that was a benchmark-code bug, not a real finding about ElasticNet,
    # so a data-driven alpha is used instead of guessing a fixed one.
    "elastic_net": lambda: ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=np.logspace(-4, 1, 30), cv=5, random_state=42
    ),
    "extra_trees": lambda: ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(random_state=42),
}


def _find_latest_csv(directory: Path, pattern: str = "*.csv") -> Path | None:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mask = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        if mask.any()
        else float("nan")
    )
    median_ae = float(median_absolute_error(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "median_ae": median_ae}


def _split(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, strategy: str):
    if strategy == "random":
        return train_test_split(X, y, test_size=0.2, random_state=42)

    groups = build_split_groups(df).loc[X.index]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model types on real engineered data.")
    parser.add_argument("--input", default=str(_PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument(
        "--split-strategy",
        choices=["group", "random"],
        default="group",
        help="'group' (default) keeps near-duplicate listings entirely in train or test; "
        "'random' is the plain "
        "80/20 split kept only for continuity with earlier comparisons.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    csv_path = _find_latest_csv(input_dir, "*.csv")
    if csv_path is None:
        print(f"ERROR: no engineered CSV found in '{input_dir}'.", file=sys.stderr)
        raise SystemExit(2)

    df = pd.read_csv(csv_path)
    is_synthetic = (
        bool(df.get("is_synthetic", pd.Series([False])).any()) or "synthetic" in csv_path.name
    )
    if is_synthetic and not args.allow_synthetic:
        print(
            f"ERROR: '{csv_path}' is synthetic data. Model comparison on synthetic data is not "
            "an honest real-market comparison. Pass --allow-synthetic to override (e.g. to "
            "smoke-test this script).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    logger.info("Loaded %d rows from %s", len(df), csv_path)

    feat_eng = FeatureEngineer()
    X, y = feat_eng.prepare_for_training(df)
    X_train, X_test, y_train, y_test = _split(X, y, df, args.split_strategy)
    logger.info(
        "Split (%s): %d train / %d test, %d features.",
        args.split_strategy,
        len(X_train),
        len(X_test),
        X.shape[1],
    )

    results = []
    predictors: dict = {}  # model_name -> callable(X_scaled_or_raw) -> y_pred_rub, for post-hoc diagnostics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_true_rub = np.expm1(y_test.values)

    for model_type in _TRAINER_BACKED_MODELS:
        try:
            trainer = ModelTrainer(model_type=model_type)
            metrics = trainer.fit(X_train, y_train, X_test, y_test, feature_names=list(X.columns))
            y_pred_rub = np.expm1(trainer.model.predict(trainer.scaler.transform(X_test)))
            metrics = {**metrics, "median_ae": float(median_absolute_error(y_true_rub, y_pred_rub))}
            results.append({"model": model_type, **metrics})
            predictors[model_type] = y_pred_rub
        except Exception as exc:  # pragma: no cover - defensive, logged not swallowed silently
            logger.error("Model '%s' failed to train: %s", model_type, exc, exc_info=True)
            results.append(
                {
                    "model": model_type,
                    "mae": None,
                    "rmse": None,
                    "r2": None,
                    "mape": None,
                    "error": str(exc),
                }
            )

    for model_name, factory in _SKLEARN_ONLY_MODELS.items():
        try:
            model = factory()
            model.fit(X_train_scaled, y_train)
            y_pred_rub = np.expm1(model.predict(X_test_scaled))
            metrics = _metrics_dict(y_true_rub, y_pred_rub)
            results.append({"model": model_name, **metrics})
            predictors[model_name] = y_pred_rub
        except Exception as exc:  # pragma: no cover - defensive, logged not swallowed silently
            logger.error("Model '%s' failed to train: %s", model_name, exc, exc_info=True)
            results.append(
                {
                    "model": model_name,
                    "mae": None,
                    "rmse": None,
                    "r2": None,
                    "mape": None,
                    "error": str(exc),
                }
            )

    results_df = pd.DataFrame(results).sort_values("mae", na_position="last").reset_index(drop=True)

    reports_dir = _PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "model_comparison_real.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("REAL-DATA MODEL COMPARISON" + (" [SYNTHETIC — smoke test]" if is_synthetic else ""))
    print("=" * 70)
    print(f"Dataset: {csv_path} ({len(df)} rows)")
    print(
        f"Split  : {len(X_train)} train / {len(X_test)} test ({args.split_strategy}, test_size=0.2, random_state=42)"
    )
    print()
    print(results_df.to_string(index=False))
    print(f"\nSaved: {out_path}")

    # ------------------------------------------------------------------
    # Deeper diagnostics for the best real (non-baseline) model
    # ------------------------------------------------------------------
    real_results = results_df[results_df["model"] != "dummy_median"].dropna(subset=["mae"])
    if real_results.empty:
        print("\nNo real model trained successfully -- skipping per-segment diagnostics.")
        return
    best_name = real_results.iloc[0]["model"]
    y_pred = predictors[best_name]
    y_true = y_true_rub
    print(f"\nBest model: {best_name} (test MAE={real_results.iloc[0]['mae']:,.0f} RUB)")

    residuals = y_true - y_pred

    print("\nResidual summary (RUB, true - predicted):")
    print(
        f"  mean={residuals.mean():,.0f}  std={residuals.std():,.0f}  "
        f"min={residuals.min():,.0f}  max={residuals.max():,.0f}"
    )

    print("\nPer-price-segment MAE (test set, quartiles of true price):")
    seg_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    try:
        seg_df["segment"] = pd.qcut(seg_df["y_true"], 4, duplicates="drop")
        for seg, group in seg_df.groupby("segment", observed=True):
            seg_mae = float(np.mean(np.abs(group["y_true"] - group["y_pred"])))
            print(f"  {str(seg):35s} n={len(group):4d}  MAE={seg_mae:,.0f} RUB")
    except ValueError as exc:
        print(f"  (could not compute price-segment breakdown: {exc})")

    city_cols = [c for c in X_test.columns if c.startswith("city_")]
    if len(city_cols) > 1:
        print("\nPer-city MAE (test set, only cities present in test split):")
        for col in city_cols:
            mask = X_test[col].values == 1
            if mask.sum() == 0:
                continue
            city_mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
            print(f"  {col:25s} n={int(mask.sum()):4d}  MAE={city_mae:,.0f} RUB")

    print("=" * 70)


if __name__ == "__main__":
    main()

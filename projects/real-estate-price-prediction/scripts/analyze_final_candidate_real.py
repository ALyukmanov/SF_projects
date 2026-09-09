"""
Deep-dive analysis of the final real-data candidate model: full metrics,
segment breakdown, residual/error analysis, and feature importance.

Usage: python scripts/analyze_final_candidate_real.py [--input path]

Retrains the winning candidate from scripts/tune_models_real.py's study
(reports/tuning_study_real.json — currently `xgboost`, tuned) on the same
group-aware split, then produces:

  - Full metrics: MAE, RMSE, R2, median AE, MAPE, sMAPE, RMSLE
  - Segment metrics: by city, by property_type/source_category, by price
    quartile, by room count, by area tier
  - Residual analysis: distribution, systematic bias check (mean residual
    by segment), largest absolute errors with enough context to inspect
    WHY the model likely missed
  - Feature importance: native (impurity/gain-based) + permutation
    importance on the held-out test set, plus a handful of individual
    prediction explanations

Saves reports/final_candidate_analysis_real.json (machine-readable) and
prints a human-readable report. Does NOT save a model artefact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import median_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data.split_pipeline import split_impute_featurize
from src.utils.logger import get_logger

logger = get_logger("analyze_final_candidate_real")

_RANDOM_SEED = 42


def _find_latest_csv(directory: Path, pattern: str = "*.csv") -> Path | None:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_safe = np.clip(y_pred, a_min=0, a_max=None)  # guard against a stray negative prediction
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred_safe)) ** 2)))


def _full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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
    abs_err = np.abs(y_true - y_pred)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "p50_abs_error_rub": float(np.percentile(abs_err, 50)),
        "p75_abs_error_rub": float(np.percentile(abs_err, 75)),
        "p90_abs_error_rub": float(np.percentile(abs_err, 90)),
        "mape": mape,
        "smape": _smape(y_true, y_pred),
        "rmsle": _rmsle(y_true, y_pred),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Deep-dive analysis of the final real-data candidate."
    )
    parser.add_argument("--input", default=str(_PROJECT_ROOT / "data" / "processed"))
    args = parser.parse_args()

    csv_path = _find_latest_csv(Path(args.input), "real_estate_cleaned*.csv") or _find_latest_csv(
        Path(args.input), "*.csv"
    )
    if csv_path is None:
        print(f"ERROR: no cleaned/engineered CSV found in '{args.input}'.", file=sys.stderr)
        raise SystemExit(2)
    if "cleaned" not in csv_path.name:
        print(
            f"WARNING: '{csv_path}' is not a real_estate_cleaned*.csv file -- it may be the "
            "fully-imputed, EDA-only real_estate_engineered.csv (imputed on the WHOLE dataset "
            "before any split, see run_feature_engineering_real.py). split_impute_featurize() "
            "below will find no missing values left to impute train-only on such a file, which "
            "is NOT the same leakage-safety guarantee as analyzing against the real cleaned CSV. "
            "Run scripts/run_feature_engineering_real.py first to regenerate it.",
            file=sys.stderr,
        )

    from scripts._tuned_params import load_tuned_xgb_params

    best_params, params_source = load_tuned_xgb_params()
    logger.info("Using xgboost params from %s: %s", params_source, best_params)

    df = pd.read_csv(csv_path)
    logger.info("Loaded %s: %d rows.", csv_path, len(df))
    # Split -> impute (train-only) -> featurize — see src/data/split_pipeline.py.
    # 'location' (building-level groups) is the honest evaluation for the geo
    # model: two flats in one building never straddle the holdout.
    split_result = split_impute_featurize(df, split_strategy="location", random_state=_RANDOM_SEED)
    X_train, X_test = split_result.X_train, split_result.X_test
    y_train, y_test = split_result.y_train, split_result.y_test
    X = pd.concat([X_train, X_test])
    # df_test combines the imputed numeric feature columns (what the model
    # actually saw) with the raw categorical/identifying columns (city,
    # source_category, address, url) needed for segment slicing below.
    test_imputed = split_result.imputer.transform(split_result.test_raw)
    assert len(test_imputed) == len(X_test), "test split row count mismatch"
    df_test = test_imputed.reset_index(drop=True).copy()
    for col in X_test.columns:
        df_test[col] = X_test[col].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(random_state=_RANDOM_SEED, n_jobs=-1, **best_params)
    model.fit(X_train_scaled, y_train)

    y_pred = np.expm1(model.predict(X_test_scaled))
    y_true = np.expm1(y_test.values)
    residuals = y_true - y_pred

    overall = _full_metrics(y_true, y_pred)

    # ------------------------------------------------------------------
    # Segment metrics
    # ------------------------------------------------------------------
    segments: dict = {}

    for city_col in [c for c in X_test.columns if c.startswith("city_")]:
        mask = X_test[city_col].values == 1
        if mask.sum() == 0:
            continue
        segments[city_col] = {"n": int(mask.sum()), **_full_metrics(y_true[mask], y_pred[mask])}

    for cat, group in df_test.groupby("source_category"):
        mask = df_test.index.isin(group.index)
        if mask.sum() < 5:
            continue
        segments[f"category_{cat}"] = {
            "n": int(mask.sum()),
            **_full_metrics(y_true[mask], y_pred[mask]),
        }

    price_q = pd.qcut(y_true, 4, duplicates="drop")
    for seg in price_q.categories:
        mask = np.asarray(price_q == seg)
        segments[f"price_{seg}"] = {
            "n": int(mask.sum()),
            **_full_metrics(y_true[mask], y_pred[mask]),
        }

    for rooms_val, group in df_test.groupby(df_test["rooms"].fillna(-1)):
        mask = df_test.index.isin(group.index)
        if mask.sum() < 10:
            continue
        label = "unknown" if rooms_val == -1 else f"{int(rooms_val)}"
        segments[f"rooms_{label}"] = {
            "n": int(mask.sum()),
            **_full_metrics(y_true[mask], y_pred[mask]),
        }

    area_q = pd.qcut(df_test["total_area"], 4, duplicates="drop")
    for seg in area_q.cat.categories:
        mask = np.asarray(area_q == seg)
        if mask.sum() < 10:
            continue
        segments[f"area_{seg}"] = {
            "n": int(mask.sum()),
            **_full_metrics(y_true[mask], y_pred[mask]),
        }

    # ------------------------------------------------------------------
    # Residual / error analysis
    # ------------------------------------------------------------------
    abs_residuals = np.abs(residuals)
    worst_idx = np.argsort(-abs_residuals)[:10]
    worst_examples = []
    for i in worst_idx:
        row = df_test.iloc[i]
        worst_examples.append(
            {
                "url": row.get("url"),
                "address": row.get("address"),
                "city": row.get("city"),
                "source_category": row.get("source_category"),
                "true_price": float(y_true[i]),
                "predicted_price": float(y_pred[i]),
                "error": float(residuals[i]),
                "total_area": row.get("total_area"),
            }
        )

    bias = {
        "mean_residual": float(residuals.mean()),
        "median_residual": float(np.median(residuals)),
        "pct_overpredicted": float((residuals < 0).mean() * 100),  # pred > true
        "pct_underpredicted": float((residuals > 0).mean() * 100),  # pred < true
    }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    native_importance = dict(zip(X.columns, model.feature_importances_.tolist()))
    native_importance_sorted = dict(sorted(native_importance.items(), key=lambda kv: -kv[1])[:15])

    logger.info(
        "Computing permutation importance on the held-out test set (this can take a moment)..."
    )
    perm = permutation_importance(
        model, X_test_scaled, y_test, n_repeats=5, random_state=_RANDOM_SEED, n_jobs=-1
    )
    perm_importance_sorted = dict(
        sorted(zip(X.columns, perm.importances_mean.tolist()), key=lambda kv: -kv[1])[:15]
    )

    result = {
        "overall_metrics": overall,
        "segments": segments,
        "residual_bias": bias,
        "worst_10_examples": worst_examples,
        "native_feature_importance_top15": native_importance_sorted,
        "permutation_importance_top15_test": perm_importance_sorted,
    }

    out_path = _PROJECT_ROOT / "reports" / "final_candidate_analysis_real.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("FINAL CANDIDATE (tuned xgboost) — DEEP-DIVE ANALYSIS")
    print("=" * 70)
    print(f"Test set: {len(y_true)} rows")
    print("\nOverall metrics:")
    for k, v in overall.items():
        print(f"  {k:12s}: {v:,.4f}" if k in ("r2",) else f"  {k:12s}: {v:,.2f}")

    print("\nResidual bias:")
    for k, v in bias.items():
        print(f"  {k}: {v:.2f}")

    print("\nTop feature importances (native, top 8):")
    for k, v in list(native_importance_sorted.items())[:8]:
        print(f"  {k:25s}: {v:.4f}")

    print("\nTop feature importances (permutation, held-out test, top 8):")
    for k, v in list(perm_importance_sorted.items())[:8]:
        print(f"  {k:25s}: {v:.4f}")

    print(f"\nSaved full analysis to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

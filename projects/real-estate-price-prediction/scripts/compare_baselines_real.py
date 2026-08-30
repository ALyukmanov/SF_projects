"""Compares the tuned real-data xgboost candidate against simple,
non-ML baselines on the IDENTICAL group-aware split -- to demonstrate ML is
actually adding value, not just that R^2 is positive. All baseline
statistics (medians) are computed from TRAIN rows only and applied to TEST
rows, so this is a fair held-out comparison, not a fitted-on-everything one.

Usage: python scripts/compare_baselines_real.py
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data.split_pipeline import split_impute_featurize

_RANDOM_SEED = 42


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
    }


def _find_cleaned_csv() -> Path:
    processed = _PROJECT_ROOT / "data" / "processed"
    cleaned = sorted(processed.glob("real_estate_cleaned.csv"))
    if cleaned:
        return cleaned[0]
    print(
        "WARNING: real_estate_cleaned.csv not found -- falling back to the "
        "fully-imputed, EDA-only real_estate_engineered.csv (imputed on the "
        "WHOLE dataset before any split; see run_feature_engineering_real.py). "
        "This baseline comparison's split_impute_featurize() call will find no "
        "missing values left to impute train-only, which is not the same "
        "guarantee as running against the real cleaned CSV. Run "
        "scripts/run_feature_engineering_real.py first to regenerate it.",
        file=sys.stderr,
    )
    return processed / "real_estate_engineered.csv"


def main() -> None:
    csv_path = _find_cleaned_csv()
    df = pd.read_csv(csv_path)

    # Split -> impute (train-only) -> featurize, same leakage-safe pipeline
    # as the actual candidate (src/data/split_pipeline.py) -- so this
    # comparison reflects the SAME holdout the promoted candidate is judged
    # on, not a separately-imputed one.
    split_result = split_impute_featurize(df, split_strategy="group", random_state=_RANDOM_SEED)
    X_train, X_test = split_result.X_train, split_result.X_test
    y_log_train = split_result.y_train

    df_train_imputed = split_result.imputer.transform(split_result.train_raw)
    df_test_imputed = split_result.imputer.transform(split_result.test_raw)
    y_train = df_train_imputed["price"].to_numpy()
    y_test = df_test_imputed["price"].to_numpy()
    df_train, df_test = df_train_imputed, df_test_imputed

    results: dict = {"n_train": int(len(X_train)), "n_test": int(len(X_test))}

    # ------------------------------------------------------------------
    # Baseline 1: global median price
    # ------------------------------------------------------------------
    global_median = float(np.median(y_train))
    pred = np.full(len(y_test), global_median)
    results["baseline_global_median"] = _metrics(y_test, pred)

    # ------------------------------------------------------------------
    # Baseline 2: median price by city
    # ------------------------------------------------------------------
    city_median = df_train.assign(price=y_train).groupby("city")["price"].median()
    pred = df_test["city"].map(city_median).fillna(global_median).to_numpy()
    results["baseline_median_by_city"] = _metrics(y_test, pred)

    # ------------------------------------------------------------------
    # Baseline 3: median price-per-sqm (global) x total_area
    # ------------------------------------------------------------------
    ppsqm_train = y_train / df_train["total_area"].to_numpy()
    global_ppsqm_median = float(np.median(ppsqm_train[np.isfinite(ppsqm_train)]))
    pred = global_ppsqm_median * df_test["total_area"].to_numpy()
    results["baseline_price_per_sqm_global"] = _metrics(y_test, pred)

    # ------------------------------------------------------------------
    # Baseline 4: median price-per-sqm by city+category x total_area
    # (falls back to city-level, then global, for unseen/small groups)
    # ------------------------------------------------------------------
    df_train_ppsqm = df_train.assign(ppsqm=ppsqm_train)
    group_median = df_train_ppsqm.groupby(["city", "source_category"])["ppsqm"].median()
    city_ppsqm_median = df_train_ppsqm.groupby("city")["ppsqm"].median()

    def _lookup_ppsqm(row) -> float:
        key = (row["city"], row["source_category"])
        if key in group_median.index and not pd.isna(group_median.loc[key]):
            return float(group_median.loc[key])
        if row["city"] in city_ppsqm_median.index:
            return float(city_ppsqm_median.loc[row["city"]])
        return global_ppsqm_median

    ppsqm_pred = df_test.apply(_lookup_ppsqm, axis=1).to_numpy()
    pred = ppsqm_pred * df_test["total_area"].to_numpy()
    results["baseline_price_per_sqm_by_city_category"] = _metrics(y_test, pred)

    # ------------------------------------------------------------------
    # ML candidate: tuned xgboost, log1p target, same split (reference)
    # ------------------------------------------------------------------
    study_path = _PROJECT_ROOT / "reports" / "tuning_study_real.json"
    xgb_params = json.loads(study_path.read_text(encoding="utf-8"))["xgboost"]["best_params"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = XGBRegressor(**xgb_params, random_state=_RANDOM_SEED, n_jobs=-1)
    model.fit(X_train_scaled, y_log_train)
    pred_ml = np.expm1(model.predict(X_test_scaled))
    results["ml_tuned_xgboost"] = _metrics(y_test, pred_ml)

    out_path = _PROJECT_ROOT / "reports" / "baseline_comparison_real.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Test rows: {len(y_test)}\n")
    for key, val in results.items():
        if isinstance(val, dict):
            print(f"{key}:")
            for k, v in val.items():
                print(f"  {k:12s}: {v:,.4f}" if k == "r2" else f"  {k:12s}: {v:,.2f}")
            print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

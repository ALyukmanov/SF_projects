"""Compact, honest comparison of modeling raw price vs log1p(price) as the
regression target -- same group-aware split, same seed, no hyperparameter
search (XGBoost reuses the already-tuned params from
reports/tuning_study_real.json; CatBoost reuses ModelTrainer's existing
defaults). Not a search for the prettiest number: reports full metrics
AND expensive-object errors specifically, since the raw-price heavy right
tail is exactly where a log target is expected to help or hurt.

Usage: python scripts/compare_log_vs_raw_target.py
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data.schema import build_split_groups
from src.features.feature_engineering import FeatureEngineer

_RANDOM_SEED = 42


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
    }


def main() -> None:
    csv_path = _PROJECT_ROOT / "data" / "processed" / "real_estate_engineered.csv"
    df = pd.read_csv(csv_path)

    engineer = FeatureEngineer()
    X, y_log = engineer.prepare_for_training(
        df
    )  # y_log = log1p(price), rows with price NaN dropped
    y_raw = df.loc[X.index, "price"].to_numpy()

    groups = build_split_groups(df).loc[X.index]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=_RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y_log, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_log_train = y_log.iloc[train_idx]
    y_raw_train, y_raw_test = y_raw[train_idx], y_raw[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    study_path = _PROJECT_ROOT / "reports" / "tuning_study_real.json"
    xgb_params = json.loads(study_path.read_text(encoding="utf-8"))["xgboost"]["best_params"]

    # Expensive-object mask, defined on the RAW test-set price so both
    # models are judged against the identical subset of rows.
    expensive_threshold = float(np.percentile(y_raw_test, 90))
    expensive_mask = y_raw_test >= expensive_threshold

    results: dict = {
        "expensive_threshold_rub": expensive_threshold,
        "n_expensive_rows": int(expensive_mask.sum()),
    }

    # ------------------------------------------------------------------
    # XGBoost: log1p(price) target (back-transformed at evaluation time)
    # ------------------------------------------------------------------
    model_log = XGBRegressor(**xgb_params, random_state=_RANDOM_SEED, n_jobs=-1)
    model_log.fit(X_train_scaled, y_log_train)
    pred_log = np.expm1(model_log.predict(X_test_scaled))
    results["xgboost_log1p_target"] = {
        **_metrics(y_raw_test, pred_log),
        "expensive_mae": float(
            mean_absolute_error(y_raw_test[expensive_mask], pred_log[expensive_mask])
        ),
    }

    # ------------------------------------------------------------------
    # XGBoost: raw price target (same hyperparameters, no back-transform)
    # ------------------------------------------------------------------
    model_raw = XGBRegressor(**xgb_params, random_state=_RANDOM_SEED, n_jobs=-1)
    model_raw.fit(X_train_scaled, y_raw_train)
    pred_raw = model_raw.predict(X_test_scaled)
    results["xgboost_raw_price_target"] = {
        **_metrics(y_raw_test, pred_raw),
        "expensive_mae": float(
            mean_absolute_error(y_raw_test[expensive_mask], pred_raw[expensive_mask])
        ),
    }

    # ------------------------------------------------------------------
    # CatBoost: same comparison, ModelTrainer's existing (untuned) defaults
    # ------------------------------------------------------------------
    try:
        from catboost import CatBoostRegressor

        cb_defaults = dict(
            iterations=500, depth=6, learning_rate=0.05, random_seed=_RANDOM_SEED, verbose=0
        )

        cb_log = CatBoostRegressor(**cb_defaults)
        cb_log.fit(X_train_scaled, y_log_train)
        pred_cb_log = np.expm1(cb_log.predict(X_test_scaled))
        results["catboost_log1p_target"] = {
            **_metrics(y_raw_test, pred_cb_log),
            "expensive_mae": float(
                mean_absolute_error(y_raw_test[expensive_mask], pred_cb_log[expensive_mask])
            ),
        }

        cb_raw = CatBoostRegressor(**cb_defaults)
        cb_raw.fit(X_train_scaled, y_raw_train)
        pred_cb_raw = cb_raw.predict(X_test_scaled)
        results["catboost_raw_price_target"] = {
            **_metrics(y_raw_test, pred_cb_raw),
            "expensive_mae": float(
                mean_absolute_error(y_raw_test[expensive_mask], pred_cb_raw[expensive_mask])
            ),
        }
    except ImportError:
        results["catboost"] = "not installed -- skipped"

    out_path = _PROJECT_ROOT / "reports" / "log_vs_raw_target_real.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Test rows: {len(y_raw_test)} | expensive (top decile, >= {expensive_threshold:,.0f} RUB): {expensive_mask.sum()}"
    )
    for key, val in results.items():
        if isinstance(val, dict):
            print(f"\n{key}:")
            for k, v in val.items():
                print(f"  {k:15s}: {v:,.4f}" if k == "r2" else f"  {k:15s}: {v:,.2f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

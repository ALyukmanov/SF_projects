"""
Compare the model with and without geo features on the location-grouped holdout.

Usage: python scripts/compare_geo_uplift.py

Loads ``data/processed/real_estate_cleaned.csv`` and, on the leakage-safe
location-grouped split (src.data.schema.build_location_groups), trains and
evaluates three variants on the same holdout:

  pre_geo            27 base features, current model's xgboost params
  geo_current_params + 18 OSM geo features, same params
  geo_tuned          + 18 OSM geo features, params from
                     reports/tuning_study_real.json (tuned by DEV-only
                     GroupKFold CV — the holdout is never seen during tuning)

For each: MAE / RMSE / R² / MAPE / median AE on the holdout.

For the best geo variant: per-city, by-coordinate-availability, and
price-tercile breakdowns, plus permutation importance and a leakage scan.

The near-duplicate-only split ("group") is also run as a secondary
reference, but it does not decide the result (it is optimistic once
coordinates exist).

Writes reports/geo_uplift.json. Saves no model artefact.
"""

from __future__ import annotations

import argparse
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

from src.data.split_pipeline import split_impute_featurize
from src.features.geo_features import GEO_FEATURE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger("compare_geo_uplift")

_RANDOM_SEED = 42

# Current model's xgboost params (models/current_model.json) — the
# apples-to-apples reference so pre-geo vs geo differ only by the feature set.
_CURRENT_XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 8,
    "learning_rate": 0.042625313784050885,
    "subsample": 0.9511057476545823,
    "colsample_bytree": 0.9989665523092093,
    "min_child_weight": 4,
    "reg_lambda": 4.042102593026443,
}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "median_ae": float(median_absolute_error(y_true, y_pred)),
    }


def _xgb(params: dict):
    from xgboost import XGBRegressor

    return XGBRegressor(random_state=_RANDOM_SEED, n_jobs=-1, tree_method="hist", **params)


def _fit_predict(model, X_train, y_train, X_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model.fit(Xtr, y_train)
    return model, scaler, np.expm1(model.predict(Xte))


def _load_tuned_params() -> dict | None:
    p = _PROJECT_ROOT / "reports" / "tuning_study_real.json"
    if not p.is_file():
        return None
    study = json.loads(p.read_text(encoding="utf-8"))
    return study.get("xgboost", {}).get("best_params")


def _evaluate_split(df: pd.DataFrame, split_strategy: str, tuned_params: dict | None) -> dict:
    split = split_impute_featurize(df, split_strategy=split_strategy, random_state=_RANDOM_SEED)
    X_train, X_test = split.X_train, split.X_test
    y_train, y_test = split.y_train, split.y_test
    y_true = np.expm1(y_test.to_numpy())
    feat_names = list(X_train.columns)
    geo_cols = [c for c in GEO_FEATURE_COLUMNS if c in feat_names]
    base_cols = [c for c in feat_names if c not in geo_cols]

    candidates: dict[str, dict] = {}

    # A: pre-geo baseline
    m, s, yp = _fit_predict(
        _xgb(_CURRENT_XGB_PARAMS), X_train[base_cols], y_train, X_test[base_cols]
    )
    candidates["pre_geo"] = {
        "feature_set": f"pre-geo ({len(base_cols)} features)",
        "params": "current",
        "metrics": _metrics(y_true, yp),
        "_pred": yp,
        "_model": m,
        "_scaler": s,
        "_cols": base_cols,
    }

    # B: geo, current params
    m, s, yp = _fit_predict(_xgb(_CURRENT_XGB_PARAMS), X_train, y_train, X_test)
    candidates["geo_current_params"] = {
        "feature_set": f"+geo ({len(feat_names)} features)",
        "params": "current",
        "metrics": _metrics(y_true, yp),
        "_pred": yp,
        "_model": m,
        "_scaler": s,
        "_cols": feat_names,
    }

    # C: geo, tuned params
    if tuned_params:
        m, s, yp = _fit_predict(_xgb(tuned_params), X_train, y_train, X_test)
        candidates["geo_tuned"] = {
            "feature_set": f"+geo ({len(feat_names)} features)",
            "params": "tuned",
            "metrics": _metrics(y_true, yp),
            "_pred": yp,
            "_model": m,
            "_scaler": s,
            "_cols": feat_names,
        }

    return {
        "split_strategy": split.split_strategy,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "candidates": candidates,
        "_X_test": X_test,
        "_y_true": y_true,
        "_test_raw": split.test_raw,
        "_feat_names": feat_names,
        "_geo_cols": geo_cols,
    }


def _winner_deep_dive(ev: dict) -> dict:
    """Stability + importance for the best geo candidate on this split."""
    cands = ev["candidates"]
    geo_names = [n for n in ("geo_tuned", "geo_current_params") if n in cands]
    winner_name = min(geo_names, key=lambda n: cands[n]["metrics"]["mae"])
    winner = cands[winner_name]
    pre = cands["pre_geo"]
    X_test, y_true = ev["_X_test"], ev["_y_true"]
    yp_win, yp_pre = winner["_pred"], pre["_pred"]

    out: dict = {"winner": winner_name}

    # per city
    per_city = {}
    for col in [c for c in X_test.columns if c.startswith("city_")]:
        m = X_test[col].to_numpy() == 1
        if m.sum() < 20:
            continue
        per_city[col] = {
            "n": int(m.sum()),
            "pre_geo_mae": _metrics(y_true[m], yp_pre[m])["mae"],
            "winner_mae": _metrics(y_true[m], yp_win[m])["mae"],
            "pre_geo_mape": _metrics(y_true[m], yp_pre[m])["mape"],
            "winner_mape": _metrics(y_true[m], yp_win[m])["mape"],
        }
    out["per_city"] = per_city

    # by coordinate availability
    by_coord = {}
    if "has_coordinates" in X_test.columns:
        hc = X_test["has_coordinates"].to_numpy() == 1
        for label, m in [("has_coordinates_1", hc), ("has_coordinates_0", ~hc)]:
            if m.sum() == 0:
                continue
            mp, mw = _metrics(y_true[m], yp_pre[m]), _metrics(y_true[m], yp_win[m])
            by_coord[label] = {
                "n": int(m.sum()),
                "pre_geo_mae": mp["mae"],
                "winner_mae": mw["mae"],
                "pre_geo_median_ae": mp["median_ae"],
                "winner_median_ae": mw["median_ae"],
                "pre_geo_mape": mp["mape"],
                "winner_mape": mw["mape"],
            }
    out["by_coordinate_availability"] = by_coord

    # price terciles
    price_tier = {}
    try:
        tiers = pd.qcut(y_true, 3, labels=["low", "mid", "high"])
        for t in ["low", "mid", "high"]:
            m = np.asarray(tiers == t)
            price_tier[t] = {
                "n": int(m.sum()),
                "pre_geo_mae": _metrics(y_true[m], yp_pre[m])["mae"],
                "winner_mae": _metrics(y_true[m], yp_win[m])["mae"],
                "winner_mape": _metrics(y_true[m], yp_win[m])["mape"],
            }
    except ValueError:
        pass
    out["price_tercile"] = price_tier

    # permutation importance for the winning geo model on its own columns
    names = winner["_cols"]
    Xw = X_test[names]
    perm = permutation_importance(
        winner["_model"],
        winner["_scaler"].transform(Xw),
        np.log1p(y_true),
        n_repeats=10,
        random_state=_RANDOM_SEED,
        n_jobs=-1,
    )
    order = np.argsort(-perm.importances_mean)
    out["permutation_importance_top15"] = {
        names[i]: float(perm.importances_mean[i]) for i in order[:15]
    }
    native = winner["_model"].feature_importances_
    out["native_importance_top15"] = dict(
        sorted(zip(names, [float(x) for x in native]), key=lambda kv: -kv[1])[:15]
    )
    # leakage scan — the model input matrix must have no identifier or
    # target-derived column
    suspicious = [
        n
        for n in names
        if any(t in n.lower() for t in ("url", "listing_id", "source_id", "price_per", "log_price"))
    ]
    out["leakage_scan"] = {
        "feature_matrix_columns": names,
        "suspicious_feature_names_in_matrix": suspicious,
        "note": "feature matrix should contain no identifiers or target-derived columns",
    }
    return out


def _print_split(title: str, ev: dict, primary: bool) -> None:
    tag = "  <-- PRIMARY" if primary else "  (secondary reference)"
    print("\n" + "=" * 78)
    print(f"{title}: {ev['split_strategy']}{tag}")
    print(f"  {ev['n_train']} train / {ev['n_test']} holdout")
    print("=" * 78)
    print(f"{'candidate':<22} {'feature_set':<22} {'MAE':>13} {'RMSE':>13} {'MAPE':>7} {'R2':>7}")
    print("-" * 78)
    for name, c in ev["candidates"].items():
        m = c["metrics"]
        print(
            f"{name:<22} {c['feature_set']:<22} {m['mae']:>13,.0f} {m['rmse']:>13,.0f} "
            f"{m['mape']:>6.1f}% {m['r2']:>7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(_PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.input) / "real_estate_cleaned.csv"
    if not csv_path.is_file():
        print(f"ERROR: {csv_path} not found — run scripts/run_feature_engineering_real.py first.")
        raise SystemExit(2)
    df = pd.read_csv(csv_path)
    if bool(df.get("is_synthetic", pd.Series([False])).any()) and not args.allow_synthetic:
        print("ERROR: cleaned CSV is synthetic. Pass --allow-synthetic to override.")
        raise SystemExit(2)
    if not any(c in df.columns for c in GEO_FEATURE_COLUMNS):
        print("ERROR: no geo columns in cleaned CSV — run scripts/run_feature_engineering_real.py.")
        raise SystemExit(2)

    tuned = _load_tuned_params()
    if tuned is None:
        logger.warning(
            "reports/tuning_study_real.json not found — 'geo_tuned' candidate skipped. "
            "Run scripts/tune_models_real.py first."
        )

    logger.info("Location split (PRIMARY)...")
    loc = _evaluate_split(df, "location", tuned)
    logger.info("Near-duplicate split (secondary reference)...")
    grp = _evaluate_split(df, "group", tuned)

    deep = _winner_deep_dive(loc)

    def _clean(ev: dict) -> dict:
        return {
            "split_strategy": ev["split_strategy"],
            "n_train": ev["n_train"],
            "n_test": ev["n_test"],
            "candidates": {
                k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in ev["candidates"].items()
            },
        }

    payload = {
        "dataset": csv_path.name,
        "n_rows": int(len(df)),
        "random_seed": _RANDOM_SEED,
        "tuned_params": tuned,
        "primary_evaluation": _clean(loc),
        "secondary_reference_near_dup": _clean(grp),
        "winner_deep_dive_location": deep,
    }
    out_path = _PROJECT_ROOT / "reports" / "geo_uplift.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "#" * 78)
    print(f"# FINAL GEO CANDIDATES — {csv_path.name} ({len(df)} rows), seed {_RANDOM_SEED}")
    print("#" * 78)
    _print_split("PRIMARY", loc, primary=True)
    _print_split("SECONDARY", grp, primary=False)

    print(f"\n--- Winner on the location holdout: {deep['winner']} ---")
    print("\nPer-city MAE (pre_geo -> winner):")
    for c, v in deep["per_city"].items():
        print(
            f"  {c:16s} n={v['n']:4d}  {v['pre_geo_mae']:>13,.0f} -> {v['winner_mae']:>13,.0f}   "
            f"MAPE {v['pre_geo_mape']:.1f}% -> {v['winner_mape']:.1f}%"
        )
    print("\nBy coordinate availability MAE (pre_geo -> winner):")
    for c, v in deep["by_coordinate_availability"].items():
        print(
            f"  {c:18s} n={v['n']:4d}  {v['pre_geo_mae']:>13,.0f} -> {v['winner_mae']:>13,.0f}   "
            f"MAPE {v['pre_geo_mape']:.1f}% -> {v['winner_mape']:.1f}%"
        )
    print("\nPrice tercile MAE (pre_geo -> winner):")
    for c, v in deep["price_tercile"].items():
        print(f"  {c:6s} n={v['n']:4d}  {v['pre_geo_mae']:>13,.0f} -> {v['winner_mae']:>13,.0f}")
    if "permutation_importance_top15" in deep:
        print("\nPermutation importance, top 15 (location holdout):")
        for k, v in deep["permutation_importance_top15"].items():
            tag = "  <-- geo" if k in GEO_FEATURE_COLUMNS else ""
            print(f"  {k:34s} {v:+.5f}{tag}")
        print(
            f"\nLeakage scan — suspicious feature names in matrix: "
            f"{deep['leakage_scan']['suspicious_feature_names_in_matrix'] or 'none'}"
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

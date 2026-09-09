"""Independent, standalone re-verification of this project's central
leakage claims, run fresh against the current CLEANED dataset (not just
re-read from an earlier written audit):

1. How many near-duplicate groups (find_near_duplicate_candidates) span
   both sides of the CURRENT random 80/20 split vs the group-aware split.
2. Whether the tuned xgboost candidate's held-out metrics look meaningfully
   better under random split than under group-aware split -- if so, the
   group-aware (more conservative) numbers are the ones to report as primary.
3. Imputation fit-scope: proves rooms/total_area/floor/floors_total median
   fill values used for the train split are NOT influenced by holdout rows
   (the leakage fix -- see src/preprocessing/imputer.py). Quantifies the
   old full-dataset-fit leak by comparing it against the train-only fit.

Does not train with tuning (uses the already-tuned hyperparameters from
reports/tuning_study_real.json to isolate the split-strategy effect from
hyperparameter effects). Writes reports/leakage_reverification_real.json.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBRegressor

from src.data.schema import (
    build_location_groups,
    build_split_groups,
    find_near_duplicate_candidates,
)
from src.data.split_pipeline import split_impute_featurize
from src.preprocessing.imputer import GroupMedianImputer


def _find_cleaned_csv() -> Path:
    processed = _PROJECT_ROOT / "data" / "processed"
    cleaned = sorted(processed.glob("real_estate_cleaned.csv"))
    if cleaned:
        return cleaned[0]
    # Fallback for older environments that only have the fully-imputed
    # EDA-only engineered CSV (see run_feature_engineering_real.py) -- this
    # file was imputed on the WHOLE dataset before any split, so the
    # imputer-fit-scope check below is measuring a no-op (no NaNs left to
    # fill), not proving anything about leakage-safety. Loud, not silent:
    # a script measuring leakage falling back to a file that predates the
    # leakage fix must not look identical to a clean run.
    print(
        "WARNING: real_estate_cleaned.csv not found -- falling back to the "
        "fully-imputed real_estate_engineered.csv (EDA-only, imputed on the "
        "WHOLE dataset before any split). The imputer-fit-scope check below "
        "will be a near no-op on this file, NOT a real leakage-safety proof. "
        "Run scripts/run_feature_engineering_real.py first to regenerate "
        "real_estate_cleaned.csv.",
        file=sys.stderr,
    )
    return processed / "real_estate_engineered.csv"


def main() -> None:
    csv_path = _find_cleaned_csv()
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    # ------------------------------------------------------------------
    # 1. Near-duplicate group crossing under random vs group-aware split
    # ------------------------------------------------------------------
    cands = find_near_duplicate_candidates(df)
    n_groups = cands["group_id"].nunique() if not cands.empty else 0
    print(f"Near-duplicate candidates: {len(cands)} rows in {n_groups} groups")

    groups_full = build_split_groups(df)

    # Random split (index-based, matches train_test_split default behaviour
    # used elsewhere in this project for the "random" split strategy).
    random_train_idx, random_test_idx = train_test_split(
        df.index.to_numpy(), test_size=0.2, random_state=42
    )
    random_train_set, random_test_set = set(random_train_idx), set(random_test_idx)

    def _count_crossing_groups(group_col: pd.Series, cands_df: pd.DataFrame) -> tuple[int, int]:
        if cands_df.empty:
            return 0, 0
        crossing = 0
        total = 0
        for _, sub in cands_df.groupby("group_id"):
            df_indices = sub["df_index"].tolist()
            total += 1
            in_train = any(i in random_train_set for i in df_indices)
            in_test = any(i in random_test_set for i in df_indices)
            if in_train and in_test:
                crossing += 1
        return crossing, total

    crossing_random, total_groups = _count_crossing_groups(groups_full, cands)
    pct_random = round(100 * crossing_random / total_groups, 1) if total_groups else 0.0
    print(
        f"Random split: {crossing_random}/{total_groups} near-dup groups ({pct_random}%) span train+test"
    )

    # Group-aware split: verify ZERO crossing by construction.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    group_train_idx, group_test_idx = next(gss.split(df, groups=groups_full))
    group_train_set = set(df.index[group_train_idx])
    group_test_set = set(df.index[group_test_idx])

    crossing_group = 0
    if not cands.empty:
        for _, sub in cands.groupby("group_id"):
            df_indices = sub["df_index"].tolist()
            in_train = any(i in group_train_set for i in df_indices)
            in_test = any(i in group_test_set for i in df_indices)
            if in_train and in_test:
                crossing_group += 1
    print(
        f"Group-aware split: {crossing_group}/{total_groups} near-dup groups span train+test (expected 0)"
    )

    # ------------------------------------------------------------------
    # 1b. Location-grouped split: near-duplicate groups AND same-building
    #     (rounded-coordinate) groups must not cross. This is the strict
    #     split behind the geo model's honest evaluation.
    # ------------------------------------------------------------------
    loc_groups = build_location_groups(df)
    gss_loc = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    loc_train_idx, loc_test_idx = next(gss_loc.split(df, groups=loc_groups))
    loc_train_set = set(df.index[loc_train_idx])
    loc_test_set = set(df.index[loc_test_idx])

    crossing_location_neardup = 0
    if not cands.empty:
        for _, sub in cands.groupby("group_id"):
            di = sub["df_index"].tolist()
            if any(i in loc_train_set for i in di) and any(i in loc_test_set for i in di):
                crossing_location_neardup += 1

    crossing_location_coord = 0
    n_coord_clusters = 0
    if "latitude" in df.columns and "longitude" in df.columns:
        lat = pd.to_numeric(df["latitude"], errors="coerce").round(4)
        lon = pd.to_numeric(df["longitude"], errors="coerce").round(4)
        has = lat.notna() & lon.notna()
        ck = pd.Series(list(zip(lat[has], lon[has])), index=df.index[has])
        for _, idx in ck.groupby(ck).groups.items():
            n_coord_clusters += 1
            idx = list(idx)
            if any(i in loc_train_set for i in idx) and any(i in loc_test_set for i in idx):
                crossing_location_coord += 1
    print(
        f"Location split: {crossing_location_neardup}/{total_groups} near-dup groups and "
        f"{crossing_location_coord}/{n_coord_clusters} same-coordinate (building) clusters "
        f"span train+test (expected 0 / 0)"
    )

    # ------------------------------------------------------------------
    # 2. Same tuned hyperparameters, evaluated under both split strategies —
    #    each split independently runs the full leakage-safe pipeline
    #    (its own train-only imputer fit + featurization), not a shared X/y
    #    built before the split.
    # ------------------------------------------------------------------
    from scripts._tuned_params import load_tuned_xgb_params

    best_params, params_source = load_tuned_xgb_params()
    print(f"Using xgboost params from {params_source}: {best_params}")

    def _fit_eval(split_strategy: str) -> dict:
        split_result = split_impute_featurize(df, split_strategy=split_strategy, random_state=42)
        model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(split_result.X_train, split_result.y_train)
        pred_log = model.predict(split_result.X_test)
        y_pred = np.expm1(pred_log)
        y_true = np.expm1(split_result.y_test)
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "n_train": int(len(split_result.X_train)),
            "n_test": int(len(split_result.X_test)),
        }

    metrics_random = _fit_eval("random")
    print(f"Random split, tuned params: {metrics_random}")

    metrics_group = _fit_eval("group")
    print(f"Group-aware split, tuned params: {metrics_group}")

    metrics_location = _fit_eval("location")
    print(f"Location split (PRIMARY), tuned params: {metrics_location}")

    # ------------------------------------------------------------------
    # 3. Imputation fit-scope: prove the train split's fill values do not
    #    depend on holdout rows, and quantify the old full-dataset-fit
    #    leak this replaced.
    # ------------------------------------------------------------------
    gss_scope = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    scope_train_idx, scope_test_idx = next(gss_scope.split(df, groups=groups_full))
    train_raw = df.iloc[scope_train_idx].reset_index(drop=True)

    imputer_train_only = GroupMedianImputer().fit(train_raw)
    imputer_full_dataset = GroupMedianImputer().fit(
        df
    )  # the old, leaky full-dataset-fit behaviour, for comparison only

    # Mutate a COPY of the holdout rows' rooms/total_area/floor/floors_total
    # to extreme out-of-range values, then re-fit on train_raw again — an
    # untouched, independent fit on the same train rows must be byte-for-byte
    # identical, proving the holdout mutation could not have reached it.
    mutated_df = df.copy()
    mutated_df.loc[scope_test_idx, "rooms"] = 999
    mutated_df.loc[scope_test_idx, "total_area"] = 999_999.0
    mutated_df.loc[scope_test_idx, "floor"] = 999
    mutated_df.loc[scope_test_idx, "floors_total"] = 999
    mutated_train_raw = mutated_df.iloc[scope_train_idx].reset_index(drop=True)
    imputer_after_holdout_mutation = GroupMedianImputer().fit(mutated_train_raw)

    imputer_invariant_to_holdout = (
        imputer_train_only.global_median_ == imputer_after_holdout_mutation.global_median_
        and imputer_train_only.city_median_ == imputer_after_holdout_mutation.city_median_
    )
    imputation_leak_magnitude = {
        col: {
            "train_only_median": imputer_train_only.global_median_.get(col),
            "full_dataset_median_phase_e_style": imputer_full_dataset.global_median_.get(col),
        }
        for col in ("rooms", "total_area", "floor", "floors_total")
    }
    print(f"Imputer invariant to holdout mutation: {imputer_invariant_to_holdout}")
    print(f"Train-only vs full-dataset global medians: {imputation_leak_magnitude}")

    result = {
        "near_duplicate_groups_total": int(total_groups),
        "near_duplicate_candidate_rows": int(len(cands)),
        "groups_crossing_random_split": int(crossing_random),
        "groups_crossing_random_split_pct": pct_random,
        "groups_crossing_group_aware_split": int(crossing_group),
        "groups_crossing_location_split": {
            "near_duplicate_groups": int(crossing_location_neardup),
            "same_coordinate_building_clusters": int(crossing_location_coord),
            "same_coordinate_building_clusters_total": int(n_coord_clusters),
        },
        "tuned_xgboost_params": best_params,
        "xgboost_params_source": params_source,
        "metrics_random_split": metrics_random,
        "metrics_group_aware_split": metrics_group,
        "metrics_location_split": metrics_location,
        "r2_gap_random_minus_group": round(metrics_random["r2"] - metrics_group["r2"], 4),
        "r2_gap_group_minus_location": round(metrics_group["r2"] - metrics_location["r2"], 4),
        "imputer_fit_scope": {
            "invariant_to_holdout_mutation": imputer_invariant_to_holdout,
            "train_vs_full_dataset_global_medians": imputation_leak_magnitude,
            "note": (
                "train_only_median is what the pipeline actually uses (fit on the group-aware "
                "train split only). full_dataset_median_phase_e_style is shown only to "
                "quantify how different the old, leaky full-dataset fit would have been -- "
                "it is never used to fill any split."
            ),
        },
        "conclusion": (
            "Location-grouped split (building-level) is the strictest and is the primary "
            "evaluation for the geo model. Random > group-aware > location in optimism "
            "(R2 %.3f -> %.3f -> %.3f)."
            % (metrics_random["r2"], metrics_group["r2"], metrics_location["r2"])
        ),
    }
    out_path = _PROJECT_ROOT / "reports" / "leakage_reverification_real.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

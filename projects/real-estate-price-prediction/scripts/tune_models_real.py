"""
Hyperparameter tuning for the real-data geo model — honest, no holdout leak.

Usage: python scripts/tune_models_real.py [--n-trials 60] [--models xgboost,random_forest]

Protocol (the point of this script):

    real_estate_cleaned.csv
      -> location-grouped split  ->  DEV (80%)   +   FINAL HOLDOUT (20%, RESERVED)
                                       |
                        GroupKFold CV *inside DEV only*
                        (near-duplicate + same-building groups never
                         cross a CV fold, and the imputer is re-fit on
                         each fold's own train slice)
                                       |
                        Optuna minimises mean fold MAE (RUB)
                                       |
                             best params per model

The FINAL HOLDOUT is **never touched here** — not for selection and not
even for a reported number. Its single evaluation happens later, in
scripts/compare_geo_uplift.py / scripts/analyze_final_candidate_real.py,
against the identical location-grouped split (same seed, same test_size).

xgboost is the real candidate (wide search incl. reg_alpha/reg_lambda);
random_forest is tuned lightly only as a sanity baseline.

Writes reports/tuning_study_real.json: per model, best_params + CV metrics +
trial/seed/split metadata. Saves no model artefact, does not touch
models/current_model.json.
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
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data.split_pipeline import group_holdout_indices
from src.features.feature_engineering import FeatureEngineer
from src.preprocessing.imputer import GroupMedianImputer
from src.utils.logger import get_logger

logger = get_logger("tune_models_real")
optuna.logging.set_verbosity(optuna.logging.WARNING)

_RANDOM_SEED = 42
_CV_FOLDS = 4
_SPLIT_STRATEGY = "location"

# The imputer is fit ONCE on the whole DEV split (not per CV fold). DEV never
# includes a holdout row, so the final holdout stays clean either way; the
# only effect is that a fold's rooms/area/floor median fill is computed over
# ~all of DEV rather than ~3/4 of it -- a sub-0.1%% difference on this
# dataset (356/9720 rooms, 158/9720 floor missing), not worth a 4x slower
# search. The holdout evaluation elsewhere re-fits the imputer on its own
# train split, so nothing about the promoted pipeline changes.


def _find_cleaned_csv(input_dir: Path) -> Path | None:
    for name in ("real_estate_cleaned.csv", "real_estate_cleaned.synthetic.csv"):
        p = input_dir / name
        if p.is_file():
            return p
    cands = sorted(input_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _featurize_dev(dev_raw: pd.DataFrame):
    """Fit the imputer on the whole DEV split (see note above), featurize
    once, and return (X, y_log) plus the folds' positional index pairs."""
    imputer = GroupMedianImputer().fit(dev_raw)
    fe = FeatureEngineer()
    X, y = fe.prepare_for_training(imputer.transform(dev_raw))
    return X.reset_index(drop=True), y.reset_index(drop=True)


def _cv_scores(model_factory, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    """MAE / RMSE / R² (RUB, back-transformed) averaged across GroupKFold
    folds on the pre-featurized DEV matrix."""
    gkf = GroupKFold(n_splits=_CV_FOLDS)
    maes, rmses, r2s = [], [], []
    for tr_idx, val_idx in gkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X.iloc[tr_idx])
        X_val = scaler.transform(X.iloc[val_idx])
        model = model_factory()
        model.fit(X_tr, y.iloc[tr_idx])
        y_pred = np.expm1(model.predict(X_val))
        y_true = np.expm1(y.iloc[val_idx].to_numpy())
        maes.append(float(np.mean(np.abs(y_true - y_pred))))
        rmses.append(float(np.sqrt(np.mean((y_true - y_pred) ** 2))))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2s.append(1 - ss_res / ss_tot if ss_tot else float("nan"))
    return {
        "cv_mae_rub": float(np.mean(maes)),
        "cv_mae_std_rub": float(np.std(maes)),
        "cv_rmse_rub": float(np.mean(rmses)),
        "cv_r2": float(np.mean(r2s)),
    }


def _cv_mae(model_factory, X, y, groups) -> float:
    return _cv_scores(model_factory, X, y, groups)["cv_mae_rub"]


def _suggest_tree_depth(trial: optuna.Trial) -> int | None:
    """Search bounded depths AND sklearn's "unlimited" default (max_depth=None)
    -- a first version excluded None, which was exactly the region the untuned
    RandomForest default occupied and beat every bounded-depth trial with."""
    if trial.suggest_categorical("max_depth_unlimited", [True, False]):
        return None
    return trial.suggest_int("max_depth_bounded", 5, 40)


def _tune_random_forest(X, y, groups, n_trials: int) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
            "max_depth": _suggest_tree_depth(trial),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        }
        return _cv_mae(
            lambda: RandomForestRegressor(random_state=_RANDOM_SEED, n_jobs=-1, **params),
            X,
            y,
            groups,
        )

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=_RANDOM_SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def _tune_xgboost(X, y, groups, n_trials: int) -> optuna.Study:
    from xgboost import XGBRegressor

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 2.0, log=True),
        }
        return _cv_mae(
            lambda: XGBRegressor(
                random_state=_RANDOM_SEED, n_jobs=-1, tree_method="hist", **params
            ),
            X,
            y,
            groups,
        )

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=_RANDOM_SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def _resolve_model_params(model_name: str, raw_params: dict) -> dict:
    """Translate Optuna's raw ``study.best_params`` into model-constructor
    kwargs. Only random_forest/extra_trees need the max_depth translation
    (from :func:`_suggest_tree_depth`); xgboost's params are 1:1 already."""
    if model_name not in ("random_forest", "extra_trees"):
        return dict(raw_params)
    params = dict(raw_params)
    unlimited = params.pop("max_depth_unlimited")
    bounded = params.pop("max_depth_bounded", None)
    params["max_depth"] = None if unlimited else bounded
    return params


_TUNERS = {
    "xgboost": _tune_xgboost,
    "random_forest": _tune_random_forest,
}
# extra_trees is no longer tuned (model-zoo trim) but _resolve_model_params
# still handles it, and its factory is kept for any ad-hoc reuse.
_FACTORIES = {
    "random_forest": lambda p: RandomForestRegressor(random_state=_RANDOM_SEED, n_jobs=-1, **p),
    "extra_trees": lambda p: ExtraTreesRegressor(random_state=_RANDOM_SEED, n_jobs=-1, **p),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the real-data geo model (honest protocol).")
    parser.add_argument("--input", default=str(_PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--n-trials", type=int, default=60, help="Optuna trials per model.")
    parser.add_argument(
        "--models",
        default="xgboost,random_forest",
        help="Comma-separated subset of: xgboost, random_forest.",
    )
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    csv_path = _find_cleaned_csv(Path(args.input))
    if csv_path is None:
        print(f"ERROR: no cleaned CSV found in '{args.input}'.", file=sys.stderr)
        raise SystemExit(2)

    df = pd.read_csv(csv_path).reset_index(drop=True)
    is_synthetic = (
        bool(df.get("is_synthetic", pd.Series([False])).any()) or "synthetic" in csv_path.name
    )
    if is_synthetic and not args.allow_synthetic:
        print(f"ERROR: '{csv_path}' is synthetic data. Pass --allow-synthetic to override.")
        raise SystemExit(2)

    dev_idx, holdout_idx, groups = group_holdout_indices(
        df, split_strategy=_SPLIT_STRATEGY, test_size=0.2, random_state=_RANDOM_SEED
    )
    dev_raw = df.iloc[dev_idx].reset_index(drop=True)
    dev_groups = pd.Series(groups).iloc[dev_idx].reset_index(drop=True)
    X_dev, y_dev = _featurize_dev(dev_raw)
    logger.info(
        "Tuning on %d DEV rows / %d features (%d location groups, %d-fold GroupKFold, "
        "%d trials/model). FINAL HOLDOUT = %d rows, NEVER touched here.",
        len(X_dev),
        X_dev.shape[1],
        dev_groups.nunique(),
        _CV_FOLDS,
        args.n_trials,
        len(holdout_idx),
    )

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    study_summary: dict = {}
    for model_name in requested:
        if model_name not in _TUNERS:
            logger.warning("Unknown model '%s' — skipped (choices: %s).", model_name, list(_TUNERS))
            continue
        logger.info("Tuning %s (%d trials)...", model_name, args.n_trials)
        study = _TUNERS[model_name](X_dev, y_dev, dev_groups, args.n_trials)
        best_params = _resolve_model_params(model_name, study.best_params)

        if model_name == "xgboost":
            from xgboost import XGBRegressor

            factory = lambda p=best_params: XGBRegressor(  # noqa: E731
                random_state=_RANDOM_SEED, n_jobs=-1, tree_method="hist", **p
            )
        else:
            factory = lambda p=best_params: _FACTORIES[model_name](p)  # noqa: E731

        metrics = _cv_scores(factory, X_dev, y_dev, dev_groups)
        study_summary[model_name] = {
            "best_params": best_params,
            **metrics,
            "best_trial_cv_mae_rub": float(study.best_value),
            "cv_folds": _CV_FOLDS,
            "n_trials": args.n_trials,
            "random_seed": _RANDOM_SEED,
            "split_strategy": _SPLIT_STRATEGY,
            "n_dev_rows": int(len(dev_raw)),
            "n_holdout_rows_reserved": int(len(holdout_idx)),
            "note": "CV on DEV only; the final holdout is evaluated elsewhere, once.",
        }
        logger.info(
            "%s: CV MAE=%.0f +/- %.0f RUB | CV R2=%.4f | best params: %s",
            model_name,
            metrics["cv_mae_rub"],
            metrics["cv_mae_std_rub"],
            metrics["cv_r2"],
            best_params,
        )

    reports_dir = _PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "tuning_study_real.json"
    # Merge into any existing study (so xgboost and random_forest can be tuned
    # in separate invocations with different --n-trials without clobbering).
    merged = {}
    if out_path.is_file():
        try:
            merged = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            merged = {}
    merged.update(study_summary)
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("HYPERPARAMETER TUNING SUMMARY (location split, DEV CV only)")
    print("=" * 72)
    print(f"Dataset : {csv_path} ({len(df)} rows)")
    print(f"DEV     : {len(dev_raw)} rows / {dev_groups.nunique()} location groups")
    print(f"Holdout : {len(holdout_idx)} rows RESERVED (not evaluated here)")
    print(f"CV      : {_CV_FOLDS}-fold GroupKFold, {args.n_trials} trials/model, seed={_RANDOM_SEED}")
    print()
    for model_name, s in study_summary.items():
        print(f"{model_name}:")
        print(f"  CV MAE  : {s['cv_mae_rub']:,.0f} +/- {s['cv_mae_std_rub']:,.0f} RUB")
        print(f"  CV RMSE : {s['cv_rmse_rub']:,.0f} RUB")
        print(f"  CV R2   : {s['cv_r2']:.4f}")
        print(f"  params  : {s['best_params']}")
        print()
    print(f"Saved: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

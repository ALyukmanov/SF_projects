"""
Honest hyperparameter tuning for the top real-data model candidates.
Usage: python scripts/tune_models_real.py [--input path] [--n-trials 20]

Tunes `random_forest`, `extra_trees`, `xgboost` (the three closest
contenders from scripts/compare_models_real.py's benchmark) using Optuna.

Protocol (the point of this script, not an implementation detail):
    train -> internal group-aware CV on train only -> best params selected
    by CV score -> refit on the FULL train set -> ONE final evaluation on
    the held-out test set, which is never touched during tuning.

The held-out test set is never scored during the search — Optuna's
objective function only ever sees group-aware CV folds carved out of the
train split, so "don't select a model by the test set" holds by
construction, not by discipline alone. CV folds are built with
``GroupKFold`` over the same near-duplicate grouping used for the main
train/test split (see ``src.data.schema.build_split_groups``), so a
near-duplicate pair can't leak across CV folds either — the same leakage
class the leakage audit's Finding 2 flagged, guarded against here too, not
just at the outer split.

Saves reports/tuning_study_real.json (per-model: best params, CV MAE,
final held-out test metrics, n_trials, seed) and prints a summary table.
Does NOT save a model artefact or touch models/current_model.json.
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data.schema import build_split_groups
from src.features.feature_engineering import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger("tune_models_real")
optuna.logging.set_verbosity(optuna.logging.WARNING)

_RANDOM_SEED = 42
_CV_FOLDS = 3


def _find_latest_csv(directory: Path, pattern: str = "*.csv") -> Path | None:
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _cv_mae(model_factory, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> float:
    """Mean MAE (RUB, back-transformed) across GroupKFold folds on (X, y)."""
    gkf = GroupKFold(n_splits=_CV_FOLDS)
    fold_maes = []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        model = model_factory()
        model.fit(X_tr_scaled, y_tr)
        y_pred = np.expm1(model.predict(X_val_scaled))
        y_true = np.expm1(y_val.values)
        fold_maes.append(float(np.mean(np.abs(y_true - y_pred))))
    return float(np.mean(fold_maes))


def _suggest_tree_depth(trial: optuna.Trial) -> int | None:
    """Search over both bounded depths AND "unlimited" (sklearn's own
    default for RandomForest/ExtraTrees, max_depth=None) -- a first version
    of this search space only tried [5, 30] and excluded None entirely,
    which turned out to be exactly the region the untuned sklearn defaults
    already occupied and outperformed every bounded-depth trial with on
    this dataset. Re-including it here so tuning can't do worse than the
    default it's supposed to improve on."""
    if trial.suggest_categorical("max_depth_unlimited", [True, False]):
        return None
    return trial.suggest_int("max_depth_bounded", 5, 40)


def _tune_random_forest(X, y, groups, n_trials: int) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": _suggest_tree_depth(trial),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        }

        def model_factory():
            return RandomForestRegressor(random_state=_RANDOM_SEED, n_jobs=-1, **params)

        return _cv_mae(model_factory, X, y, groups)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=_RANDOM_SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def _tune_extra_trees(X, y, groups, n_trials: int) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": _suggest_tree_depth(trial),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        }

        def model_factory():
            return ExtraTreesRegressor(random_state=_RANDOM_SEED, n_jobs=-1, **params)

        return _cv_mae(model_factory, X, y, groups)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=_RANDOM_SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def _tune_xgboost(X, y, groups, n_trials: int) -> optuna.Study:
    from xgboost import XGBRegressor

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        def model_factory():
            return XGBRegressor(random_state=_RANDOM_SEED, n_jobs=-1, **params)

        return _cv_mae(model_factory, X, y, groups)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=_RANDOM_SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def _resolve_model_params(model_name: str, raw_params: dict) -> dict:
    """Translate Optuna's raw ``study.best_params`` (which contains the
    trial-suggestion names, e.g. ``max_depth_unlimited``/``max_depth_bounded``
    from :func:`_suggest_tree_depth`) into actual model-constructor kwargs.
    ``xgboost``'s params already match XGBRegressor kwargs 1:1 (no
    translation needed) since its objective suggests them directly."""
    if model_name not in ("random_forest", "extra_trees"):
        return dict(raw_params)
    params = dict(raw_params)
    unlimited = params.pop("max_depth_unlimited")
    bounded = params.pop("max_depth_bounded", None)
    params["max_depth"] = None if unlimited else bounded
    return params


_TUNERS = {
    "random_forest": (
        _tune_random_forest,
        lambda p: RandomForestRegressor(random_state=_RANDOM_SEED, n_jobs=-1, **p),
    ),
    "extra_trees": (
        _tune_extra_trees,
        lambda p: ExtraTreesRegressor(random_state=_RANDOM_SEED, n_jobs=-1, **p),
    ),
    "xgboost": (_tune_xgboost, None),  # factory built lazily below (needs XGBRegressor import)
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune top real-data model candidates (honest protocol)."
    )
    parser.add_argument("--input", default=str(_PROJECT_ROOT / "data" / "processed"))
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Optuna trials per model (default: 20)."
    )
    parser.add_argument("--allow-synthetic", action="store_true")
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
            f"ERROR: '{csv_path}' is synthetic data. Pass --allow-synthetic to override.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    feat_eng = FeatureEngineer()
    X, y = feat_eng.prepare_for_training(df)
    groups_all = build_split_groups(df).loc[X.index]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=_RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups_all))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups_all.iloc[train_idx]

    logger.info(
        "Tuning on %d train rows (%d CV folds, %d trials/model); held-out test = %d rows, "
        "NEVER scored during search.",
        len(X_train),
        _CV_FOLDS,
        args.n_trials,
        len(X_test),
    )

    study_summary = {}
    for model_name, (tuner_fn, factory_fn) in _TUNERS.items():
        logger.info("Tuning %s...", model_name)
        study = tuner_fn(X_train, y_train, groups_train, args.n_trials)
        best_params = _resolve_model_params(model_name, study.best_params)
        best_cv_mae = study.best_value

        if model_name == "xgboost":
            from xgboost import XGBRegressor

            def factory_fn(p):
                return XGBRegressor(random_state=_RANDOM_SEED, n_jobs=-1, **p)

        # Final, single evaluation on the held-out test set -- refit on the
        # FULL train split with the winning params, never re-tuned against test.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        final_model = factory_fn(best_params)
        final_model.fit(X_train_scaled, y_train)
        y_pred = np.expm1(final_model.predict(X_test_scaled))
        y_true = np.expm1(y_test.values)
        test_mae = float(np.mean(np.abs(y_true - y_pred)))
        test_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        test_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        study_summary[model_name] = {
            "best_params": best_params,
            "cv_mae_rub": best_cv_mae,
            "cv_folds": _CV_FOLDS,
            "n_trials": args.n_trials,
            "random_seed": _RANDOM_SEED,
            "test_mae_rub": test_mae,
            "test_rmse_rub": test_rmse,
            "test_r2": test_r2,
        }
        logger.info(
            "%s: CV MAE=%.0f RUB (train-only) | held-out test MAE=%.0f RUB, RMSE=%.0f RUB, R2=%.4f",
            model_name,
            best_cv_mae,
            test_mae,
            test_rmse,
            test_r2,
        )

    reports_dir = _PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "tuning_study_real.json"
    out_path.write_text(json.dumps(study_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING SUMMARY" + (" [SYNTHETIC — smoke test]" if is_synthetic else ""))
    print("=" * 70)
    print(f"Dataset: {csv_path} ({len(df)} rows)")
    print(
        f"Train/test: {len(X_train)}/{len(X_test)} (group-aware split, held-out test never scored during search)"
    )
    print(
        f"CV: {_CV_FOLDS}-fold GroupKFold on train only, {args.n_trials} Optuna trials/model, seed={_RANDOM_SEED}"
    )
    print()
    for model_name, summary in study_summary.items():
        print(f"{model_name}:")
        print(f"  best params    : {summary['best_params']}")
        print(f"  CV MAE (train) : {summary['cv_mae_rub']:,.0f} RUB")
        print(f"  TEST MAE       : {summary['test_mae_rub']:,.0f} RUB")
        print(f"  TEST RMSE      : {summary['test_rmse_rub']:,.0f} RUB")
        print(f"  TEST R2        : {summary['test_r2']:.4f}")
        print()
    print(f"Saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

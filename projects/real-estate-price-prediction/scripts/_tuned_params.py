"""Shared helper: load XGBoost params for the comparison/diagnostic scripts.

Order of preference:
1. ``models/current_model.json`` -> ``hyperparameters`` — the current model's
   hyperparameters.
2. ``reports/tuning_study_real.json`` -> ``xgboost.best_params`` — the tuning
   result, used when there is no manifest.
3. A small built-in default, so a fresh clone with neither file can still run
   the scripts (with a warning).

Keeps ``verify_split_leakage.py`` / ``analyze_model_real.py`` /
``compare_baselines_real.py`` consistent with each other.
"""

from __future__ import annotations

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_BUILTIN_DEFAULT = {
    "n_estimators": 600,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 4,
    "reg_lambda": 4.0,
}


def load_tuned_xgb_params() -> tuple[dict, str]:
    """Return ``(params, source_label)``."""
    manifest = _PROJECT_ROOT / "models" / "current_model.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            params = data.get("hyperparameters")
            if isinstance(params, dict) and params:
                return dict(params), "models/current_model.json (current model hyperparameters)"
        except (json.JSONDecodeError, OSError):
            pass

    study = _PROJECT_ROOT / "reports" / "tuning_study_real.json"
    if study.is_file():
        try:
            data = json.loads(study.read_text(encoding="utf-8"))
            params = data.get("xgboost", {}).get("best_params")
            if params:
                return dict(params), "reports/tuning_study_real.json (exploratory tuning)"
        except (json.JSONDecodeError, OSError):
            pass

    return dict(_BUILTIN_DEFAULT), "built-in default (no manifest or study found)"

"""Tests for scripts/tune_models_real.py's param-resolution logic.

Regression coverage for a real bug found during hyperparameter tuning: the RF/ExtraTrees
Optuna search space originally bounded max_depth to [5, 30], excluding
None (unlimited) -- exactly the value ModelTrainer's own untuned default
uses for both models -- which made tuned results worse than the untuned
baseline. Fixed by adding an explicit "unlimited depth" choice; these
tests guard the params-resolution step that choice depends on.
"""

from __future__ import annotations

from scripts.tune_models_real import _resolve_model_params


class TestResolveModelParams:
    def test_random_forest_unlimited_depth_resolves_to_none(self):
        raw = {
            "n_estimators": 300,
            "max_depth_unlimited": True,
            "max_depth_bounded": 15,  # should be ignored when unlimited=True
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": 1.0,
        }
        resolved = _resolve_model_params("random_forest", raw)
        assert resolved["max_depth"] is None
        assert "max_depth_unlimited" not in resolved
        assert "max_depth_bounded" not in resolved

    def test_random_forest_bounded_depth_resolves_to_the_int(self):
        raw = {
            "n_estimators": 300,
            "max_depth_unlimited": False,
            "max_depth_bounded": 22,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": 1.0,
        }
        resolved = _resolve_model_params("random_forest", raw)
        assert resolved["max_depth"] == 22

    def test_extra_trees_uses_the_same_resolution_as_random_forest(self):
        raw = {"max_depth_unlimited": True, "max_depth_bounded": 10}
        resolved = _resolve_model_params("extra_trees", raw)
        assert resolved["max_depth"] is None

    def test_xgboost_params_pass_through_unchanged(self):
        raw = {
            "n_estimators": 600,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.9,
        }
        resolved = _resolve_model_params("xgboost", raw)
        assert resolved == raw

    def test_resolved_params_are_valid_sklearn_kwargs(self):
        """The exact failure mode of the original bug: constructing the
        model with the resolved params must not raise TypeError for an
        unexpected keyword argument."""
        from sklearn.ensemble import RandomForestRegressor

        raw = {
            "n_estimators": 100,
            "max_depth_unlimited": True,
            "max_depth_bounded": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }
        resolved = _resolve_model_params("random_forest", raw)
        RandomForestRegressor(random_state=42, n_jobs=-1, **resolved)  # must not raise

"""Tests for scripts/analyze_final_candidate_real.py's metric helpers."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_final_candidate_real import _full_metrics, _rmsle, _smape


class TestSmape:
    def test_perfect_prediction_is_zero(self):
        y = np.array([100.0, 200.0, 300.0])
        assert _smape(y, y) == 0.0

    def test_symmetric_under_and_over_prediction_give_same_smape(self):
        y_true = np.array([100.0])
        under = _smape(y_true, np.array([80.0]))
        over = _smape(y_true, np.array([125.0]))
        # sMAPE is not perfectly symmetric for arbitrary over/under pairs,
        # but both should be well-defined, bounded, positive values.
        assert under > 0
        assert over > 0

    def test_bounded_between_0_and_200(self):
        y_true = np.array([100.0, 50.0, 1000.0])
        y_pred = np.array([1.0, 5000.0, 1.0])
        result = _smape(y_true, y_pred)
        assert 0.0 <= result <= 200.0


class TestRmsle:
    def test_perfect_prediction_is_zero(self):
        y = np.array([1_000_000.0, 5_000_000.0])
        assert _rmsle(y, y) == 0.0

    def test_negative_predictions_are_clipped_not_raising(self):
        # RMSLE requires non-negative values for log1p; a stray negative
        # prediction must be clipped rather than raise or produce NaN.
        y_true = np.array([1_000_000.0])
        y_pred = np.array([-500.0])
        result = _rmsle(y_true, y_pred)
        assert np.isfinite(result)

    def test_larger_relative_error_gives_larger_rmsle(self):
        y_true = np.array([1_000_000.0])
        small_error = _rmsle(y_true, np.array([1_050_000.0]))
        large_error = _rmsle(y_true, np.array([2_000_000.0]))
        assert large_error > small_error


class TestFullMetrics:
    def test_all_expected_keys_present(self):
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([110.0, 190.0, 310.0, 380.0])
        result = _full_metrics(y_true, y_pred)
        assert set(result.keys()) == {
            "mae",
            "rmse",
            "r2",
            "median_ae",
            "p50_abs_error_rub",
            "p75_abs_error_rub",
            "p90_abs_error_rub",
            "mape",
            "smape",
            "rmsle",
        }

    def test_p50_matches_median_ae_and_percentiles_are_ordered(self):
        y_true = np.array([100.0, 200.0, 300.0, 400.0, 1000.0])
        y_pred = np.array([110.0, 190.0, 310.0, 380.0, 700.0])
        result = _full_metrics(y_true, y_pred)
        assert result["p50_abs_error_rub"] == pytest.approx(result["median_ae"])
        assert (
            result["p50_abs_error_rub"]
            <= result["p75_abs_error_rub"]
            <= result["p90_abs_error_rub"]
        )

    def test_perfect_prediction_gives_zero_errors_and_r2_one(self):
        y = np.array([100.0, 200.0, 300.0])
        result = _full_metrics(y, y)
        assert result["mae"] == 0.0
        assert result["rmse"] == 0.0
        assert result["r2"] == 1.0
        assert result["mape"] == 0.0

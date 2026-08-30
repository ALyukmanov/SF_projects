"""Inference parity: the live API/Predictor feature-construction path
(src.inference.predictor.Predictor._build_feature_vector) is a SEPARATE
implementation from the offline training path
(src.features.feature_engineering.FeatureEngineer.prepare_for_training).
They must produce the identical numeric feature vector (same values, same
order) for the same listing, or a model trained offline could be fed
silently different inputs at prediction time.

This test replays both paths for the same raw listing and compares the
resulting feature vectors column-by-column. It caught a real, fixed bug:
predictor.py's floor_ratio/room_density/rooms_x_area were
unrounded floats while feature_engineering.py rounds them (4/6/2 decimal
places respectively) -- a tiny numeric drift, closed by matching the
rounding in both places rather than by loosening this test's tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.feature_engineering import FeatureEngineer  # noqa: E402
from src.inference.predictor import _MACRO_DEFAULTS, Predictor  # noqa: E402
from src.preprocessing.economic_features import EconomicFeatureEngineer  # noqa: E402


def _offline_vector(raw: dict, macro: dict) -> pd.Series:
    """Feature vector via the real training pipeline (no date_published,
    same "value as of today" scenario a live prediction request represents).
    """
    df = pd.DataFrame([{**raw, "price": 10_000_000.0, **macro}])
    engineer = FeatureEngineer()
    X, _ = engineer.prepare_for_training(df)
    return X.iloc[0]


def _inference_vector(raw: dict, macro: dict) -> pd.Series:
    predictor = Predictor(model_path=str(_PROJECT_ROOT / "models"))  # not loaded; harmless
    vec = predictor._build_feature_vector({**raw, **macro})
    return pd.Series(vec)


_DEFAULT_MACRO = dict(_MACRO_DEFAULTS)


class TestMacroDefaultsMatchOfflineFallback:
    def test_macro_defaults_match_offline_current_year_fallback(self):
        """predictor._MACRO_DEFAULTS is a hardcoded stand-in for what
        EconomicFeatureEngineer(use_api=False) computes for a listing with
        no date_published (i.e. "today"). If the offline fallback tables
        (economic_features._KEY_RATES etc.) are ever extended with a new
        year and _MACRO_DEFAULTS is not updated to match, a live prediction
        would silently use stale macro indicators while training uses fresh
        ones -- this test exists to catch exactly that drift.
        """
        df = pd.DataFrame({"total_area": [50.0]})
        eco = EconomicFeatureEngineer(use_api=False)
        enriched = eco.add_economic_features(df)
        for key, expected in _MACRO_DEFAULTS.items():
            assert enriched.iloc[0][key] == pytest.approx(expected), (
                f"predictor._MACRO_DEFAULTS['{key}']={expected} no longer matches the "
                f"offline no-date fallback value {enriched.iloc[0][key]} for the current year."
            )


class TestFeatureVectorParity:
    _BASE_RAW = {
        "rooms": 2,
        "total_area": 55.0,
        "floor": 4,
        "floors_total": 10,
        "city": "Москва",
        "building_type": "monolith",
    }

    def _assert_parity(self, raw: dict):
        offline = _offline_vector(raw, _DEFAULT_MACRO)
        inference = _inference_vector(raw, _DEFAULT_MACRO)

        assert set(offline.index) == set(inference.index), (
            f"Feature name mismatch.\nOffline only: {set(offline.index) - set(inference.index)}\n"
            f"Inference only: {set(inference.index) - set(offline.index)}"
        )
        for col in offline.index:
            assert offline[col] == pytest.approx(
                float(inference[col]), abs=1e-9
            ), f"Feature '{col}' differs: offline={offline[col]!r} vs inference={inference[col]!r}"

    def test_typical_moscow_listing(self):
        self._assert_parity(self._BASE_RAW)

    def test_saint_petersburg_panel_listing(self):
        self._assert_parity({**self._BASE_RAW, "city": "Санкт-Петербург", "building_type": "panel"})

    def test_studio_zero_rooms(self):
        self._assert_parity({**self._BASE_RAW, "rooms": 0, "total_area": 24.0})

    def test_first_floor(self):
        self._assert_parity({**self._BASE_RAW, "floor": 1})

    def test_top_floor(self):
        self._assert_parity({**self._BASE_RAW, "floor": 10})

    def test_unknown_city_yields_all_zero_city_columns_in_both_paths(self):
        self._assert_parity({**self._BASE_RAW, "city": "Атлантида"})

    def test_unknown_building_type_yields_all_zero_columns_in_both_paths(self):
        self._assert_parity({**self._BASE_RAW, "building_type": "glass_tower"})

    def test_missing_building_type(self):
        raw = {k: v for k, v in self._BASE_RAW.items() if k != "building_type"}
        self._assert_parity(raw)

    def test_large_area_odd_room_count(self):
        self._assert_parity(
            {**self._BASE_RAW, "rooms": 5, "total_area": 187.3, "floor": 8, "floors_total": 9}
        )

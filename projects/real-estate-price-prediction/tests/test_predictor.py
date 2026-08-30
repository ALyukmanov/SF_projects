"""Tests for the inference predictor."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.inference.predictor import Predictor

# ---------------------------------------------------------------------------
# Supported cities (from config.yaml / predictor._CITY_BASE_PRICE_PER_SQM)
# ---------------------------------------------------------------------------
_SUPPORTED_CITIES = [
    "москва",
    "санкт-петербург",
    "екатеринбург",
    "новосибирск",
    "казань",
    "нижний новгород",
    "самара",
    "краснодар",
]

# A standard feature dict used across most tests
_SAMPLE_FEATURES = {
    "rooms": 2,
    "total_area": 55.0,
    "floor": 4,
    "floors_total": 10,
    "city": "москва",
    "year_built": 2005,
}


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def demo_predictor(tmp_path: Path) -> Predictor:
    """A Predictor pointed at an empty directory → DEMO mode."""
    empty_models_dir = tmp_path / "models_empty"
    empty_models_dir.mkdir()
    p = Predictor(model_path=str(empty_models_dir))
    p.load()  # will return False, stays in DEMO mode
    return p


# ===========================================================================
# Tests
# ===========================================================================


class TestPredictorDemoMode:
    def test_predictor_demo_mode(self, demo_predictor: Predictor):
        """Predictor with no model file should operate in DEMO mode."""
        result = demo_predictor.predict(_SAMPLE_FEATURES)
        assert (
            result.get("mode") == "demo"
        ), f"Expected mode='demo', got mode='{result.get('mode')}'."

    def test_predictor_predict_returns_dict(self, demo_predictor: Predictor):
        """predict() must return a dict with price, price_min, price_max keys."""
        result = demo_predictor.predict(_SAMPLE_FEATURES)
        for key in ("price", "price_min", "price_max"):
            assert key in result, f"Key '{key}' missing from prediction result."

    def test_predictor_price_positive(self, demo_predictor: Predictor):
        """Predicted price must be strictly positive."""
        result = demo_predictor.predict(_SAMPLE_FEATURES)
        assert result["price"] > 0, "Predicted price should be > 0."

    def test_predictor_price_range_valid(self, demo_predictor: Predictor):
        """price_min <= price <= price_max must hold."""
        result = demo_predictor.predict(_SAMPLE_FEATURES)
        assert (
            result["price_min"] <= result["price"] <= result["price_max"]
        ), f"Price range invalid: {result['price_min']} <= {result['price']} <= {result['price_max']}"

    def test_predictor_all_cities(self, tmp_path: Path):
        """Prediction should succeed for every supported city."""
        empty_models_dir = tmp_path / "models_cities"
        empty_models_dir.mkdir()

        for city in _SUPPORTED_CITIES:
            predictor = Predictor(model_path=str(empty_models_dir))
            predictor.load()

            features = {**_SAMPLE_FEATURES, "city": city}
            result = predictor.predict(features)

            assert result["price"] > 0, f"Price should be > 0 for city='{city}'."
            assert (
                result["price_min"] <= result["price"] <= result["price_max"]
            ), f"Price range invalid for city='{city}'."

    def test_predictor_is_ready_false_when_no_model(self, tmp_path: Path):
        """is_ready() must return False when no model file is present."""
        empty_models_dir = tmp_path / "models_no_model"
        empty_models_dir.mkdir()
        predictor = Predictor(model_path=str(empty_models_dir))
        predictor.load()
        assert (
            predictor.is_ready() is False
        ), "is_ready() should return False when no .pkl model file exists."


class TestPredictorEdgeCases:
    def test_studio_apartment(self, demo_predictor: Predictor):
        """rooms=0 (studio) should still produce a valid positive price."""
        features = {**_SAMPLE_FEATURES, "rooms": 0}
        result = demo_predictor.predict(features)
        assert result["price"] > 0

    def test_first_floor_cheaper_than_middle(self, demo_predictor: Predictor):
        """A first-floor flat should be <= a mid-floor flat in DEMO mode."""
        features_first = {**_SAMPLE_FEATURES, "floor": 1, "floors_total": 10}
        features_mid = {**_SAMPLE_FEATURES, "floor": 5, "floors_total": 10}
        price_first = demo_predictor.predict(features_first)["price"]
        price_mid = demo_predictor.predict(features_mid)["price"]
        # First floor has a -5% penalty in DEMO mode
        assert (
            price_first <= price_mid
        ), "First floor price should be <= mid-floor price in DEMO mode."

    def test_larger_area_higher_price(self, demo_predictor: Predictor):
        """Larger area should produce a higher price in DEMO mode."""
        features_small = {**_SAMPLE_FEATURES, "total_area": 30.0}
        features_large = {**_SAMPLE_FEATURES, "total_area": 120.0}
        price_small = demo_predictor.predict(features_small)["price"]
        price_large = demo_predictor.predict(features_large)["price"]
        assert price_large > price_small, "Larger area should produce a higher predicted price."

    def test_confidence_in_valid_range(self, demo_predictor: Predictor):
        """confidence key should be in [0, 1] when present."""
        result = demo_predictor.predict(_SAMPLE_FEATURES)
        if "confidence" in result:
            assert (
                0.0 <= result["confidence"] <= 1.0
            ), f"confidence={result['confidence']} is outside [0, 1]."

    def test_model_info_is_demo_when_no_model(self, demo_predictor: Predictor):
        """model_info['is_demo'] should be True in DEMO mode."""
        info = demo_predictor.model_info
        assert info["is_demo"] is True

    def test_unknown_city_uses_default(self, demo_predictor: Predictor):
        """An unknown city should fall back to the default price-per-sqm."""
        features = {**_SAMPLE_FEATURES, "city": "тьмутаракань"}
        result = demo_predictor.predict(features)
        assert result["price"] > 0, "Unknown city should still return a positive price."

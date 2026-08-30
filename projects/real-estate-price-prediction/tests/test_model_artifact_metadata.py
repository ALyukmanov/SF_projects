"""
Contract tests for model-artefact provenance metadata and prediction-interval
ordering.

Trains a tiny throwaway ``random_forest`` model (not xgboost — lighter/
faster and needs no extra dependency beyond scikit-learn, which is already a
hard dependency of the test suite) on the existing ``sample_cleaned_df``
fixture data via ``ModelTrainer.fit()`` + ``ModelTrainer.save()``, entirely
inside ``tmp_path``. Nothing is written to the real ``models/`` directory.

Covers what was NOT already exercised by ``tests/test_predictor.py``
(demo-mode price-range ordering only) or ``tests/test_api.py``
(input-validation only): artefact provenance fields surfacing through
``Predictor.model_info``, and price_min <= price <= price_max holding in
**model** mode (not just demo/heuristic mode).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.feature_engineering import FEATURE_SCHEMA_VERSION, FeatureEngineer  # noqa: E402
from src.inference.predictor import Predictor  # noqa: E402
from src.models.trainer import ModelTrainer  # noqa: E402

_DATASET_SHA_STUB = "deadbeef" * 8  # not a real hash — just a stable test stand-in


@pytest.fixture
def trained_artifact_dir(sample_cleaned_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Train + save a tiny random_forest artefact with full provenance metadata."""
    engineer = FeatureEngineer()
    X, y = engineer.prepare_for_training(sample_cleaned_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    trainer = ModelTrainer(
        model_type="random_forest",
        config={"params": {"n_estimators": 20, "max_depth": 4}},
    )
    trainer.fit(X_train, y_train, X_test, y_test, feature_names=list(X.columns))

    metadata = {
        "data_source": "synthetic_demo_generator",
        "is_synthetic": True,
        "dataset_sha256": _DATASET_SHA_STUB,
        "dataset_rows": len(sample_cleaned_df),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    model_dir = tmp_path / "models"
    trainer.save(str(model_dir), metadata=metadata, tag="synthetic")
    return model_dir


class TestArtifactProvenanceMetadata:
    """model artefact metadata fields present after training."""

    def test_metadata_fields_present_after_training(self, trained_artifact_dir: Path):
        predictor = Predictor(model_path=str(trained_artifact_dir))
        assert predictor.load() is True

        info = predictor.model_info
        assert info["data_source"] == "synthetic_demo_generator"
        assert info["is_synthetic"] is True
        assert info["dataset_sha256"] == _DATASET_SHA_STUB
        assert info["feature_schema_version"] == FEATURE_SCHEMA_VERSION
        assert info["dataset_rows"] == 20
        assert info["is_demo"] is False

    def test_prediction_interval_persisted_and_well_formed(self, trained_artifact_dir: Path):
        predictor = Predictor(model_path=str(trained_artifact_dir))
        predictor.load()

        interval = predictor.model_info["prediction_interval"]
        assert interval, "prediction_interval must be a non-empty dict after training."
        assert interval["method"] == "residual_quantile_holdout"
        assert "lower_offset" in interval
        assert "upper_offset" in interval
        assert interval["lower_offset"] <= interval["upper_offset"]
        assert 0.0 <= interval["observed_coverage"] <= 1.0

    def test_predictor_is_ready_true_with_loaded_model(self, trained_artifact_dir: Path):
        predictor = Predictor(model_path=str(trained_artifact_dir))
        predictor.load()
        assert predictor.is_ready() is True


class TestPredictionIntervalOrderingModelMode:
    """price_min <= price <= price_max must hold in ML mode too, not just DEMO."""

    _SAMPLE_FEATURES = {
        "rooms": 2,
        "total_area": 55.0,
        "floor": 4,
        "floors_total": 10,
        "city": "москва",
        "building_type": "monolith",
        "year_built": 2005,
    }

    def test_price_range_valid_in_model_mode(self, trained_artifact_dir: Path):
        predictor = Predictor(model_path=str(trained_artifact_dir))
        predictor.load()

        result = predictor.predict(self._SAMPLE_FEATURES)
        assert result["mode"] == "model", (
            "Expected mode='model' with a loaded artefact; got "
            f"'{result['mode']}' — model inference silently fell back to DEMO."
        )
        assert result["price"] > 0
        assert result["price_min"] <= result["price"] <= result["price_max"]
        assert result["interval_method"] == "residual_quantile_holdout"

    @pytest.mark.parametrize("city", ["москва", "самара", "неизвестный город", "  МОСКВА  "])
    def test_price_range_valid_across_known_and_unknown_cities(
        self, trained_artifact_dir: Path, city: str
    ):
        predictor = Predictor(model_path=str(trained_artifact_dir))
        predictor.load()

        features = {**self._SAMPLE_FEATURES, "city": city}
        result = predictor.predict(features)
        assert result["price"] > 0
        assert result["price_min"] <= result["price"] <= result["price_max"]

    def test_studio_in_model_mode_does_not_crash(self, trained_artifact_dir: Path):
        predictor = Predictor(model_path=str(trained_artifact_dir))
        predictor.load()

        features = {**self._SAMPLE_FEATURES, "rooms": 0}
        result = predictor.predict(features)
        assert result["price"] > 0
        assert result["price_min"] <= result["price"] <= result["price_max"]

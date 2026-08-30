"""Tests for scripts/promote_model.py's fail-closed validation gate.

Builds tiny throwaway artefacts with joblib inside tmp_path (never touching
the real models/ directory or current_model.json) and drives
validate_candidate() directly against each failure mode the promotion
mechanism is meant to catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LinearRegression

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.promote_model import (  # noqa: E402
    ValidationError,
    _expected_feature_names,
    validate_candidate,
)
from src.features.feature_engineering import FEATURE_SCHEMA_VERSION  # noqa: E402


def _valid_artefact(**overrides) -> dict:
    model = LinearRegression().fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])
    artefact = {
        "model": model,
        "model_type": "linear_regression",
        "trained_at": "2026-08-27T00:00:00",
        "feature_names": _expected_feature_names(),
        "metrics": {"mae": 1_000_000.0, "rmse": 1_500_000.0, "r2": 0.7},
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "is_synthetic": False,
        "data_source": "restate",
        "dataset_path": "data/processed/real_estate_engineered.csv",
    }
    artefact.update(overrides)
    return artefact


def _save(tmp_path: Path, artefact, name: str = "candidate.pkl") -> Path:
    path = tmp_path / name
    joblib.dump(artefact, path)
    return path


class TestValidCandidatePasses:
    def test_well_formed_candidate_passes(self, tmp_path):
        path = _save(tmp_path, _valid_artefact())
        empty_manifest = tmp_path / "no_such_current_model.json"
        artefact, warnings = validate_candidate(
            path, allow_synthetic_promotion=False, manifest_path=empty_manifest
        )
        assert artefact["model_type"] == "linear_regression"
        # No current_model.json manifest in this sandbox, so no comparison warnings.
        assert warnings == []


class TestMissingOrCorruptFile:
    def test_missing_file_fails_closed(self, tmp_path):
        with pytest.raises(ValidationError, match="does not exist"):
            validate_candidate(tmp_path / "nope.pkl", allow_synthetic_promotion=False)

    def test_non_pkl_extension_rejected(self, tmp_path):
        path = tmp_path / "candidate.txt"
        path.write_text("not a model")
        with pytest.raises(ValidationError, match=r"\.pkl"):
            validate_candidate(path, allow_synthetic_promotion=False)

    def test_corrupt_pkl_fails_closed(self, tmp_path):
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"not actually a pickle")
        with pytest.raises(ValidationError, match="failed to load"):
            validate_candidate(path, allow_synthetic_promotion=False)


class TestMissingMetadata:
    def test_missing_required_field_rejected(self, tmp_path):
        artefact = _valid_artefact()
        del artefact["data_source"]
        path = _save(tmp_path, artefact)
        with pytest.raises(ValidationError, match="missing required fields"):
            validate_candidate(path, allow_synthetic_promotion=False)

    def test_model_without_predict_rejected(self, tmp_path):
        artefact = _valid_artefact(model=object())
        path = _save(tmp_path, artefact)
        with pytest.raises(ValidationError, match="no usable 'model'"):
            validate_candidate(path, allow_synthetic_promotion=False)


class TestSchemaCompatibility:
    def test_wrong_feature_schema_version_rejected(self, tmp_path):
        artefact = _valid_artefact(feature_schema_version="1.0")
        path = _save(tmp_path, artefact)
        with pytest.raises(ValidationError, match="feature_schema_version"):
            validate_candidate(path, allow_synthetic_promotion=False)

    def test_mismatched_feature_names_rejected(self, tmp_path):
        artefact = _valid_artefact(feature_names=["totally", "different", "columns"])
        path = _save(tmp_path, artefact)
        with pytest.raises(ValidationError, match="feature_names does not match"):
            validate_candidate(path, allow_synthetic_promotion=False)

    def test_duplicate_feature_names_rejected(self, tmp_path):
        names = _expected_feature_names()
        artefact = _valid_artefact(feature_names=names + [names[0]])
        path = _save(tmp_path, artefact)
        with pytest.raises(ValidationError, match="duplicates"):
            validate_candidate(path, allow_synthetic_promotion=False)


class TestMetricSanity:
    def test_nan_metric_rejected(self, tmp_path):
        artefact = _valid_artefact(metrics={"mae": float("nan"), "rmse": 1.0, "r2": 0.5})
        path = _save(tmp_path, artefact)
        with pytest.raises(ValidationError, match="NaN"):
            validate_candidate(path, allow_synthetic_promotion=False)

    def test_negative_mae_rejected(self, tmp_path):
        artefact = _valid_artefact(metrics={"mae": -1.0, "rmse": 1.0, "r2": 0.5})
        path = _save(tmp_path, artefact)
        with pytest.raises(ValidationError, match="positive"):
            validate_candidate(path, allow_synthetic_promotion=False)

    def test_absurd_r2_rejected(self, tmp_path):
        artefact = _valid_artefact(metrics={"mae": 1.0, "rmse": 1.0, "r2": -100.0})
        path = _save(tmp_path, artefact)
        with pytest.raises(ValidationError, match="sane range"):
            validate_candidate(path, allow_synthetic_promotion=False)

    def test_missing_metric_key_rejected(self, tmp_path):
        artefact = _valid_artefact(metrics={"mae": 1.0, "rmse": 1.0})
        path = _save(tmp_path, artefact)
        with pytest.raises(ValidationError, match="missing required keys"):
            validate_candidate(path, allow_synthetic_promotion=False)


class TestProvenanceLeakage:
    def test_absolute_windows_path_rejected(self, tmp_path):
        artefact = _valid_artefact(dataset_path=r"C:\Users\someuser\project\data.csv")
        path = _save(tmp_path, artefact)
        with pytest.raises(ValidationError, match="absolute"):
            validate_candidate(path, allow_synthetic_promotion=False)

    def test_missing_dataset_path_is_a_warning_not_a_failure(self, tmp_path):
        artefact = _valid_artefact()
        del artefact["dataset_path"]
        path = _save(tmp_path, artefact)
        _, warnings = validate_candidate(path, allow_synthetic_promotion=False)
        assert any("dataset_path" in w for w in warnings)


class TestSyntheticPromotionGuard:
    def test_synthetic_candidate_blocked_without_flag(self, tmp_path):
        artefact = _valid_artefact(is_synthetic=True)
        path = _save(tmp_path, artefact)
        with pytest.raises(ValidationError, match="allow-synthetic-promotion"):
            validate_candidate(path, allow_synthetic_promotion=False)

    def test_synthetic_candidate_allowed_with_flag(self, tmp_path):
        artefact = _valid_artefact(is_synthetic=True)
        path = _save(tmp_path, artefact)
        validate_candidate(path, allow_synthetic_promotion=True)

"""
API contract tests not already covered by ``tests/test_api.py`` (which
focuses on ``POST /predict`` input-validation 422s).

Covers:
- ``GET /health`` and ``GET /model-info`` are well-formed when a model IS
  loaded (test_api.py only implicitly exercises whatever happens to be in
  the real ``models/`` directory at collection time; these tests instead
  deterministically load a freshly-trained throwaway model from tmp_path so
  the assertions don't depend on incidental repository state).
- Malformed/invalid ``POST /predict`` input still returns a clean 422, not a
  500 with leaked internals (a few cases not already in test_api.py).
- A regression guard against the CORS wildcard-origin + credentials
  misconfiguration (main.py:132-138).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split
from starlette.middleware.cors import CORSMiddleware

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import api.main as main_module  # noqa: E402
from src.features.feature_engineering import FeatureEngineer  # noqa: E402
from src.inference.predictor import Predictor  # noqa: E402
from src.models.trainer import ModelTrainer  # noqa: E402

_BASE_PAYLOAD = {
    "rooms": 2,
    "total_area": 55.0,
    "floor": 5,
    "floors_total": 16,
    "city": "Москва",
    "building_type": "монолитный",
    "year_built": 2010,
}


# ===========================================================================
# CORS regression guard (static config check via the app object)
# ===========================================================================


def test_cors_does_not_combine_wildcard_origin_with_credentials():
    """Regression guard: allow_origins=['*'] + allow_credentials=True is an
    invalid CORS combination that browsers reject outright — previously
    present in main.py, since fixed. Must not silently regress.
    """
    cors_entries = [m for m in main_module.app.user_middleware if m.cls is CORSMiddleware]
    assert cors_entries, "CORSMiddleware must be registered on the app."

    kwargs = cors_entries[0].kwargs
    allow_origins = kwargs.get("allow_origins", [])
    allow_credentials = kwargs.get("allow_credentials", False)

    if "*" in allow_origins:
        assert allow_credentials is False, (
            "allow_origins=['*'] combined with allow_credentials=True is an "
            "invalid CORS configuration."
        )
    # Regression guard specific to this app's current, intentional config:
    # an explicit local-origin allowlist with no wildcard and no credentials.
    assert "*" not in allow_origins
    assert allow_credentials is False


# ===========================================================================
# /health and /model-info with a model actually loaded
# ===========================================================================


@pytest.fixture
def client_with_loaded_model(sample_cleaned_df: pd.DataFrame, tmp_path: Path, monkeypatch):
    """A TestClient wired to a Predictor with a real (tiny, throwaway)
    trained artefact — deterministic, independent of whatever .pkl files
    happen to exist in the real models/ directory.
    """
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
        "dataset_sha256": "cafebabe" * 8,
        "dataset_rows": len(sample_cleaned_df),
    }
    model_dir = tmp_path / "models"
    trainer.save(str(model_dir), metadata=metadata, tag="synthetic")

    test_predictor = Predictor(model_path=str(model_dir))
    assert test_predictor.load() is True
    monkeypatch.setattr(main_module, "_predictor", test_predictor)

    with TestClient(main_module.app) as c:
        yield c


class TestHealthWithLoadedModel:
    def test_health_reports_model_loaded_true(self, client_with_loaded_model: TestClient):
        resp = client_with_loaded_model.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_info"]["is_demo"] is False
        assert body["model_info"]["model_type"] == "random_forest"
        assert "version" in body

    def test_predict_uses_model_mode_when_loaded(self, client_with_loaded_model: TestClient):
        resp = client_with_loaded_model.post("/predict", json=_BASE_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "model"
        assert body["price_min"] <= body["price"] <= body["price_max"]


class TestModelInfoWithLoadedModel:
    def test_model_info_returns_full_provenance(self, client_with_loaded_model: TestClient):
        resp = client_with_loaded_model.get("/model-info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data_source"] == "synthetic_demo_generator"
        assert body["is_synthetic"] is True
        assert body["dataset_sha256"] == "cafebabe" * 8
        assert body["dataset_rows"] == 20
        assert "feature_schema_version" in body
        assert body["prediction_interval"]["method"] == "residual_quantile_holdout"


# ===========================================================================
# Additional malformed-input 422 cases (test_api.py already covers most
# floor-related cases; these cover other fields and structurally-wrong
# payloads).
# ===========================================================================


class TestMalformedInputReturnsClean422:
    @pytest.fixture(scope="class")
    def client(self):
        with TestClient(main_module.app) as c:
            yield c

    def test_negative_total_area_is_rejected(self, client: TestClient):
        resp = client.post("/predict", json={**_BASE_PAYLOAD, "total_area": -10.0})
        assert resp.status_code == 422
        assert "Traceback" not in resp.text

    def test_total_area_above_max_is_rejected(self, client: TestClient):
        resp = client.post("/predict", json={**_BASE_PAYLOAD, "total_area": 10_000.0})
        assert resp.status_code == 422

    def test_rooms_above_max_is_rejected(self, client: TestClient):
        resp = client.post("/predict", json={**_BASE_PAYLOAD, "rooms": 99})
        assert resp.status_code == 422

    def test_city_as_wrong_type_is_rejected(self, client: TestClient):
        resp = client.post("/predict", json={**_BASE_PAYLOAD, "city": ["Москва"]})
        assert resp.status_code == 422

    def test_year_built_out_of_range_is_rejected(self, client: TestClient):
        resp = client.post("/predict", json={**_BASE_PAYLOAD, "year_built": 1500})
        assert resp.status_code == 422

    def test_empty_body_is_rejected(self, client: TestClient):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_completely_wrong_shape_body_is_rejected_not_500(self, client: TestClient):
        resp = client.post("/predict", json={"unexpected": "shape", "nested": {"a": 1}})
        assert resp.status_code == 422
        body = resp.json()
        # Must be a structured FastAPI/pydantic validation error, not a raw
        # unhandled-exception leak.
        assert "detail" in body
        assert "Traceback" not in resp.text
        assert "site-packages" not in resp.text

    def test_error_response_does_not_leak_internal_paths(self, client: TestClient):
        resp = client.post("/predict", json={**_BASE_PAYLOAD, "floor": 0})
        assert resp.status_code == 422
        assert "C:\\" not in resp.text
        assert "/home/" not in resp.text

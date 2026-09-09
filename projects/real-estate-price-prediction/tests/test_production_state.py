"""Guards the CURRENT production state (models/current_model.json + the
artefact it points at) against an accidental regression back to the
synthetic model or a broken/missing artefact reference.

This is intentionally about *whatever is currently promoted*, not a specific
model type -- the production model is real (restate.ru-trained xgboost) as
of this commit, and scripts/promote_model.py is the only sanctioned way to
change it.
These tests read the real models/current_model.json directly; they never
write to it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MODELS_DIR = _PROJECT_ROOT / "models"
_MANIFEST_PATH = _MODELS_DIR / "current_model.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not _MANIFEST_PATH.is_file():
        pytest.skip(f"{_MANIFEST_PATH} not present in this checkout -- no production model set.")
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


class TestProductionManifestValid:
    def test_manifest_is_valid_json_with_required_keys(self, manifest: dict):
        for key in (
            "filename",
            "model_type",
            "is_synthetic",
            "data_source",
            "feature_schema_version",
        ):
            assert key in manifest, f"current_model.json is missing required key '{key}'"

    def test_manifest_artifact_exists_on_disk(self, manifest: dict):
        """The single most important release gate: current_model.json must
        never point at a file that doesn't exist -- this is a hard blocker
        in the promotion gate. A clean clone with no local training history
        must still be able to resolve this."""
        artifact_path = _MODELS_DIR / manifest["filename"]
        assert artifact_path.is_file(), (
            f"current_model.json references '{manifest['filename']}', which does not exist "
            f"under {_MODELS_DIR} -- this is a release blocker, not a warning."
        )

    def test_manifest_artifact_hash_matches_recorded_fingerprint(self, manifest: dict):
        import hashlib

        artifact_path = _MODELS_DIR / manifest["filename"]
        if "artifact_sha256" not in manifest:
            pytest.skip("Older manifest without a recorded artifact_sha256.")
        actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert actual == manifest["artifact_sha256"]


class TestProductionIsRealNotSynthetic:
    """As of this commit, production is deliberately the real restate.ru
    candidate -- these tests fail loudly if that ever
    silently regresses back to the synthetic demo model. If a human
    deliberately re-promotes a synthetic artefact (e.g. rolling back), these
    tests should be updated in the SAME commit as that decision, not left to
    fail unexplained -- see scripts/promote_model.py's --allow-synthetic-
    promotion gate, which already requires an explicit, deliberate flag for
    exactly that scenario.
    """

    def test_is_synthetic_is_false(self, manifest: dict):
        assert manifest["is_synthetic"] is False

    def test_data_source_is_restate(self, manifest: dict):
        assert "restate" in str(manifest.get("data_source", "")).lower()

    def test_feature_schema_version_matches_current_code(self, manifest: dict):
        from src.features.feature_engineering import FEATURE_SCHEMA_VERSION

        assert manifest["feature_schema_version"] == FEATURE_SCHEMA_VERSION

    def test_metrics_present_and_finite(self, manifest: dict):
        metrics = manifest.get("metrics", {})
        for key in ("mae", "rmse", "r2"):
            assert key in metrics, f"metrics missing '{key}'"
            assert metrics[key] == metrics[key], f"metrics['{key}'] is NaN"  # NaN != NaN


class TestProductionModelLoads:
    def test_predictor_loads_the_production_artifact(self, manifest: dict):
        from src.inference.predictor import Predictor

        predictor = Predictor(model_path=str(_MODELS_DIR))
        loaded = predictor.load()
        assert loaded is True
        assert predictor.is_ready() is True
        info = predictor.model_info
        assert info["is_demo"] is False
        assert info["is_synthetic"] is False

    def test_production_artifact_unpickles_with_expected_keys(self, manifest: dict):
        artifact_path = _MODELS_DIR / manifest["filename"]
        # Locally-produced artefact from this repo's own training pipeline
        # (same trust boundary as src/models/trainer.py) -- not untrusted input.
        artifact = joblib.load(artifact_path)
        for key in ("model", "scaler", "feature_names", "metrics"):
            assert key in artifact, f"production artifact is missing '{key}'"
        assert len(artifact["feature_names"]) > 0

    def test_predictor_produces_a_real_prediction(self, manifest: dict):
        from src.inference.predictor import Predictor

        predictor = Predictor(model_path=str(_MODELS_DIR))
        assert predictor.load() is True
        result = predictor.predict(
            {"rooms": 2, "total_area": 55.0, "floor": 5, "floors_total": 16, "city": "Москва"}
        )
        assert result["mode"] == "model"
        assert result["price"] > 0


class TestProductionGeoModel:
    """The promoted production model is the geo model — guard its shape."""

    def test_manifest_is_the_location_grouped_geo_model(self, manifest: dict):
        assert manifest["split_strategy"] == "location_grouped_80_20_random_state_42"

    def test_artifact_has_45_features_including_geo(self, manifest: dict):
        from src.features.geo_features import GEO_FEATURE_COLUMNS

        artifact = joblib.load(_MODELS_DIR / manifest["filename"])
        names = artifact["feature_names"]
        assert len(names) == 45
        for col in GEO_FEATURE_COLUMNS:
            assert col in names
        assert artifact.get("geo_enabled") is True
        assert (artifact.get("imputer") or {}).get("fitted") is True

    def test_predictor_reports_geo_enabled_and_poi_available(self, manifest: dict):
        from src.inference.predictor import Predictor

        p = Predictor(model_path=str(_MODELS_DIR))
        assert p.load() is True
        info = p.model_info
        assert info["geo_enabled"] is True
        # osm_poi.csv is expected to be present in a working checkout
        assert info["geo_poi_available"] is True
        assert p._geo_unavailable is False

    def test_previous_artifact_is_kept_for_rollback(self, manifest: dict):
        prev = manifest.get("previous_current_filename")
        assert prev, "manifest must record previous_current_filename for rollback"
        assert (_MODELS_DIR / prev).is_file(), f"rollback target {prev} missing from models/"

    def test_prediction_uses_coordinates_when_given(self, manifest: dict):
        from src.inference.predictor import Predictor

        p = Predictor(model_path=str(_MODELS_DIR))
        assert p.load() is True
        base = {"rooms": 2, "total_area": 55.0, "floor": 5, "floors_total": 16, "city": "Москва"}
        near_centre = p.predict({**base, "latitude": 55.7558, "longitude": 37.6173})["price"]
        far_edge = p.predict({**base, "latitude": 55.55, "longitude": 37.35})["price"]
        # geo features are actually consumed -> two locations give different prices
        assert near_centre != far_edge

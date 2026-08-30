"""Live-verifies the FastAPI app against the real (non-synthetic) trained
candidate WITHOUT touching models/current_model.json -- switching the
production model is a separate, explicit step (see scripts/promote_model.py)
and must never be a side effect of running this test.

Mechanism (matches the project's existing artefact-selection contract,
Predictor.load()'s docstring): a real .pkl is copied into an ISOLATED
tmp_path directory together with its own current_model.json manifest, and
api.main._predictor (a plain module-level global, not a FastAPI Depends())
is monkeypatched to a fresh Predictor(model_path=<that isolated dir>) before
the TestClient's lifespan startup runs. The real models/ directory and its
current_model.json are never opened for writing anywhere in this file.

This proves the full real path: artefact -> Predictor.load() -> FastAPI
route handlers -> JSON response, with is_synthetic=False, real metrics, and
real data_source -- while production keeps serving the synthetic model.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MODELS_DIR = _PROJECT_ROOT / "models"

# Captured at module import time (before any fixture in this file runs) so
# TestExactArtifactLoaded.test_production_current_model_json_untouched can
# prove the real manifest is byte-identical after this file's isolated
# fixtures run, whatever it currently says -- see that test's docstring.
_PRODUCTION_MANIFEST_SHA256_AT_COLLECTION = hashlib.sha256(
    (_MODELS_DIR / "current_model.json").read_bytes()
).hexdigest()


def _find_real_group_aware_xgboost_candidate() -> Path:
    """Locate the current real (is_synthetic=False), group-aware-split
    xgboost candidate in models/ by inspecting artefact metadata directly,
    rather than hardcoding a timestamped filename that will go stale.
    """
    found = []
    for pkl in sorted(_MODELS_DIR.glob("*.pkl")):
        try:
            # Locally-produced artefacts from this repo's own training pipeline
            # (same trust boundary as src/models/trainer.py) -- not untrusted input.
            artefact = joblib.load(pkl)
        except Exception:
            continue
        if (
            isinstance(artefact, dict)
            and artefact.get("model_type") == "xgboost"
            and artefact.get("is_synthetic") is False
            and artefact.get("split_strategy") == "group_aware_80_20_random_state_42"
        ):
            found.append((artefact.get("trained_at") or "", pkl))
    if not found:
        pytest.skip(
            "No real group-aware xgboost candidate found in models/. Run "
            "scripts/run_model_training_real.py --model xgboost --split-strategy group "
            "--params-from-study xgboost first."
        )
    found.sort(key=lambda t: t[0])
    return found[-1][1]


@pytest.fixture(scope="module")
def real_candidate_path() -> Path:
    return _find_real_group_aware_xgboost_candidate()


@pytest.fixture(scope="module")
def real_candidate_artifact(real_candidate_path: Path) -> dict:
    return joblib.load(real_candidate_path)


@pytest.fixture(scope="module")
def isolated_model_dir(tmp_path_factory, real_candidate_path: Path) -> Path:
    """A throwaway directory holding a COPY of the real candidate plus its
    own current_model.json -- never the real models/current_model.json.
    """
    tmp_dir = tmp_path_factory.mktemp("real_candidate_models")
    dest = tmp_dir / real_candidate_path.name
    shutil.copy2(real_candidate_path, dest)

    artefact = joblib.load(dest)
    manifest = {
        "filename": dest.name,
        "model_type": artefact.get("model_type"),
        "trained_at": artefact.get("trained_at"),
        "is_synthetic": artefact.get("is_synthetic"),
        "data_source": artefact.get("data_source"),
        "feature_schema_version": artefact.get("feature_schema_version"),
        "metrics": artefact.get("metrics"),
        "artifact_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
    }
    (tmp_dir / "current_model.json").write_text(
        json.dumps(manifest, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return tmp_dir


@pytest.fixture()
def real_candidate_client(isolated_model_dir: Path, monkeypatch):
    """A TestClient serving api.main.app with _predictor swapped to load
    ONLY from isolated_model_dir. Real models/current_model.json is never
    touched by this fixture.
    """
    import api.main as main_module
    from src.inference.predictor import Predictor

    test_predictor = Predictor(model_path=str(isolated_model_dir))
    monkeypatch.setattr(main_module, "_predictor", test_predictor)

    with TestClient(main_module.app) as client:
        yield client, test_predictor


class TestExactArtifactLoaded:
    """Proves WHICH .pkl was actually loaded -- path + hash + size."""

    def test_isolated_manifest_hash_matches_source_file_exactly(
        self, real_candidate_path: Path, isolated_model_dir: Path
    ):
        expected_hash = hashlib.sha256(real_candidate_path.read_bytes()).hexdigest()
        expected_size = real_candidate_path.stat().st_size
        manifest = json.loads(
            (isolated_model_dir / "current_model.json").read_text(encoding="utf-8")
        )
        copied = isolated_model_dir / manifest["filename"]

        assert manifest["filename"] == real_candidate_path.name
        assert manifest["artifact_sha256"] == expected_hash
        assert copied.stat().st_size == expected_size

    def test_production_current_model_json_untouched(self, real_candidate_client):
        """The real models/current_model.json must be byte-identical before
        and after this file's isolated-candidate fixtures run -- proves the
        isolation mechanism (a copy in a throwaway tmp_path dir, see the
        module docstring) never leaks a write into the real production
        manifest, regardless of which model happens to be promoted at the
        time these tests run (see tests/test_production_state.py for
        assertions about what the real manifest *should* currently say).
        """
        before_hash = _PRODUCTION_MANIFEST_SHA256_AT_COLLECTION
        after_hash = hashlib.sha256((_MODELS_DIR / "current_model.json").read_bytes()).hexdigest()
        assert after_hash == before_hash


class TestHealthAndModelInfoReportRealProvenanceHonestly:
    def test_health_reports_real_model_loaded(self, real_candidate_client, real_candidate_artifact):
        client, _ = real_candidate_client
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_loaded"] is True
        info = body["model_info"]
        assert info["is_synthetic"] is False
        assert info["is_demo"] is False
        assert info["model_type"] == "xgboost"
        assert info["data_source"] == real_candidate_artifact["data_source"]
        assert "restate" in info["data_source"].lower()

    def test_model_info_metrics_match_real_candidate_exactly(
        self, real_candidate_client, real_candidate_artifact
    ):
        client, _ = real_candidate_client
        body = client.get("/model-info").json()
        expected = real_candidate_artifact["metrics"]
        assert body["metrics"]["mae"] == pytest.approx(expected["mae"])
        assert body["metrics"]["rmse"] == pytest.approx(expected["rmse"])
        assert body["metrics"]["r2"] == pytest.approx(expected["r2"])
        assert body["feature_schema_version"] == real_candidate_artifact["feature_schema_version"]

    def test_no_silent_fallback_to_synthetic_fields(self, real_candidate_client):
        client, _ = real_candidate_client
        body = client.get("/model-info").json()
        assert body["is_synthetic"] is False
        assert body["data_source"] != "synthetic_demo_generator"


class TestPredictWithRealCandidate:
    def _predict(self, client, **overrides):
        payload = {
            "rooms": 2,
            "total_area": 55.0,
            "floor": 4,
            "floors_total": 10,
            "city": "Москва",
            "building_type": "монолитный",
        }
        payload.update(overrides)
        return client.post("/predict", json=payload)

    def test_moscow_prediction_uses_real_model_mode(self, real_candidate_client):
        client, _ = real_candidate_client
        resp = self._predict(client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "model"
        assert body["price"] > 0
        assert body["price_min"] <= body["price"] <= body["price_max"]

    def test_saint_petersburg_prediction(self, real_candidate_client):
        client, _ = real_candidate_client
        resp = self._predict(
            client, city="Санкт-Петербург", rooms=3, total_area=80.0, floor=2, floors_total=5
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "model"
        assert body["price"] > 0

    def test_missing_optional_building_type(self, real_candidate_client):
        client, _ = real_candidate_client
        payload = {
            "rooms": 1,
            "total_area": 38.0,
            "floor": 3,
            "floors_total": 9,
            "city": "Казань",
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["mode"] == "model"

    def test_studio_rooms_zero(self, real_candidate_client):
        client, _ = real_candidate_client
        resp = self._predict(client, rooms=0, total_area=22.0, floor=1, floors_total=5)
        assert resp.status_code == 200
        assert resp.json()["price"] > 0

    def test_near_max_boundary_object(self, real_candidate_client):
        client, _ = real_candidate_client
        resp = self._predict(client, rooms=10, total_area=499.0, floor=1, floors_total=1)
        assert resp.status_code == 200
        assert resp.json()["price"] > 0

    def test_invalid_floor_exceeds_total_still_422(self, real_candidate_client):
        client, _ = real_candidate_client
        resp = self._predict(client, floor=12, floors_total=10)
        assert resp.status_code == 422

    def test_invalid_negative_area_still_422(self, real_candidate_client):
        client, _ = real_candidate_client
        resp = client.post(
            "/predict",
            json={
                "rooms": 2,
                "total_area": -5.0,
                "floor": 4,
                "floors_total": 10,
                "city": "Москва",
            },
        )
        assert resp.status_code == 422

    def test_unknown_city_does_not_crash(self, real_candidate_client):
        client, _ = real_candidate_client
        resp = self._predict(client, city="Атлантида")
        assert resp.status_code == 200
        assert resp.json()["mode"] == "model"

    def test_russian_formatted_price_is_clean_utf8(self, real_candidate_client):
        client, _ = real_candidate_client
        resp = self._predict(client)
        formatted = resp.json()["formatted_price"]
        assert isinstance(formatted, str)
        # Round-trips cleanly and contains a recognisable Russian unit word.
        assert formatted.encode("utf-8").decode("utf-8") == formatted
        assert ("млн" in formatted) or ("руб" in formatted) or ("тыс" in formatted)


def _direct_model_price(raw_features: dict, real_candidate_artifact: dict) -> float:
    """Predict via the candidate artefact directly (joblib model + the
    OFFLINE FeatureEngineer pipeline), bypassing Predictor/API entirely --
    the independent reference path for inference-parity comparison. Assumes
    *raw_features* has no missing rooms/total_area/floor/floors_total (use
    ``_direct_model_price_via_imputer`` below for that case).
    """
    from src.features.feature_engineering import FeatureEngineer

    df = pd.DataFrame([{**raw_features, "price": 1.0}])  # dummy target, only need X
    engineer = FeatureEngineer()
    X, _ = engineer.prepare_for_training(df)

    feature_list = real_candidate_artifact["feature_names"]
    X = X.reindex(columns=feature_list, fill_value=0)

    model = real_candidate_artifact["model"]
    scaler = real_candidate_artifact.get("scaler")
    X_in = scaler.transform(X) if scaler is not None else X.values

    log_price = float(model.predict(X_in)[0])
    return float(np.expm1(log_price))


def _direct_model_price_via_imputer(
    raw_features_with_nan: dict, real_candidate_artifact: dict
) -> float:
    """Same reference path as ``_direct_model_price``, but first fills any
    missing rooms/total_area/floor/floors_total using the candidate
    artefact's own persisted train-fit ``GroupMedianImputer`` -- the offline
    equivalent of what ``Predictor._default_for`` now does for a live
    request that omits one of those fields.
    """
    from src.features.feature_engineering import FeatureEngineer
    from src.preprocessing.imputer import GroupMedianImputer

    imputer = GroupMedianImputer.from_dict(real_candidate_artifact.get("imputer"))
    df = pd.DataFrame([{**raw_features_with_nan, "price": 1.0}])
    df = imputer.transform(df)

    engineer = FeatureEngineer()
    X, _ = engineer.prepare_for_training(df)
    feature_list = real_candidate_artifact["feature_names"]
    X = X.reindex(columns=feature_list, fill_value=0)

    model = real_candidate_artifact["model"]
    scaler = real_candidate_artifact.get("scaler")
    X_in = scaler.transform(X) if scaler is not None else X.values
    log_price = float(model.predict(X_in)[0])
    return float(np.expm1(log_price))


def _cleaned_csv_test_split():
    """Return (test_raw, imputer) from the SAME leakage-safe split the real
    candidate was actually trained on (group-aware, seed=42) -- used to pull
    real holdout rows, including ones with genuinely missing rooms/floor/
    floors_total (restate.ru does not publish these fields for some listing
    categories -- not a scraping gap). Skips if the cleaned CSV isn't
    present locally (kept small/gitignored).
    """
    csv_path = _PROJECT_ROOT / "data" / "processed" / "real_estate_cleaned.csv"
    if not csv_path.exists():
        pytest.skip(f"{csv_path} not present locally -- cannot pull a real holdout row.")

    from src.data.split_pipeline import split_impute_featurize

    df = pd.read_csv(csv_path)
    split_result = split_impute_featurize(df, split_strategy="group", random_state=42)
    return split_result.test_raw


def _row_to_raw_dict(row: pd.Series, omit: tuple = ()) -> dict:
    raw = {
        "rooms": None if pd.isna(row["rooms"]) else int(row["rooms"]),
        "total_area": float(row["total_area"]),
        "floor": None if pd.isna(row["floor"]) else int(row["floor"]),
        "floors_total": None if pd.isna(row["floors_total"]) else int(row["floors_total"]),
        "city": str(row["city"]),
        "building_type": str(row["building_type"]) if pd.notna(row.get("building_type")) else None,
    }
    for key in omit:
        raw[key] = None
    return raw


def _real_holdout_row_features() -> dict:
    """One real listing's raw features from the group-aware TEST split,
    with a complete (non-missing) rooms/floor/floors_total -- the "typical
    object from the holdout real dataset" case.
    """
    test_raw = _cleaned_csv_test_split()
    complete = test_raw.dropna(subset=["rooms", "floor", "floors_total", "total_area"])
    if complete.empty:
        pytest.skip("No complete (non-missing) row found in the test split.")
    return _row_to_raw_dict(complete.iloc[0])


class TestDirectModelVsApiInferenceParity:
    """Compares the candidate model's prediction computed two independent
    ways for a fixed regression fixture of listings: direct joblib-model +
    offline FeatureEngineer inference, vs the live API (TestClient) path.
    A mismatch here would mean a listing gets a different predicted price
    depending on which code path evaluates it -- exactly what task #17's
    feature-vector parity test guards against, checked here end-to-end at
    the price level instead of the raw feature-vector level.
    """

    _CASES = {
        "moscow_flat": {
            "rooms": 2,
            "total_area": 55.0,
            "floor": 4,
            "floors_total": 10,
            "city": "Москва",
            "building_type": "monolith",
        },
        "spb_flat": {
            "rooms": 3,
            "total_area": 80.0,
            "floor": 2,
            "floors_total": 5,
            "city": "Санкт-Петербург",
            "building_type": "panel",
        },
        "no_building_type": {
            "rooms": 1,
            "total_area": 38.0,
            "floor": 3,
            "floors_total": 9,
            "city": "Казань",
            "building_type": None,
        },
        "studio_rooms_zero": {
            "rooms": 0,
            "total_area": 22.0,
            "floor": 1,
            "floors_total": 5,
            "city": "Москва",
            "building_type": "panel",
        },
        "near_max_boundary": {
            "rooms": 10,
            "total_area": 499.0,
            "floor": 1,
            "floors_total": 1,
            "city": "Санкт-Петербург",
            "building_type": "brick",
        },
    }

    @pytest.mark.parametrize("case_name", list(_CASES.keys()))
    def test_fixed_case_parity(self, case_name, real_candidate_client, real_candidate_artifact):
        client, _ = real_candidate_client
        raw = self._CASES[case_name]

        direct_price = _direct_model_price(raw, real_candidate_artifact)

        api_payload = {k: v for k, v in raw.items() if v is not None}
        resp = client.post("/predict", json=api_payload)
        assert resp.status_code == 200
        api_price = resp.json()["price"]

        assert api_price == pytest.approx(
            direct_price, rel=1e-4, abs=2.0
        ), f"[{case_name}] direct={direct_price!r} vs api={api_price!r}"

    def test_real_holdout_row_parity(self, real_candidate_client, real_candidate_artifact):
        raw = _real_holdout_row_features()
        client, _ = real_candidate_client

        direct_price = _direct_model_price(raw, real_candidate_artifact)

        api_payload = {k: v for k, v in raw.items() if v is not None}
        resp = client.post("/predict", json=api_payload)
        assert resp.status_code == 200
        api_price = resp.json()["price"]

        assert api_price == pytest.approx(
            direct_price, rel=1e-4, abs=2.0
        ), f"[real_holdout_row] raw={raw} direct={direct_price!r} vs api={api_price!r}"


class TestMissingValueInferenceParity:
    """rooms/floor/floors_total can be genuinely missing in the raw
    scraped data. The FastAPI request schema requires these fields (a real HTTP client can
    never omit them — omitting any of them is a 422, covered in
    tests/test_api.py), so this exercises the lower-level
    ``Predictor.predict()`` Python path directly with a field genuinely
    absent from the dict, which is exactly what
    ``Predictor._build_feature_vector``'s ``_default_for()`` fallback (fed
    by the artefact's persisted train-fit imputer) exists to handle. Compares
    against the offline reference path using the SAME persisted imputer.
    """

    def _predictor(self, isolated_model_dir: Path):
        from src.inference.predictor import Predictor

        p = Predictor(model_path=str(isolated_model_dir))
        assert p.load() is True
        return p

    def test_missing_rooms_uses_train_fitted_median_not_hardcoded_default(
        self, isolated_model_dir: Path, real_candidate_artifact: dict
    ):
        test_raw = _cleaned_csv_test_split()
        candidates = test_raw[test_raw["rooms"].isna()]
        if candidates.empty:
            pytest.skip("No test-split row with missing rooms found.")
        raw = _row_to_raw_dict(candidates.iloc[0])
        raw_missing_rooms = {k: v for k, v in raw.items() if k != "rooms" and v is not None}

        direct_price = _direct_model_price_via_imputer(
            {**raw, "rooms": None}, real_candidate_artifact
        )
        predictor_price = self._predictor(isolated_model_dir).predict(raw_missing_rooms)["price"]

        assert predictor_price == pytest.approx(
            direct_price, rel=1e-4, abs=2.0
        ), f"missing rooms: direct={direct_price!r} vs predictor={predictor_price!r}"

    def test_missing_floor_and_floors_total_uses_train_fitted_median(
        self, isolated_model_dir: Path, real_candidate_artifact: dict
    ):
        test_raw = _cleaned_csv_test_split()
        candidates = test_raw[test_raw["floor"].isna() & test_raw["floors_total"].isna()]
        if candidates.empty:
            pytest.skip("No test-split row with missing floor/floors_total found.")
        raw = _row_to_raw_dict(candidates.iloc[0])
        raw_missing_floor = {
            k: v for k, v in raw.items() if k not in ("floor", "floors_total") and v is not None
        }

        direct_price = _direct_model_price_via_imputer(
            {**raw, "floor": None, "floors_total": None}, real_candidate_artifact
        )
        predictor_price = self._predictor(isolated_model_dir).predict(raw_missing_floor)["price"]

        assert predictor_price == pytest.approx(
            direct_price, rel=1e-4, abs=2.0
        ), f"missing floor/floors_total: direct={direct_price!r} vs predictor={predictor_price!r}"

    def test_missing_rooms_fill_value_equals_persisted_imputer_median(
        self, isolated_model_dir: Path, real_candidate_artifact: dict
    ):
        """Directly proves the fallback used is the artefact's train-fit
        median, not the old hardcoded literal (2)."""
        from src.preprocessing.imputer import GroupMedianImputer

        imputer = GroupMedianImputer.from_dict(real_candidate_artifact.get("imputer"))
        assert imputer.fitted_, "Real candidate artefact must carry a fitted imputer."

        predictor = self._predictor(isolated_model_dir)
        raw_no_rooms = {
            "total_area": 55.0,
            "floor": 4,
            "floors_total": 10,
            "city": "Москва",
        }
        # Directly inspect the feature vector's 'rooms' value (bypassing the
        # model itself) via the same internal method the API path uses.
        vector = predictor._build_feature_vector(raw_no_rooms)
        assert vector["rooms"] == pytest.approx(imputer.global_median_["rooms"])

    def test_missing_rooms_fill_value_tracks_an_arbitrary_imputer_value(
        self, isolated_model_dir: Path
    ):
        """Discriminating variant of the test above: the real candidate's
        actual train-fit median for rooms happens to be 2.0, which coincides
        with the OLD hardcoded literal this replaced -- so that test alone
        cannot tell the two implementations apart. Here the loaded
        predictor's imputer is swapped for one with a deliberately
        unmistakable median (7 -- integer, since GroupMedianImputer.transform()
        rounds rooms to the nearest int, and nowhere near 2), and the
        feature vector must reflect exactly that value -- proof the
        fallback genuinely reads from ``self._imputer``, not a hardcoded
        constant that happens to match by coincidence.
        """
        from src.preprocessing.imputer import GroupMedianImputer

        predictor = self._predictor(isolated_model_dir)
        fake_imputer = GroupMedianImputer()
        fake_imputer.fitted_ = True
        fake_imputer.global_median_ = {
            "rooms": 7.0,
            "total_area": 55.0,
            "floor": 4.0,
            "floors_total": 10.0,
        }
        fake_imputer.city_median_ = {}
        fake_imputer.city_rooms_median_total_area_ = {}
        predictor._imputer = fake_imputer

        raw_no_rooms = {"total_area": 55.0, "floor": 4, "floors_total": 10, "city": "Москва"}
        vector = predictor._build_feature_vector(raw_no_rooms)
        assert vector["rooms"] == pytest.approx(7.0)

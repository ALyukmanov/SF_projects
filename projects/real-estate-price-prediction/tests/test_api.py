"""
API-level regression tests for POST /predict input validation.

These tests exercise the FastAPI layer directly (schema + endpoint validation),
independent of whether a trained model artefact is present. They do not assert
exact predicted prices, since a real .pkl model is git-ignored and may be
absent in a fresh checkout (the API falls back to DEMO mode in that case).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


_BASE_PAYLOAD = {
    "rooms": 2,
    "total_area": 55.0,
    "floor": 5,
    "floors_total": 16,
    "city": "Москва",
    "building_type": "монолитный",
    "year_built": 2010,
}


def test_valid_request_returns_200(client: TestClient) -> None:
    resp = client.post("/predict", json=_BASE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["price"] > 0
    assert body["mode"] in ("model", "demo")


def test_floor_zero_is_rejected(client: TestClient) -> None:
    resp = client.post("/predict", json={**_BASE_PAYLOAD, "floor": 0})
    assert resp.status_code == 422
    assert "floor" in resp.text.lower()


def test_negative_floor_is_rejected(client: TestClient) -> None:
    resp = client.post("/predict", json={**_BASE_PAYLOAD, "floor": -3})
    assert resp.status_code == 422


def test_floor_above_total_floors_is_rejected(client: TestClient) -> None:
    resp = client.post("/predict", json={**_BASE_PAYLOAD, "floor": 20, "floors_total": 16})
    assert resp.status_code == 422
    assert "cannot exceed" in resp.text


def test_missing_floor_is_rejected(client: TestClient) -> None:
    payload = {k: v for k, v in _BASE_PAYLOAD.items() if k != "floor"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_non_numeric_floor_is_rejected(client: TestClient) -> None:
    resp = client.post("/predict", json={**_BASE_PAYLOAD, "floor": "ground"})
    assert resp.status_code == 422


def test_boundary_floor_equals_total_floors_is_accepted(client: TestClient) -> None:
    resp = client.post("/predict", json={**_BASE_PAYLOAD, "floor": 16, "floors_total": 16})
    assert resp.status_code == 200


def test_floor_zero_does_not_reach_model(client: TestClient, caplog) -> None:
    """A rejected request must short-circuit before the predictor is invoked."""
    caplog.clear()
    resp = client.post("/predict", json={**_BASE_PAYLOAD, "floor": 0})
    assert resp.status_code == 422
    assert "Prediction result" not in caplog.text


# ---------------------------------------------------------------------------
# property_category / prediction_reliability / segment_support
# ---------------------------------------------------------------------------


def test_predict_without_property_category_is_standard_reliability(client: TestClient) -> None:
    resp = client.post("/predict", json=_BASE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction_reliability"] == "standard"
    assert body["segment_support"] is None


def test_predict_with_ordinary_category_is_standard_reliability(client: TestClient) -> None:
    resp = client.post(
        "/predict", json={**_BASE_PAYLOAD, "property_category": "2_rooms_flats_sale"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction_reliability"] == "standard"
    assert body["segment_support"] is None


@pytest.mark.parametrize("category", ["cottages_sale", "room_sale"])
def test_predict_with_low_support_category_is_flagged(client: TestClient, category: str) -> None:
    resp = client.post("/predict", json={**_BASE_PAYLOAD, "property_category": category})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction_reliability"] == "limited_data"
    assert body["segment_support"]["category"] == category
    assert body["segment_support"]["n_holdout"] > 0
    assert "reason" in body["segment_support"]


def test_predict_with_unknown_category_is_not_rejected(client: TestClient) -> None:
    """property_category is free text (like city/building_type), not a
    strict enum — an unrecognised value must not 422, and must fall back to
    standard reliability rather than crash."""
    resp = client.post(
        "/predict", json={**_BASE_PAYLOAD, "property_category": "totally_unknown_category"}
    )
    assert resp.status_code == 200
    assert resp.json()["prediction_reliability"] == "standard"


def test_low_support_category_does_not_change_the_predicted_price(client: TestClient) -> None:
    """property_category must be honesty-metadata only — the model has no
    such feature, so the point estimate for identical rooms/area/floor/city
    must be identical with or without it."""
    resp_plain = client.post("/predict", json=_BASE_PAYLOAD)
    resp_tagged = client.post(
        "/predict", json={**_BASE_PAYLOAD, "property_category": "cottages_sale"}
    )
    assert resp_plain.json()["price"] == resp_tagged.json()["price"]


def test_property_categories_endpoint_lists_reliability(client: TestClient) -> None:
    resp = client.get("/property-categories")
    assert resp.status_code == 200
    by_code = {row["code"]: row["reliability"] for row in resp.json()}
    assert by_code["cottages_sale"] == "limited_data"
    assert by_code["room_sale"] == "limited_data"
    assert by_code["2_rooms_flats_sale"] == "standard"

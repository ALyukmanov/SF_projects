"""Сервис рекомендаций: загрузка артефакта, эндпоинты, обработка ошибок.

Тесты требуют собранный артефакт ``artifacts/model/`` (скрипт
``scripts/train_service_model.py``). Без него — пропуск.
"""
from __future__ import annotations

import numpy as np
import pytest

from retail_recommender.config import PROJECT_ROOT

pytest.importorskip("fastapi", reason="сервис требует fastapi (pip install -r requirements.txt)")

MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.npz").exists(),
    reason="нет artifacts/model/ — запустите scripts/train_service_model.py",
)


@pytest.fixture(scope="module")
def recommender():
    from retail_recommender.service.recommender import ServiceRecommender

    return ServiceRecommender(MODEL_DIR)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from retail_recommender.service.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def warm_visitor(recommender):
    """Визитор с историей, у товаров которой есть co-occurrence соседи."""
    data = np.load(MODEL_DIR / "model.npz")
    uids, offs = data["user_ids"], data["user_offsets"]
    items = data["user_hist_items"]
    for k in np.argsort(np.diff(offs))[::-1][:50]:
        hist = items[offs[k]:offs[k + 1]]
        if any(recommender._neighbours.get(int(it)) for it in hist):
            return int(uids[k])
    raise AssertionError("не нашёлся тёплый визитор для теста")


# ---------------------------------------------------------------- artifact
def test_artifact_loads(recommender):
    info = recommender.model_info()
    assert info["model_type"] == "item_item_cooccurrence"
    assert info["catalog_size"] > 0
    assert info["test_metrics"]["recall@10"] > 0


def test_array_neighbours_roundtrips_dict():
    """ArrayNeighbours отдаёт ровно те же пары, что исходный dict соседей."""
    from retail_recommender.service.artifact import ArrayNeighbours, neighbours_to_arrays

    src = {
        10: [(11, 0.9), (12, 0.5)],
        12: [(10, 0.5)],
        20: [(21, 0.7), (22, 0.6), (23, 0.1)],
    }
    a = neighbours_to_arrays(src)
    nb = ArrayNeighbours(
        a["neighbour_item_ids"], a["neighbour_offsets"],
        a["neighbour_ids"], a["neighbour_sims"],
    )
    for item, pairs in src.items():
        got = nb.get(item)
        assert [g[0] for g in got] == [p[0] for p in pairs]
        assert [round(s, 5) for _, s in got] == [round(s, 5) for _, s in pairs]
    assert nb.get(999) == ()
    assert len(nb) == 3 and bool(nb) is True


# ---------------------------------------------------------------- /health
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok", "model": "item_item_cooccurrence"}


# ---------------------------------------------------------------- /model-info
def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] == "item_item_cooccurrence"
    assert body["default_n"] == 10 and body["max_n"] == 100
    assert body["test_metrics"]["recall@10"] == pytest.approx(0.01156, abs=1e-4)


# ---------------------------------------------------------------- /recommend
def test_recommend_known_user(client, recommender, warm_visitor):
    r = client.post("/recommend", json={"visitorid": warm_visitor, "n": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["visitorid"] == warm_visitor
    assert body["known_user"] is True
    assert body["fallback"] is False
    ids = [it["itemid"] for it in body["recommendations"]]
    assert len(ids) == 10
    assert len(set(ids)) == 10
    assert set(ids) <= recommender._catalog          # только валидные товары
    hist_items, _ = recommender._history(warm_visitor)
    assert not (set(ids) & set(hist_items))          # seen-items policy


def test_recommend_unknown_user_falls_back(client, recommender):
    r = client.post("/recommend", json={"visitorid": -12345, "n": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["known_user"] is False
    assert body["fallback"] is True
    ids = [it["itemid"] for it in body["recommendations"]]
    assert len(ids) == 7
    assert set(ids) <= recommender._catalog


def test_recommend_default_n(client):
    r = client.post("/recommend", json={"visitorid": -1})
    assert r.status_code == 200
    assert len(r.json()["recommendations"]) == 10


def test_recommend_n_respected(client):
    r = client.post("/recommend", json={"visitorid": -1, "n": 3})
    assert r.status_code == 200
    assert len(r.json()["recommendations"]) == 3


# ---------------------------------------------------------------- ошибки -> 4xx
@pytest.mark.parametrize(
    "payload",
    [
        {"visitorid": 1, "n": 0},
        {"visitorid": 1, "n": -5},
        {"visitorid": 1, "n": 1000},
        {"visitorid": "abc"},
        {"n": 10},
        {"visitorid": 1, "unexpected": 1},
    ],
)
def test_recommend_invalid_request(client, payload):
    r = client.post("/recommend", json=payload)
    assert r.status_code == 422


def test_recommend_malformed_json(client):
    r = client.post(
        "/recommend", content="{not valid json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------- /metrics
def test_metrics_endpoint(client):
    client.post("/recommend", json={"visitorid": -1})
    client.post("/recommend", json={"visitorid": 1, "n": -1})  # ошибка
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    text = r.text
    for name in (
        "requests_total",
        "recommend_requests_total",
        "fallback_requests_total",
        "request_errors_total",
        "uptime_seconds",
    ):
        assert name in text

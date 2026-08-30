import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_service import ModelNotLoadedError, ModelService, RetrainInProgressError
from app.schemas import WineFeatures

SAMPLE_WINE = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11,
    "total sulfur dioxide": 34,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


@pytest.fixture
def client(isolated_artifacts):
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("model_version", "model_type", "feature_names", "accuracy", "f1_score", "roc_auc"):
        assert key in body
    assert body["feature_names"]


def test_predict_valid_input(client):
    resp = client.post("/predict", json=SAMPLE_WINE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("low_quality", "good_quality")
    assert 0.0 <= body["good_quality_probability"] <= 1.0
    assert body["model_version"]


def test_predict_missing_field_returns_422(client):
    bad_payload = dict(SAMPLE_WINE)
    del bad_payload["alcohol"]
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_wrong_type_returns_422(client):
    bad_payload = dict(SAMPLE_WINE)
    bad_payload["alcohol"] = "not-a-number"
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_unknown_field_rejected(client):
    bad_payload = dict(SAMPLE_WINE)
    bad_payload["extra_unexpected_field"] = 1.0
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post("/predict/batch", json=[SAMPLE_WINE, SAMPLE_WINE])
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    for item in body:
        assert item["prediction"] in (0, 1)


def test_predict_batch_empty_list_rejected(client):
    resp = client.post("/predict/batch", json=[])
    assert resp.status_code == 400


def test_retrain_changes_model_version(client):
    before = client.get("/model/info").json()["model_version"]
    resp = client.post("/retrain")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] != before
    assert body["training_duration_seconds"] >= 0

    after = client.get("/model/info").json()["model_version"]
    assert after == body["model_version"]


def test_metrics_endpoint_exposes_prometheus_format(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    for metric in (
        "ml_model_accuracy",
        "ml_model_f1_score",
        "ml_predictions_total",
        "ml_prediction_duration_seconds",
        "ml_retraining_total",
        "ml_model_info",
    ):
        assert metric in text


def test_predictions_total_increments_after_predict(client):
    before_text = client.get("/metrics").text
    client.post("/predict", json=SAMPLE_WINE)
    after_text = client.get("/metrics").text

    def extract_total(text: str) -> float:
        for line in text.splitlines():
            if line.startswith("ml_predictions_total "):
                return float(line.split()[-1])
        return 0.0

    assert extract_total(after_text) > extract_total(before_text)


def test_model_service_predict_before_load_raises():
    service = ModelService()
    with pytest.raises(ModelNotLoadedError):
        service.predict_one(WineFeatures(**SAMPLE_WINE))


def test_model_service_rejects_concurrent_retrain(isolated_artifacts):
    """Второй параллельный /retrain отклоняется сразу через неблокирующий
    threading.Lock, независимо от атомарности на уровне файлов."""
    service = ModelService()
    service.ensure_ready()

    service._retrain_lock.acquire()
    try:
        with pytest.raises(RetrainInProgressError):
            service.retrain()
    finally:
        service._retrain_lock.release()


def test_model_service_keeps_serving_previous_model_after_failed_retrain(isolated_artifacts, monkeypatch):
    """Упавший retrain не должен трогать текущую модель: сервис перечитывает
    файлы только после успешного train_and_save."""
    import app.model_service as model_service_module

    service = ModelService()
    service.ensure_ready()
    old_version = service.get_model_version()

    def failing_train_and_save(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(model_service_module, "train_and_save", failing_train_and_save)

    with pytest.raises(RuntimeError):
        service.retrain()

    assert service.get_model_version() == old_version
    prediction, probability, version = service.predict_one(WineFeatures(**SAMPLE_WINE))
    assert version == old_version
    assert prediction in (0, 1)

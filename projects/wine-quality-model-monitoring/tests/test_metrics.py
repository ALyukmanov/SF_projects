from prometheus_client import generate_latest

from app import metrics

SAMPLE_METADATA = {
    "model_version": "20260101T000000000000Z",
    "trained_at": "2026-01-01T00:00:00Z",
    "dataset_rows": 1599,
    "train_rows": 1279,
    "test_rows": 320,
    "accuracy": 0.81,
    "precision": 0.79,
    "recall": 0.83,
    "f1_score": 0.81,
    "roc_auc": 0.88,
    "positive_class_threshold": 6,
    "feature_names": ["alcohol"],
    "model_name": "wine-quality-random-forest",
    "model_type": "RandomForestClassifier",
}


def test_set_model_quality_metrics_updates_gauges():
    metrics.set_model_quality_metrics(SAMPLE_METADATA)

    assert metrics.ml_model_accuracy._value.get() == SAMPLE_METADATA["accuracy"]
    assert metrics.ml_model_precision._value.get() == SAMPLE_METADATA["precision"]
    assert metrics.ml_model_recall._value.get() == SAMPLE_METADATA["recall"]
    assert metrics.ml_model_f1_score._value.get() == SAMPLE_METADATA["f1_score"]
    assert metrics.ml_model_roc_auc._value.get() == SAMPLE_METADATA["roc_auc"]
    assert metrics.ml_model_training_rows._value.get() == SAMPLE_METADATA["train_rows"]
    assert metrics.ml_model_test_rows._value.get() == SAMPLE_METADATA["test_rows"]


def test_record_prediction_success_increments_counters():
    before = metrics.ml_predictions_total._value.get()
    metrics.record_prediction_success(0.01, 0.75, 1)
    after = metrics.ml_predictions_total._value.get()
    assert after == before + 1


def test_record_prediction_error_increments_counter():
    before = metrics.ml_prediction_errors_total._value.get()
    metrics.record_prediction_error()
    after = metrics.ml_prediction_errors_total._value.get()
    assert after == before + 1


def test_record_training_attempt_success_and_failure():
    before_total = metrics.ml_retraining_total._value.get()
    before_failures = metrics.ml_retraining_failures_total._value.get()

    metrics.record_training_attempt(success=True, duration_seconds=1.0)
    metrics.record_training_attempt(success=False, duration_seconds=0.5)

    assert metrics.ml_retraining_total._value.get() == before_total + 2
    assert metrics.ml_retraining_failures_total._value.get() == before_failures + 1


def test_generate_latest_exposes_all_required_metrics():
    metrics.set_model_quality_metrics(SAMPLE_METADATA)
    metrics.record_prediction_success(0.01, 0.5, 0)
    metrics.record_training_attempt(success=True, duration_seconds=1.0)
    metrics.set_model_loaded(True)

    output = generate_latest().decode("utf-8")

    required_metric_names = [
        "ml_model_accuracy",
        "ml_model_precision",
        "ml_model_recall",
        "ml_model_f1_score",
        "ml_model_roc_auc",
        "ml_model_training_rows",
        "ml_model_test_rows",
        "ml_model_last_training_timestamp_seconds",
        "ml_model_info",
        "ml_predictions_total",
        "ml_prediction_errors_total",
        "ml_prediction_duration_seconds",
        "ml_prediction_probability",
        "ml_prediction_class_total",
        "ml_retraining_total",
        "ml_retraining_failures_total",
        "ml_training_duration_seconds",
        "ml_model_version_total",
        "ml_model_loaded",
        "ml_service_uptime_seconds",
    ]
    for name in required_metric_names:
        assert name in output, f"missing metric: {name}"

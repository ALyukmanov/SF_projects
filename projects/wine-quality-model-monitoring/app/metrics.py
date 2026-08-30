"""Метрики Prometheus и функции для их обновления.

Все метрики — синглтоны на уровне модуля, их обновляют и API, и слой
обучения. Labels намеренно низкокардинальные: без ID запросов и сырых
значений признаков. Метка `model_version` встречается только в
`ml_model_info`, и старая серия удаляется при каждом retrain, чтобы
метрика не копила устаревшие версии.
"""
from __future__ import annotations

import time

from prometheus_client import Counter, Gauge, Histogram

_START_TIME = time.time()

# ---------------------------------------------------------------------
# Метрики качества модели (обновляются после каждого успешного обучения)
# ---------------------------------------------------------------------
ml_model_accuracy = Gauge("ml_model_accuracy", "Accuracy of the currently served model on its test split")
ml_model_precision = Gauge("ml_model_precision", "Precision of the currently served model on its test split")
ml_model_recall = Gauge("ml_model_recall", "Recall of the currently served model on its test split")
ml_model_f1_score = Gauge("ml_model_f1_score", "F1-score of the currently served model on its test split")
ml_model_roc_auc = Gauge("ml_model_roc_auc", "ROC AUC of the currently served model on its test split")
ml_model_training_rows = Gauge("ml_model_training_rows", "Number of rows used to train the current model")
ml_model_test_rows = Gauge("ml_model_test_rows", "Number of rows used to test the current model")
ml_model_last_training_timestamp_seconds = Gauge(
    "ml_model_last_training_timestamp_seconds",
    "Unix timestamp of the last successful training run",
)
ml_model_info = Gauge(
    "ml_model_info",
    "Static information about the currently served model",
    ["model_name", "model_version", "model_type"],
)
ml_model_loaded = Gauge("ml_model_loaded", "Whether a model is currently loaded (1) or not (0)")
ml_service_uptime_seconds = Gauge("ml_service_uptime_seconds", "Seconds since the API process started")
ml_service_uptime_seconds.set_function(lambda: time.time() - _START_TIME)

# ---------------------------------------------------------------------
# Метрики инференса
# ---------------------------------------------------------------------
ml_predictions_total = Counter("ml_predictions_total", "Total number of predictions served")
ml_prediction_errors_total = Counter("ml_prediction_errors_total", "Total number of failed prediction requests")
ml_prediction_duration_seconds = Histogram(
    "ml_prediction_duration_seconds",
    "Time spent computing a single prediction",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
ml_prediction_probability = Histogram(
    "ml_prediction_probability",
    "Distribution of predicted good-quality probabilities",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
ml_prediction_class_total = Counter(
    "ml_prediction_class_total",
    "Total predictions grouped by predicted class",
    ["predicted_class"],
)

# ---------------------------------------------------------------------
# Метрики обучения
# ---------------------------------------------------------------------
ml_retraining_total = Counter("ml_retraining_total", "Total number of retraining attempts")
ml_retraining_failures_total = Counter("ml_retraining_failures_total", "Total number of failed retraining attempts")
ml_training_duration_seconds = Histogram(
    "ml_training_duration_seconds",
    "Duration of training runs",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
)
ml_model_version_total = Counter("ml_model_version_total", "Total number of distinct model versions produced")

_last_model_info_labels: tuple | None = None


def set_model_quality_metrics(metadata: dict) -> None:
    """Обновляет все gauge-метрики качества модели по метаданным обучения."""
    global _last_model_info_labels

    ml_model_accuracy.set(metadata["accuracy"])
    ml_model_precision.set(metadata["precision"])
    ml_model_recall.set(metadata["recall"])
    ml_model_f1_score.set(metadata["f1_score"])
    ml_model_roc_auc.set(metadata["roc_auc"])
    ml_model_training_rows.set(metadata["train_rows"])
    ml_model_test_rows.set(metadata["test_rows"])

    trained_at = time.strptime(metadata["trained_at"], "%Y-%m-%dT%H:%M:%SZ")
    ml_model_last_training_timestamp_seconds.set(time.mktime(trained_at))

    labels = (
        metadata.get("model_name", "wine-quality-random-forest"),
        metadata["model_version"],
        metadata.get("model_type", "RandomForestClassifier"),
    )
    if _last_model_info_labels is not None and _last_model_info_labels != labels:
        try:
            ml_model_info.remove(*_last_model_info_labels)
        except KeyError:
            pass
    ml_model_info.labels(*labels).set(1)
    _last_model_info_labels = labels


def record_training_attempt(success: bool, duration_seconds: float) -> None:
    ml_retraining_total.inc()
    ml_training_duration_seconds.observe(duration_seconds)
    if success:
        ml_model_version_total.inc()
    else:
        ml_retraining_failures_total.inc()


def record_prediction_success(duration_seconds: float, probability: float, predicted_class: int) -> None:
    ml_predictions_total.inc()
    ml_prediction_duration_seconds.observe(duration_seconds)
    ml_prediction_probability.observe(probability)
    ml_prediction_class_total.labels(predicted_class=str(predicted_class)).inc()


def record_prediction_error() -> None:
    ml_prediction_errors_total.inc()


def set_model_loaded(loaded: bool) -> None:
    ml_model_loaded.set(1 if loaded else 0)

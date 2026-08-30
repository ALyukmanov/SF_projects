"""Обработчики маршрутов API."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import metrics
from app.model_service import ModelNotLoadedError, RetrainInProgressError, model_service
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RetrainResponse,
    WineFeatures,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_LABELS = {0: "low_quality", 1: "good_quality"}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = model_service.is_loaded()
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_version=model_service.get_model_version() if loaded else None,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        metadata = model_service.get_metadata()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return ModelInfoResponse(**metadata)


@router.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures) -> PredictionResponse:
    started = time.perf_counter()
    try:
        prediction, probability, version = model_service.predict_one(features)
    except ModelNotLoadedError as exc:
        metrics.record_prediction_error()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - любая ошибка инференса превращается в чистый 500
        metrics.record_prediction_error()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed") from exc

    duration = time.perf_counter() - started
    metrics.record_prediction_success(duration, probability, prediction)

    return PredictionResponse(
        prediction=prediction,
        label=_LABELS[prediction],
        good_quality_probability=probability,
        model_version=version,
    )


@router.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(features_list: list[WineFeatures]) -> list[PredictionResponse]:
    if not features_list:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body must contain at least one item")

    started = time.perf_counter()
    try:
        predictions, probabilities, version = model_service.predict_many(features_list)
    except ModelNotLoadedError as exc:
        metrics.record_prediction_error()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - тот же принцип, что и для одиночного /predict
        metrics.record_prediction_error()
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Batch prediction failed") from exc

    duration = time.perf_counter() - started
    per_item_duration = duration / len(features_list)
    results = []
    for prediction, probability in zip(predictions, probabilities):
        metrics.record_prediction_success(per_item_duration, probability, prediction)
        results.append(
            PredictionResponse(
                prediction=prediction,
                label=_LABELS[prediction],
                good_quality_probability=probability,
                model_version=version,
            )
        )
    return results


@router.post("/retrain", response_model=RetrainResponse)
def retrain() -> RetrainResponse:
    try:
        metadata, duration = model_service.retrain()
    except RetrainInProgressError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - обучение упало, старая модель продолжает работать
        metrics.record_training_attempt(success=False, duration_seconds=0.0)
        logger.exception("Retraining failed, previous model remains in service")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining failed, previous model version remains active: {exc}",
        ) from exc

    metrics.record_training_attempt(success=True, duration_seconds=duration)
    metrics.set_model_quality_metrics(metadata)
    metrics.set_model_loaded(True)

    return RetrainResponse(
        status="ok",
        model_version=metadata["model_version"],
        training_duration_seconds=duration,
        metadata=ModelInfoResponse(**metadata),
    )


@router.get("/metrics")
def metrics_endpoint() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

"""Управляет моделью в памяти процесса.

Предсказания можно делать параллельно, а retrain — только один за раз.
Если обучение падает, текущая модель не меняется: `train_and_save`
подменяет файлы на диске только при полном успехе, и только тогда сервис
перечитывает их.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import pandas as pd

from app import config
from app.schemas import WineFeatures
from app.training import train_and_save
from app.utils import read_json

logger = logging.getLogger(__name__)


class RetrainInProgressError(RuntimeError):
    """Второй retrain запущен, пока идёт первый."""


class ModelNotLoadedError(RuntimeError):
    """Модель ещё не загружена."""


class ModelService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._retrain_lock = threading.Lock()
        self._model = None
        self._metadata: Optional[dict] = None

    # -- загрузка -----------------------------------------------------
    def ensure_ready(self) -> None:
        """Загружает модель с диска, при отсутствии — обучает новую."""
        if not config.MODEL_PATH.exists() or not config.METADATA_PATH.exists():
            logger.info("No existing model artifacts found, training a new model")
            train_and_save()
        self.reload_from_disk()

    def reload_from_disk(self) -> None:
        model = joblib.load(config.MODEL_PATH)
        metadata = read_json(config.METADATA_PATH)
        with self._lock:
            self._model = model
            self._metadata = metadata

    # -- доступ к данным -----------------------------------------------
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def get_metadata(self) -> dict:
        with self._lock:
            if self._metadata is None:
                raise ModelNotLoadedError("Model metadata is not available")
            return dict(self._metadata)

    def get_model_version(self) -> Optional[str]:
        with self._lock:
            if self._metadata is None:
                return None
            return self._metadata.get("model_version")

    # -- инференс --------------------------------------------------------
    def _predict_dataframe(self, rows: List[dict]):
        with self._lock:
            if self._model is None:
                raise ModelNotLoadedError("Model is not loaded")
            model = self._model
            version = self._metadata["model_version"]

        frame = pd.DataFrame(rows)[config.FEATURE_NAMES]
        predictions = model.predict(frame)
        probabilities = model.predict_proba(frame)[:, 1]
        return predictions, probabilities, version

    def predict_one(self, features: WineFeatures) -> Tuple[int, float, str]:
        predictions, probabilities, version = self._predict_dataframe(
            [features.to_feature_dict()]
        )
        return int(predictions[0]), float(probabilities[0]), version

    def predict_many(self, features_list: List[WineFeatures]):
        rows = [f.to_feature_dict() for f in features_list]
        predictions, probabilities, version = self._predict_dataframe(rows)
        return (
            [int(p) for p in predictions],
            [float(p) for p in probabilities],
            version,
        )

    # -- переобучение --------------------------------------------------
    def retrain(self, dataset_path: Optional[Path] = None) -> Tuple[dict, float]:
        if not self._retrain_lock.acquire(blocking=False):
            raise RetrainInProgressError("A retraining job is already running")
        try:
            started = time.perf_counter()
            metadata = train_and_save(dataset_path)
            duration = time.perf_counter() - started
            self.reload_from_disk()
            return metadata, duration
        finally:
            self._retrain_lock.release()


model_service = ModelService()

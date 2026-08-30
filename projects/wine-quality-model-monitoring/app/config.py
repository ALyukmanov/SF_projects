"""Пути и константы проекта.

Всё определяется относительно корня проекта и переопределяется через
переменные окружения — один и тот же код работает и локально, и в контейнере.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("ML_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))

DATASET_PATH = Path(os.environ.get("ML_DATASET_PATH", DATA_DIR / "winequality-red.csv"))
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
MODEL_TMP_PATH = ARTIFACTS_DIR / "model.joblib.tmp"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
METADATA_TMP_PATH = ARTIFACTS_DIR / "model_metadata.json.tmp"

FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET_COLUMN = "quality"
REQUIRED_COLUMNS = FEATURE_NAMES + [TARGET_COLUMN]

POSITIVE_CLASS_THRESHOLD = 6  # quality >= 6 -> хорошее вино (is_good = 1)
CSV_SEPARATOR = ";"

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_NAME = "wine-quality-random-forest"
MODEL_TYPE = "RandomForestClassifier"

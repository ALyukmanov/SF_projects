"""Обучение классификатора качества вина.

Запускается как CLI (`python -m app.training`) или вызывается из `/retrain`.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app import config

logger = logging.getLogger(__name__)


class DatasetValidationError(ValueError):
    """Датасет не соответствует ожидаемому формату."""


@dataclass
class TrainingResult:
    model: Pipeline
    metadata: dict


def load_dataset(dataset_path: Optional[Path] = None) -> pd.DataFrame:
    """Загружает и проверяет CSV с данными."""
    path = Path(dataset_path) if dataset_path is not None else config.DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, sep=config.CSV_SEPARATOR)

    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetValidationError(
            f"Dataset is missing required columns: {missing}"
        )

    if df.empty:
        raise DatasetValidationError("Dataset has no rows")

    # Исходный датасет без пропусков, но в дописанных вручную строках они возможны.
    df = df.copy()
    for column in config.REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    rows_before = len(df)
    df = df.dropna(subset=config.REQUIRED_COLUMNS)
    dropped = rows_before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with missing/invalid values", dropped)

    if df.empty:
        raise DatasetValidationError("Dataset has no valid rows after cleaning")

    return df


def build_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_good"] = (df[config.TARGET_COLUMN] >= config.POSITIVE_CLASS_THRESHOLD).astype(int)
    return df


def _make_model_version() -> str:
    # Микросекунды нужны, чтобы версии не совпадали при быстрых повторных
    # обучениях подряд (например, в тестах).
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(dataset_path: Optional[Path] = None) -> TrainingResult:
    """Обучает классификатор и возвращает pipeline вместе с метриками и метаданными."""

    raw_df = load_dataset(dataset_path)
    df = build_binary_target(raw_df)

    X = df[config.FEATURE_NAMES]
    y = df["is_good"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    metadata = {
        "model_version": _make_model_version(),
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        **metrics,
        "positive_class_threshold": config.POSITIVE_CLASS_THRESHOLD,
        "feature_names": list(config.FEATURE_NAMES),
        "model_name": config.MODEL_NAME,
        "model_type": config.MODEL_TYPE,
    }

    return TrainingResult(model=pipeline, metadata=metadata)


def save_model(result: TrainingResult) -> None:
    """Атомарно сохраняет модель и метаданные.

    Сначала пишет во временные файлы, потом переключает их `replace`'ом —
    крах на середине записи не испортит текущую рабочую модель.
    """
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.model, config.MODEL_TMP_PATH)
    config.METADATA_TMP_PATH.write_text(
        json.dumps(result.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # os.replace атомарен и на POSIX, и на Windows NTFS.
    config.MODEL_TMP_PATH.replace(config.MODEL_PATH)
    config.METADATA_TMP_PATH.replace(config.METADATA_PATH)


def train_and_save(dataset_path: Optional[Path] = None) -> dict:
    """Обучает модель и атомарно заменяет старые артефакты.

    При ошибке ничего не трогает — старая модель остаётся рабочей.
    """
    result = train_model(dataset_path)
    save_model(result)
    logger.info(
        "Trained model version=%s accuracy=%.4f f1=%.4f",
        result.metadata["model_version"],
        result.metadata["accuracy"],
        result.metadata["f1_score"],
    )
    return result.metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Обучение классификатора качества вина")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Путь к CSV с данными (по умолчанию data/winequality-red.csv)",
    )
    args = parser.parse_args()

    metadata = train_and_save(args.dataset)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

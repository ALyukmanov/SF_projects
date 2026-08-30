"""Pydantic-схемы запросов и ответов API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class WineFeatures(BaseModel):
    """Один образец вина. Алиасы полей совпадают с колонками датасета
    (с пробелами) — это часть контракта API."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    fixed_acidity: float = Field(alias="fixed acidity", ge=0)
    volatile_acidity: float = Field(alias="volatile acidity", ge=0)
    citric_acid: float = Field(alias="citric acid", ge=0)
    residual_sugar: float = Field(alias="residual sugar", ge=0)
    chlorides: float = Field(alias="chlorides", ge=0)
    free_sulfur_dioxide: float = Field(alias="free sulfur dioxide", ge=0)
    total_sulfur_dioxide: float = Field(alias="total sulfur dioxide", ge=0)
    density: float = Field(alias="density", gt=0)
    pH: float = Field(alias="pH", ge=0, le=14)
    sulphates: float = Field(alias="sulphates", ge=0)
    alcohol: float = Field(alias="alcohol", ge=0)

    def to_feature_dict(self) -> dict:
        return self.model_dump(by_alias=True)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    good_quality_probability: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    model_version: str
    trained_at: str
    feature_names: List[str]
    dataset_rows: int
    train_rows: int
    test_rows: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    positive_class_threshold: int


class RetrainResponse(BaseModel):
    status: str
    model_version: str
    training_duration_seconds: float
    metadata: ModelInfoResponse


class ErrorResponse(BaseModel):
    detail: str

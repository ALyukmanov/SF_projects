"""Схемы запросов/ответов сервиса (Pydantic v2)."""
from __future__ import annotations

from pydantic import BaseModel, Field

MAX_N = 100


class RecommendRequest(BaseModel):
    model_config = {"extra": "forbid"}

    visitorid: int = Field(..., description="идентификатор визитора (браузера/сессии)")
    n: int | None = Field(
        default=None, gt=0, le=MAX_N,
        description=f"сколько товаров вернуть (1..{MAX_N}); по умолчанию — из модели",
    )


class RecommendedItem(BaseModel):
    itemid: int


class RecommendResponse(BaseModel):
    visitorid: int
    recommendations: list[RecommendedItem]
    fallback: bool = Field(
        ..., description="true — рекомендации совпали с популярным (нет персонализации)"
    )
    known_user: bool = Field(..., description="есть ли история визитора в модели")


class HealthResponse(BaseModel):
    status: str
    model: str


class ModelInfoResponse(BaseModel):
    model_type: str
    version: str
    built_at: str
    default_n: int
    max_n: int
    query_items: int
    filter_seen: bool
    catalog_size: int
    items_with_neighbours: int
    users_with_history: int
    history_window_end: str
    train_end: str
    test_window: list[str]
    test_metrics: dict

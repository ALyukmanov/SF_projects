"""FastAPI-приложение сервиса рекомендаций.

Запуск:
    uvicorn retail_recommender.service.app:app --host 0.0.0.0 --port 8000

Каталог модели берётся из переменной окружения ``RETAIL_RECOMMENDER_MODEL_DIR``
(по умолчанию ``artifacts/model``).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from retail_recommender.service import metrics
from retail_recommender.service.recommender import ServiceRecommender
from retail_recommender.service.schemas import (
    HealthResponse,
    ModelInfoResponse,
    RecommendedItem,
    RecommendRequest,
    RecommendResponse,
)

MODEL_DIR = os.environ.get("RETAIL_RECOMMENDER_MODEL_DIR", "artifacts/model")

_state: dict[str, ServiceRecommender] = {}


def _load_recommender() -> ServiceRecommender:
    path = Path(MODEL_DIR)
    if not (path / "model.npz").exists() or not (path / "meta.json").exists():
        raise RuntimeError(
            f"артефакт модели не найден в {path.resolve()} — "
            "сначала выполните: python scripts/train_service_model.py"
        )
    return ServiceRecommender(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["recommender"] = _load_recommender()
    yield
    _state.clear()


app = FastAPI(
    title="RetailRocket recommender service",
    description="MVP-сервис Top-N рекомендаций товаров (item-item co-occurrence).",
    version="1.0.0",
    lifespan=lifespan,
)


def get_recommender() -> ServiceRecommender:
    return _state["recommender"]


@app.middleware("http")
async def _count_requests(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception:
        _record(request, 500)
        raise
    _record(request, response.status_code)
    return response


def _record(request: Request, status: int) -> None:
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    metrics.REQUESTS_TOTAL.labels(request.method, path, str(status)).inc()
    if status >= 400:
        metrics.REQUEST_ERRORS_TOTAL.labels(str(status)).inc()


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    rec = _state.get("recommender")
    model = rec.meta["model_type"] if rec else "unloaded"
    return HealthResponse(status="ok" if rec else "loading", model=model)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**get_recommender().model_info())


@app.get("/metrics", tags=["ops"])
def prometheus_metrics() -> Response:
    payload, content_type = metrics.render()
    return Response(content=payload, media_type=content_type)


@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(req: RecommendRequest) -> RecommendResponse:
    rec = get_recommender()
    result = rec.recommend(req.visitorid, req.n)

    metrics.RECOMMEND_REQUESTS_TOTAL.inc()
    if result.fallback:
        metrics.FALLBACK_REQUESTS_TOTAL.inc()

    return RecommendResponse(
        visitorid=req.visitorid,
        recommendations=[RecommendedItem(itemid=i) for i in result.recommendations],
        fallback=result.fallback,
        known_user=result.known_user,
    )


@app.exception_handler(RuntimeError)
async def _runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})

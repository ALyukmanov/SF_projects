"""Точка входа FastAPI-приложения.

Запуск локально:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app import metrics
from app.api import router
from app.model_service import model_service
from prometheus_client import Counter, Gauge, Histogram

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"]
)
http_requests_in_progress = Gauge(
    "http_requests_in_progress", "HTTP requests currently in flight", ["method", "endpoint"]
)


def _endpoint_label(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return request.url.path


class HttpMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        method = request.method
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            endpoint = request.url.path
            http_requests_total.labels(method=method, endpoint=endpoint, status_code="500").inc()
            raise
        endpoint = _endpoint_label(request)
        duration = time.perf_counter() - started
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
        http_requests_total.labels(method=method, endpoint=endpoint, status_code=str(response.status_code)).inc()
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.ensure_ready()
        metrics.set_model_quality_metrics(model_service.get_metadata())
        metrics.set_model_loaded(True)
        logger.info("Model ready: version=%s", model_service.get_model_version())
    except Exception:
        metrics.set_model_loaded(False)
        logger.exception("Failed to prepare model at startup; service starts without a loaded model")
    yield


app = FastAPI(
    title="API мониторинга ML-модели качества вина",
    description="Бинарный классификатор качества вина на FastAPI, метрики — в Prometheus и Grafana.",
    version="1.0.0",
    lifespan=lifespan,
)


class _InProgressMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        method = request.method
        path = request.url.path
        http_requests_in_progress.labels(method=method, endpoint=path).inc()
        try:
            return await call_next(request)
        finally:
            http_requests_in_progress.labels(method=method, endpoint=path).dec()


app.add_middleware(HttpMetricsMiddleware)
app.add_middleware(_InProgressMiddleware)

app.include_router(router)

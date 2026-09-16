"""Runtime-метрики сервиса в формате Prometheus.

Отдельный monitoring stack не поднимается — это просто endpoint ``/metrics``
с текущими счётчиками процесса.
"""
from __future__ import annotations

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    disable_created_metrics,
    generate_latest,
)

disable_created_metrics()
REGISTRY = CollectorRegistry()

REQUESTS_TOTAL = Counter(
    "requests_total", "All HTTP requests", ["method", "path", "status"],
    registry=REGISTRY,
)
RECOMMEND_REQUESTS_TOTAL = Counter(
    "recommend_requests_total", "Successful recommendation requests", registry=REGISTRY,
)
FALLBACK_REQUESTS_TOTAL = Counter(
    "fallback_requests_total", "Responses served by the popularity fallback",
    registry=REGISTRY,
)
REQUEST_ERRORS_TOTAL = Counter(
    "request_errors_total", "Requests answered with 4xx/5xx", ["status"], registry=REGISTRY,
)
_UPTIME = Gauge("uptime_seconds", "Process uptime in seconds", registry=REGISTRY)

_STARTED_AT = time.monotonic()


def render() -> tuple[bytes, str]:
    _UPTIME.set(time.monotonic() - _STARTED_AT)
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST

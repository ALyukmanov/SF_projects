"""
Real Estate Price Prediction API
FastAPI REST service with prediction endpoint.

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that src.* imports resolve
# regardless of the working directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from api.schemas import (
    BuildingTypeEnum,
    CityEnum,
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    PropertyCategoryEnum,
)
from src.inference.predictor import Predictor
from src.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = get_logger("api.main")

# ---------------------------------------------------------------------------
# Predictor — module-level singleton
# ---------------------------------------------------------------------------
_predictor: Predictor = Predictor(model_path="models")

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
_API_VERSION = "1.0.0"
_API_TITLE = "Real Estate Price Prediction API"
_API_DESCRIPTION = """
## Прогноз цен на недвижимость в России

REST-сервис для оценки рыночной стоимости квартир на вторичном рынке
недвижимости России. Использует обученную модель машинного обучения
(конкретный алгоритм — см. `GET /model-info`, не захардкожен здесь, чтобы
это описание не расходилось с реальным артефактом при его смене); при
отсутствии модели автоматически переключается в **DEMO-режим** с
эвристическими ценами.

### Поддерживаемые города
Москва, Санкт-Петербург, Екатеринбург, Новосибирск, Казань,
Нижний Новгород, Самара, Краснодар.

### Как пользоваться
1. Отправьте `POST /predict` с параметрами квартиры.
2. Получите оценочную стоимость и оценочный диапазон (не гарантия цены и не
   формальный статистический доверительный интервал — см. `interval_method`
   в ответе и MODEL_CARD.md).
3. `GET /health` — проверка состояния сервиса.
4. `GET /model-info` — полные метаданные модели и происхождения данных.
5. `GET /cities` — список поддерживаемых городов.

**Важно:** модель не заменяет профессиональную оценку недвижимости.
"""


# ---------------------------------------------------------------------------
# Lifespan — startup & shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; log on shutdown."""
    logger.info("Starting %s v%s", _API_TITLE, _API_VERSION)
    loaded = _predictor.load()
    if loaded:
        logger.info(
            "Model loaded successfully | type=%s | trained_at=%s",
            _predictor.model_info.get("model_type"),
            _predictor.model_info.get("trained_at"),
        )
    else:
        logger.warning(
            "No trained model found — running in DEMO (heuristic) mode. "
            "Place a .pkl artefact in the 'models/' directory to enable ML predictions."
        )
    yield
    logger.info("Shutting down %s", _API_TITLE)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=_API_TITLE,
    description=_API_DESCRIPTION,
    version=_API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
#
# `allow_origins=["*"]` combined with `allow_credentials=True` is an invalid
# combination per the Fetch/CORS spec — browsers reject credentialed
# requests to a wildcard origin, so the previous config silently did not do
# what it looked like it did. For local demo purposes (Streamlit on 8501,
# the FastAPI docs UI itself, and localhost dev servers) we allow no
# credentials and an explicit local-origin allowlist instead of a wildcard.
# ---------------------------------------------------------------------------

_DEFAULT_CORS_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
#
# Never return the raw exception text to the client — it can leak internal
# details (file paths, library internals, stack-trace fragments). Clients
# get a generic, stable message; the real exception is only logged
# server-side with a correlation-friendly path/exception-type pair.
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Internal server error. Details were logged server-side and are not exposed to clients.",
            error_type=type(exc).__name__,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _format_price(price: float) -> str:
    """Return a human-readable price string in Russian style.

    Examples:
        15_400_000 -> '15.4 млн руб'
        850_000    -> '850 тыс руб'
    """
    if price >= 1_000_000:
        millions = price / 1_000_000
        if millions == int(millions):
            return f"{int(millions)} млн руб"
        return f"{millions:.1f} млн руб"
    thousands = price / 1_000
    if thousands == int(thousands):
        return f"{int(thousands)} тыс руб"
    return f"{thousands:.0f} тыс руб"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get(
    "/",
    include_in_schema=False,
    summary="Redirect to interactive docs",
)
async def root() -> RedirectResponse:
    """Redirect browser requests to the Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Service"],
)
async def health() -> HealthResponse:
    """Return service health status and model information.

    - **status**: always ``"ok"`` when the service is running.
    - **model_loaded**: ``true`` when a trained ML model is active.
    - **model_info**: metadata dict (type, training date, metrics).
    - **version**: API semantic version.
    """
    return HealthResponse(
        status="ok",
        model_loaded=_predictor.is_ready(),
        model_info=_predictor.model_info,
        version=_API_VERSION,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error — bad input"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
    summary="Predict apartment price",
    tags=["Prediction"],
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict the market price of a Russian residential apartment.

    The body must contain basic apartment characteristics.  See the schema
    below for field descriptions and constraints.

    **Returns** a `PredictionResponse` containing:
    - Predicted price (RUB)
    - Estimated range (min / max) — NOT a statistical confidence interval,
      see ``interval_method``
    - Price per sqm
    - Model confidence score (R² in ML mode)
    - Operating mode: ``model`` or ``demo``
    - Human-readable formatted price string
    """
    logger.info(
        "Prediction request | city=%s rooms=%d area=%.1f floor=%d/%d",
        request.city,
        request.rooms,
        request.total_area,
        request.floor,
        request.floors_total,
    )

    # Basic cross-field validation
    if request.floor > request.floors_total:
        raise HTTPException(
            status_code=422,
            detail=(
                f"floor ({request.floor}) cannot exceed " f"floors_total ({request.floors_total})"
            ),
        )

    try:
        result: Dict[str, Any] = _predictor.predict(request.model_dump())
    except RuntimeError as exc:
        # Operational misconfiguration the caller/operator can act on (today:
        # a geo model loaded without data/external/osm_poi.csv). Safe to
        # surface verbatim — it names only a well-known artefact path.
        logger.error("Prediction unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(detail=str(exc), error_type="ModelDependencyMissing").model_dump(),
        ) from exc
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail="Prediction failed. Details were logged server-side.",
                error_type=type(exc).__name__,
            ).model_dump(),
        ) from exc

    price: float = result["price"]
    price_per_sqm: float = round(price / request.total_area, 2) if request.total_area else 0.0
    formatted_price: str = _format_price(price)

    logger.info(
        "Prediction result | price=%.0f mode=%s confidence=%.2f",
        price,
        result["mode"],
        result["confidence"],
    )

    return PredictionResponse(
        price=price,
        price_min=result["price_min"],
        price_max=result["price_max"],
        price_per_sqm=price_per_sqm,
        confidence=result["confidence"],
        interval_method=result.get("interval_method", "unknown"),
        mode=result["mode"],
        formatted_price=formatted_price,
        prediction_reliability=result.get("prediction_reliability", "standard"),
        segment_support=result.get("segment_support"),
    )


@app.get(
    "/model-info",
    summary="Detailed model / data provenance metadata",
    tags=["Service"],
)
async def model_info() -> Dict[str, Any]:
    """Return full model provenance and prediction-interval calibration info.

    Unlike ``/health`` (a lightweight liveness check), this endpoint exposes
    everything known about the currently loaded artefact: data source,
    whether it was trained on synthetic/demo data, the feature schema
    version, dataset hash/row count if available, and the prediction
    interval's calibration method + observed coverage on its holdout set.
    """
    return _predictor.model_info


@app.get(
    "/cities",
    response_model=List[Dict[str, str]],
    summary="List supported cities",
    tags=["Reference"],
)
async def cities() -> List[Dict[str, str]]:
    """Return the list of cities supported by the prediction model.

    Each entry contains:
    - **code**: short ASCII identifier
    - **name**: Russian city name
    """
    return [{"code": member.name, "name": member.value} for member in CityEnum]


@app.get(
    "/building-types",
    response_model=List[Dict[str, str]],
    summary="List supported building types",
    tags=["Reference"],
)
async def building_types() -> List[Dict[str, str]]:
    """Return the list of building types recognised by the model."""
    return [{"code": member.name, "name": member.value} for member in BuildingTypeEnum]


@app.get(
    "/property-categories",
    response_model=List[Dict[str, str]],
    summary="List known property categories and their prediction reliability",
    tags=["Reference"],
)
async def property_categories() -> List[Dict[str, str]]:
    """Return known values for `PredictionRequest.property_category`, each
    tagged with the reliability level it will produce in `/predict`'s
    response.
    """
    from src.inference.predictor import LOW_SUPPORT_SEGMENTS

    return [
        {
            "code": member.value,
            "reliability": "limited_data" if member.value in LOW_SUPPORT_SEGMENTS else "standard",
        }
        for member in PropertyCategoryEnum
    ]

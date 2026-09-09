"""
Pydantic v2 schemas for the Real Estate Price Prediction API.

All request/response models used by the FastAPI endpoints are defined here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CityEnum(str, Enum):
    moskva = "Москва"
    spb = "Санкт-Петербург"
    ekaterinburg = "Екатеринбург"
    novosibirsk = "Новосибирск"
    kazan = "Казань"
    nizhniy_novgorod = "Нижний Новгород"
    samara = "Самара"
    krasnodar = "Краснодар"


class BuildingTypeEnum(str, Enum):
    """Building type. The first four values are recognised by the trained
    model's fixed one-hot categories (see BUILDING_TYPE_ALIASES in
    src/features/feature_engineering.py). ``monolith_brick`` and ``other``
    are accepted but the training data never contained these as a distinct
    category — the model treats them as "unknown building type" (safe
    all-zero encoding), not as a specific bucket. This is disclosed in
    MODEL_CARD.md and the dashboard, not hidden.
    """

    panel = "панельный"
    brick = "кирпичный"
    monolith = "монолитный"
    monolith_brick = "монолитно-кирпичный"
    block = "блочный"
    other = "другое"


class PropertyCategoryEnum(str, Enum):
    """Listing category, matching the values collected in `source_category`
    (see data/processed/real_estate_cleaned.csv). Optional and does NOT
    change the numeric prediction — the model has no category feature (see
    src/inference/predictor.py::LOW_SUPPORT_SEGMENTS). It only lets the API
    return honest `prediction_reliability`/`segment_support` metadata: for
    `cottages_sale` and `room_sale`, the data source does not publish
    rooms/floor/floors_total for these listing types at all, so those
    fields are filled from the ordinary-flat population and the model's
    holdout accuracy there is materially worse.
    """

    rooms_1 = "1_rooms_flats_sale"
    rooms_2 = "2_rooms_flats_sale"
    rooms_3 = "3_rooms_flats_sale"
    studio = "studio_flats_sale"
    new_erect = "new_erect_flats_sale"
    cottages = "cottages_sale"
    room = "room_sale"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    """Input features for a single price-prediction request."""

    rooms: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of rooms (0 = studio)",
        examples=[2],
    )
    total_area: float = Field(
        ...,
        gt=0,
        le=500,
        description="Total area in sqm",
        examples=[55.0],
    )
    floor: int = Field(
        ...,
        ge=1,
        le=100,
        description="Floor number",
        examples=[5],
    )
    floors_total: int = Field(
        ...,
        ge=1,
        le=100,
        description="Total floors in building",
        examples=[16],
    )
    city: str = Field(
        ...,
        description="City name in Russian",
        examples=["Москва"],
    )
    building_type: Optional[str] = Field(
        None,
        description="Building type (панельный, кирпичный, монолитный, …)",
        examples=["монолитный"],
    )
    year_built: Optional[int] = Field(
        None,
        ge=1900,
        le=2026,
        description="Year the building was constructed",
        examples=[2010],
    )
    latitude: Optional[float] = Field(
        None,
        ge=-90,
        le=90,
        description=(
            "Listing latitude (WGS84). Optional. When the loaded model uses "
            "OSM geo features and both latitude and longitude are given, the "
            "API computes nearest-POI distances / radius counts from the "
            "local osm_poi.csv exactly as the training pipeline did. Omit "
            "both and the request is scored with has_coordinates=0 and "
            "sentinel geo values, the same way coordinate-less listings were "
            "handled at training time."
        ),
        examples=[55.7539],
    )
    longitude: Optional[float] = Field(
        None,
        ge=-180,
        le=180,
        description="Listing longitude (WGS84). Optional; see latitude.",
        examples=[37.6208],
    )
    property_category: Optional[str] = Field(
        None,
        description=(
            "Optional listing category (see PropertyCategoryEnum / GET "
            "/property-categories for known values, e.g. 'cottages_sale', "
            "'room_sale', '2_rooms_flats_sale'). Does NOT change the "
            "predicted price (the model has no such feature) — it only "
            "makes the response's prediction_reliability/segment_support "
            "honest for categories (cottages_sale, room_sale) where the "
            "data source structurally lacks rooms/floor/floors_total. Free "
            "text like city/building_type, not a strict enum: an unrecognised "
            "value is treated the same as omitting it (prediction_reliability="
            "'standard'), never rejected. Omit for an ordinary flat."
        ),
        examples=["2_rooms_flats_sale"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "rooms": 2,
                "total_area": 55.0,
                "floor": 5,
                "floors_total": 16,
                "city": "Москва",
                "building_type": "монолитный",
                "year_built": 2010,
            }
        }
    }


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """Prediction result returned by POST /predict."""

    price: float = Field(..., description="Predicted price in RUB")
    price_min: float = Field(
        ...,
        description=(
            "Lower bound of the estimated range in RUB. NOT a statistical "
            "confidence interval — see interval_method."
        ),
    )
    price_max: float = Field(..., description="Upper bound of the estimated range in RUB")
    price_per_sqm: float = Field(..., description="Price per sqm in RUB")
    confidence: float = Field(
        ..., description="Model R² (ML mode) or a fixed heuristic value (DEMO mode), in [0, 1]"
    )
    interval_method: str = Field(
        default="unknown",
        description=(
            "How price_min/price_max were derived: 'residual_quantile_holdout' "
            "(calibrated on the training holdout set), "
            "'heuristic_fixed_fraction_uncalibrated' (older artefact without a "
            "calibrated interval), or 'demo_heuristic_fixed_fraction' (DEMO mode)."
        ),
    )
    mode: str = Field(..., description="'model' when a trained ML model is used, 'demo' otherwise")
    formatted_price: str = Field(
        ..., description="Human-readable price string, e.g. '15.4 млн руб'"
    )
    prediction_reliability: str = Field(
        default="standard",
        description=(
            "'standard' for ordinary flat categories, or 'limited_data' when "
            "property_category was given as a structurally underrepresented "
            "category (cottages_sale, room_sale) — see segment_support for why."
        ),
    )
    segment_support: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Present only when prediction_reliability='limited_data': holdout "
            "sample size, holdout R² and the reason this category's estimate "
            "is less reliable than an ordinary flat's."
        ),
    )


class HealthResponse(BaseModel):
    """Response returned by GET /health."""

    status: str = Field(..., description="'ok' when the service is running")
    model_loaded: bool = Field(..., description="True when a trained model is loaded")
    model_info: dict = Field(..., description="Metadata about the loaded model")
    version: str = Field(..., description="API version string")


class ErrorResponse(BaseModel):
    """Returned on 4xx / 5xx errors."""

    detail: str = Field(..., description="Human-readable error message")
    error_type: str = Field(..., description="Error category, e.g. 'PredictionError'")

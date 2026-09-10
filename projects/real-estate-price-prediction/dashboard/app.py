"""
Real Estate Price Prediction Dashboard
Interactive Streamlit application for apartment price estimation
and exploratory analytics.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that src.* imports resolve
# regardless of the working directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.inference.predictor import Predictor

# ---------------------------------------------------------------------------
# Page configuration — must be the FIRST streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CITIES: List[str] = [
    "Москва",
    "Санкт-Петербург",
    "Екатеринбург",
    "Новосибирск",
    "Казань",
    "Нижний Новгород",
    "Самара",
    "Краснодар",
]

BUILDING_TYPES: List[str] = [
    "панельный",
    "кирпичный",
    "монолитный",
    "монолитно-кирпичный",
    "блочный",
    "другое",
]

ROOMS_LABELS: List[str] = ["Студия", "1", "2", "3", "4+"]
ROOMS_VALUES: Dict[str, int] = {
    "Студия": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4+": 4,
}

# Base price per sqm (RUB) used to generate realistic demo data
_CITY_BASE_PRICE: Dict[str, float] = {
    "Москва": 280_000.0,
    "Санкт-Петербург": 180_000.0,
    "Екатеринбург": 90_000.0,
    "Новосибирск": 85_000.0,
    "Казань": 80_000.0,
    "Нижний Новгород": 75_000.0,
    "Самара": 72_000.0,
    "Краснодар": 78_000.0,
}

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_PRIMARY = "#1F77B4"
_SECONDARY = "#FF7F0E"
_SUCCESS = "#2CA02C"
_DANGER = "#D62728"
_CHART_COLORS = px.colors.qualitative.Plotly

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
    /* Sidebar header */
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1F77B4;
        margin-bottom: 0.5rem;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
        border-left: 5px solid #1F77B4;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }

    /* Mode badge */
    .badge-model {
        display: inline-block;
        background: #2CA02C;
        color: white;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
    }
    .badge-demo {
        display: inline-block;
        background: #FF7F0E;
        color: white;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
    }

    /* Section header */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #1F77B4;
        padding-bottom: 4px;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Info table */
    .info-table td {
        padding: 4px 12px 4px 0;
        font-size: 0.9rem;
    }
    .info-table td:first-child {
        color: #666;
        font-weight: 500;
    }
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Загрузка модели…")
def load_predictor() -> Predictor:
    """Load (or initialise) the Predictor singleton.  Cached across sessions."""
    predictor = Predictor(model_path="models")
    predictor.load()
    return predictor


@st.cache_data(show_spinner=False)
def load_analytics_data() -> tuple[pd.DataFrame, bool]:
    """Load processed data if available, otherwise generate synthetic demo data.

    "is_real_data" here means "not synthetic" — a file is only reported as
    real if it does NOT have an `is_synthetic=True` column and its filename
    does not contain "synthetic" (see DATA_CARD.md / scripts/run_feature_engineering_real.py).
    This matters because every dataset currently produced by this project's
    default commands IS synthetic, and the Analytics page must not present
    it as real market data.

    Returns:
        (DataFrame, is_real_data)
    """
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    for csv_file in sorted(
        processed_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        try:
            df = pd.read_csv(csv_file)
            required = {"city", "rooms", "total_area", "floor", "floors_total", "price"}
            if required.issubset(set(df.columns)) and len(df) >= 10:
                is_synthetic = bool(df.get("is_synthetic", pd.Series([False])).any()) or (
                    "synthetic" in csv_file.name
                )
                # Ensure price_per_sqm column exists
                if "price_per_sqm" not in df.columns:
                    df["price_per_sqm"] = (
                        df["price"] / df["total_area"].replace(0, float("nan"))
                    ).round(2)
                # Normalise city names to title case to match CITIES list
                if df["city"].str.islower().any():
                    city_map = {c.lower(): c for c in CITIES}
                    df["city"] = df["city"].str.lower().map(city_map).fillna(df["city"])
                df = df[df["city"].isin(CITIES)]
                if len(df) >= 10:
                    return df[
                        [
                            "city",
                            "rooms",
                            "total_area",
                            "floor",
                            "floors_total",
                            "price",
                            "price_per_sqm",
                        ]
                    ].copy(), (not is_synthetic)
        except Exception:
            continue
    return generate_demo_data(), False


@st.cache_data(show_spinner=False)
def generate_demo_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset for the Analytics page.

    The data is realistic in terms of city-level price distributions,
    room counts, and building ages.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    rows = []
    for _ in range(n):
        city = rng.choice(CITIES)
        rooms = rng.choices([0, 1, 2, 3, 4], weights=[5, 25, 35, 25, 10])[0]
        area = float(np_rng.normal(loc=20 + rooms * 18, scale=10))
        area = max(18.0, min(area, 200.0))
        floor = rng.randint(1, 20)
        floors_total = rng.randint(floor, max(floor, 20))
        year_built = rng.randint(1960, 2023)

        base = _CITY_BASE_PRICE[city]
        noise = float(np_rng.normal(1.0, 0.15))
        price_per_sqm = base * noise
        price = price_per_sqm * area

        # First / top floor adjustments
        if floor == 1:
            price *= 0.95
        elif floor == floors_total:
            price *= 0.97

        rows.append(
            {
                "city": city,
                "rooms": rooms,
                "total_area": round(area, 1),
                "floor": floor,
                "floors_total": floors_total,
                "year_built": year_built,
                "price": round(price),
                "price_per_sqm": round(price_per_sqm),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_price(price: float) -> str:
    """Format price as a human-readable Russian string."""
    if price >= 1_000_000:
        return f"{price / 1_000_000:.2f} млн ₽"
    return f"{price / 1_000:.0f} тыс ₽"


def fmt_number(value: float) -> str:
    """Format large numbers with space as thousands separator."""
    return f"{value:,.0f}".replace(",", "\u202f")


def round_to_sensible_precision(price: float) -> float:
    """Round a price to a precision that doesn't overstate model accuracy.

    The model's own MAE is on the order of 1-2 million RUB (see model_info
    metrics) \u2014 displaying a prediction to the exact ruble (e.g.
    "10 856 840 \u20bd") implies false precision. Round to the nearest 10 000 RUB,
    which is still far finer than the model's actual error margin but avoids
    a misleadingly exact-looking number.
    """
    return round(price / 10_000) * 10_000


# Plausible training-data ranges (from the 200-row synthetic demo dataset \u2014
# see DATA_CARD.md). Inputs outside these ranges are not "wrong", but the
# model has not seen comparable examples during training, so predictions
# there are extrapolations and should carry an explicit warning.
_TRAINING_RANGES: Dict[str, tuple] = {
    "total_area": (20.0, 150.0),
    "floor": (1, 24),
    "year_built": (1960, 2023),
}
_KNOWN_BUILDING_TYPES_FOR_MODEL = {
    "\u043f\u0430\u043d\u0435\u043b\u044c\u043d\u044b\u0439",
    "\u043a\u0438\u0440\u043f\u0438\u0447\u043d\u044b\u0439",
    "\u043c\u043e\u043d\u043e\u043b\u0438\u0442\u043d\u044b\u0439",
    "\u0431\u043b\u043e\u0447\u043d\u044b\u0439",
}


def out_of_range_warnings(
    total_area: float, floor: int, year_built: Optional[int], building_type: Optional[str]
) -> List[str]:
    """Return a list of human-readable warnings for inputs outside the
    training data's observed range (\u00a7K13 of the modernization brief)."""
    warnings: List[str] = []
    lo, hi = _TRAINING_RANGES["total_area"]
    if not (lo <= total_area <= hi):
        warnings.append(
            f"\u041f\u043b\u043e\u0449\u0430\u0434\u044c {total_area:.0f} \u043c\u00b2 \u0432\u044b\u0445\u043e\u0434\u0438\u0442 \u0437\u0430 \u043f\u0440\u0435\u0434\u0435\u043b\u044b \u0434\u0438\u0430\u043f\u0430\u0437\u043e\u043d\u0430 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u0445 \u0434\u0430\u043d\u043d\u044b\u0445 "
            f"({lo:.0f}\u2013{hi:.0f} \u043c\u00b2) \u2014 \u043f\u0440\u043e\u0433\u043d\u043e\u0437 \u044f\u0432\u043b\u044f\u0435\u0442\u0441\u044f \u044d\u043a\u0441\u0442\u0440\u0430\u043f\u043e\u043b\u044f\u0446\u0438\u0435\u0439."
        )
    lo, hi = _TRAINING_RANGES["floor"]
    if not (lo <= floor <= hi):
        warnings.append(
            f"\u042d\u0442\u0430\u0436 {floor} \u0432\u044b\u0445\u043e\u0434\u0438\u0442 \u0437\u0430 \u043f\u0440\u0435\u0434\u0435\u043b\u044b \u0434\u0438\u0430\u043f\u0430\u0437\u043e\u043d\u0430 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u0445 \u0434\u0430\u043d\u043d\u044b\u0445 ({lo:.0f}\u2013{hi:.0f})."
        )
    if year_built is not None:
        lo, hi = _TRAINING_RANGES["year_built"]
        if not (lo <= year_built <= hi):
            warnings.append(
                f"\u0413\u043e\u0434 \u043f\u043e\u0441\u0442\u0440\u043e\u0439\u043a\u0438 {year_built} \u0432\u044b\u0445\u043e\u0434\u0438\u0442 \u0437\u0430 \u043f\u0440\u0435\u0434\u0435\u043b\u044b \u0434\u0438\u0430\u043f\u0430\u0437\u043e\u043d\u0430 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u0445 \u0434\u0430\u043d\u043d\u044b\u0445 "
                f"({lo:.0f}\u2013{hi:.0f})."
            )
    if building_type is not None and building_type not in _KNOWN_BUILDING_TYPES_FOR_MODEL:
        warnings.append(
            f"\u0422\u0438\u043f \u0437\u0434\u0430\u043d\u0438\u044f \u00ab{building_type}\u00bb \u043d\u0435 \u0432\u0445\u043e\u0434\u0438\u0442 \u0432 4 \u043a\u0430\u0442\u0435\u0433\u043e\u0440\u0438\u0438, \u043f\u0440\u0435\u0434\u0441\u0442\u0430\u0432\u043b\u0435\u043d\u043d\u044b\u0435 \u0432 "
            "\u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u0445 \u0434\u0430\u043d\u043d\u044b\u0445 (\u043f\u0430\u043d\u0435\u043b\u044c\u043d\u044b\u0439/\u043a\u0438\u0440\u043f\u0438\u0447\u043d\u044b\u0439/\u043c\u043e\u043d\u043e\u043b\u0438\u0442\u043d\u044b\u0439/\u0431\u043b\u043e\u0447\u043d\u044b\u0439) \u2014 \u043c\u043e\u0434\u0435\u043b\u044c \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0430\u0435\u0442 "
            "\u0435\u0433\u043e \u043a\u0430\u043a \u00ab\u0442\u0438\u043f \u043d\u0435 \u0443\u043a\u0430\u0437\u0430\u043d\u00bb."
        )
    return warnings


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    """Render the sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">🏠 Оценка недвижимости</div>', unsafe_allow_html=True
        )
        st.caption("Real Estate Price Prediction")
        st.divider()

        page = st.radio(
            "Навигация",
            options=["Прогноз", "Аналитика", "О проекте"],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        predictor = load_predictor()
        if predictor.is_ready():
            info = predictor.model_info
            if info.get("is_synthetic"):
                st.warning("ML-модель загружена (обучена на **синтетических** данных)", icon="⚠️")
            else:
                st.success("ML-модель загружена", icon="✅")
            st.caption(f"Тип: {info.get('model_type', 'N/A')}")
            st.caption(f"Дата обучения: {info.get('trained_at', 'N/A')}")
            st.caption(f"Источник данных: {info.get('data_source', 'N/A')}")
            if info.get("geo_enabled"):
                cov = (info.get("coordinate_coverage") or {}).get("coordinate_coverage_pct")
                cov_txt = f" · координаты у {cov:.0f}% объявлений" if cov else ""
                if info.get("geo_poi_available"):
                    st.caption(f"Геопризнаки OSM: включены{cov_txt}")
                else:
                    st.error(
                        "Модель использует геопризнаки OSM, но файл "
                        "`data/external/osm_poi.csv` не найден — прогноз недоступен. "
                        "Запустите `python scripts/prepare_osm_poi.py`.",
                        icon="⚠️",
                    )
            else:
                st.caption("Геопризнаки OSM: не используются")
        else:
            st.warning("DEMO-режим (модель не найдена)", icon="⚠️")
            st.caption(
                "Используется эвристическая оценка по средним ценам городов, а не ML-модель."
            )

        st.divider()
        st.caption("Учебный проект · не заменяет профессиональную оценку недвижимости")

    return page


# ---------------------------------------------------------------------------
# Prediction page
# ---------------------------------------------------------------------------


def render_prediction_page() -> None:
    st.title("🏠 Прогноз цены недвижимости")
    st.markdown("Укажите характеристики квартиры в боковой панели и нажмите **Рассчитать**.")
    st.divider()

    # --- Sidebar inputs ---
    with st.sidebar:
        st.markdown('<div class="section-header">Параметры квартиры</div>', unsafe_allow_html=True)

        city = st.selectbox("Город", options=CITIES, index=0)

        rooms_label = st.selectbox("Количество комнат", options=ROOMS_LABELS, index=2)
        rooms = ROOMS_VALUES[rooms_label]

        total_area = st.slider(
            "Общая площадь, м²",
            min_value=20,
            max_value=200,
            value=55,
            step=1,
        )

        col_fl1, col_fl2 = st.columns(2)
        with col_fl1:
            floor = st.number_input("Этаж", min_value=1, max_value=100, value=5, step=1)
        with col_fl2:
            floors_total = st.number_input("Этажей", min_value=1, max_value=100, value=16, step=1)

        if floor > floors_total:
            st.warning("Этаж не может превышать количество этажей в доме.")
            floors_total = int(floor)

        building_type = st.selectbox(
            "Тип здания",
            options=["(не указан)"] + BUILDING_TYPES,
            index=0,
        )
        building_type_val: Optional[str] = None if building_type == "(не указан)" else building_type

        use_year = st.checkbox("Указать год постройки", value=False)
        year_built: Optional[int] = None
        if use_year:
            year_built = st.number_input(
                "Год постройки", min_value=1900, max_value=2026, value=2000, step=1
            )

        # --- Coordinates (optional) — used only if the loaded model has geo features ---
        _pred_geo = load_predictor()
        _geo_on = _pred_geo.is_ready() and _pred_geo.model_info.get("geo_enabled")
        latitude: Optional[float] = None
        longitude: Optional[float] = None
        if _geo_on:
            use_coords = st.checkbox(
                "Указать координаты (расстояния до метро, школ и т.д.)", value=False
            )
            if use_coords:
                latitude = st.number_input(
                    "Широта",
                    min_value=-90.0,
                    max_value=90.0,
                    value=55.7558,
                    step=0.0001,
                    format="%.4f",
                )
                longitude = st.number_input(
                    "Долгота",
                    min_value=-180.0,
                    max_value=180.0,
                    value=37.6173,
                    step=0.0001,
                    format="%.4f",
                )
                st.caption(
                    "Только Москва и Санкт-Петербург (для них есть данные OSM). Без "
                    "координат прогноз работает, но без геопризнаков."
                )

        property_category_label = st.selectbox(
            "Тип объекта",
            options=["Квартира", "Комната", "Дом / коттедж"],
            index=0,
            help="Для комнат и домов источник не публикует часть полей — оценка менее надёжна.",
        )
        _cat_map = {"Комната": "room_sale", "Дом / коттедж": "cottages_sale"}
        property_category: Optional[str] = _cat_map.get(property_category_label)

        st.divider()
        calculate = st.button("🔍 Рассчитать", type="primary", use_container_width=True)

    # --- Prediction result ---
    if calculate:
        features: Dict[str, Any] = {
            "rooms": rooms,
            "total_area": float(total_area),
            "floor": int(floor),
            "floors_total": int(floors_total),
            "city": city,
            "building_type": building_type_val,
            "year_built": year_built,
        }
        if latitude is not None and longitude is not None:
            features["latitude"] = float(latitude)
            features["longitude"] = float(longitude)
        if property_category is not None:
            features["property_category"] = property_category

        predictor = load_predictor()
        try:
            with st.spinner("Вычисляем оценку…"):
                result = predictor.predict(features)
        except RuntimeError as exc:
            st.error(f"Прогноз недоступен: {exc}", icon="🚫")
            st.stop()

        # Round to a precision that doesn't overstate the model's actual
        # accuracy (MAE is on the order of millions of RUB — see "О модели").
        price: float = round_to_sensible_precision(result["price"])
        price_min: float = round_to_sensible_precision(result["price_min"])
        price_max: float = round_to_sensible_precision(result["price_max"])
        confidence: float = result["confidence"]
        mode: str = result["mode"]
        interval_method: str = result.get("interval_method", "unknown")
        price_per_sqm: float = price / total_area if total_area else 0.0

        range_warnings = out_of_range_warnings(
            total_area=float(total_area),
            floor=int(floor),
            year_built=year_built,
            building_type=building_type_val,
        )
        if range_warnings:
            st.warning(
                "**Внимание — экстраполяция за пределы обучающих данных:**\n\n"
                + "\n".join(f"- {w}" for w in range_warnings),
                icon="⚠️",
            )

        # --- Main metric row ---
        st.markdown("### Результат оценки")
        col_main, col_right = st.columns([2, 1])

        with col_main:
            st.metric(
                label="Оценочная стоимость",
                value=f"{fmt_number(price)} ₽",
                delta=None,
            )

        with col_right:
            badge_class = "badge-model" if mode == "model" else "badge-demo"
            badge_label = "ML-модель" if mode == "model" else "Эвристика"
            st.markdown(
                f'<span class="{badge_class}">Режим: {badge_label}</span>',
                unsafe_allow_html=True,
            )

        # --- 3-column detail row ---
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Цена", fmt_price(price))
        with c2:
            st.metric("Цена за м²", f"{fmt_number(price_per_sqm)} ₽/м²")
        with c3:
            metric_label = "R² модели" if mode == "model" else "Условная достоверность"
            st.metric(metric_label, f"{confidence * 100:.0f}%")

        # --- Estimated range bar (NOT a formal statistical confidence interval) ---
        st.markdown('<div class="section-header">Оценочный диапазон</div>', unsafe_allow_html=True)
        _interval_captions = {
            "residual_quantile_holdout": (
                "Диапазон откалиброван по остаткам модели на отложенной выборке "
                "(целевое покрытие указано в MODEL_CARD.md). Это оценочный диапазон, "
                "**не формальный статистический доверительный интервал**."
            ),
            "heuristic_fixed_fraction_uncalibrated": (
                "⚠️ Диапазон — нескалиброванная эвристика (фиксированный процент от цены), "
                "не проверенная на исторических данных."
            ),
            "demo_heuristic_fixed_fraction": (
                "Диапазон — простая эвристика DEMO-режима (±15% от оценки), не основан на модели."
            ),
        }
        st.caption(_interval_captions.get(interval_method, "Метод расчёта диапазона не определён."))

        fig_range = go.Figure()
        fig_range.add_trace(
            go.Scatter(
                x=[price_min, price, price_max],
                y=[0, 0, 0],
                mode="markers+lines",
                marker=dict(
                    size=[14, 22, 14],
                    color=[_SECONDARY, _PRIMARY, _SECONDARY],
                    symbol=["circle", "diamond", "circle"],
                ),
                line=dict(color=_PRIMARY, width=4),
                text=[
                    f"Мин: {fmt_price(price_min)}",
                    f"Оценка: {fmt_price(price)}",
                    f"Макс: {fmt_price(price_max)}",
                ],
                hoverinfo="text",
            )
        )
        fig_range.update_layout(
            height=120,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            yaxis=dict(visible=False),
            xaxis=dict(
                tickformat=",.0f",
                title="Цена, ₽",
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_range, use_container_width=True)

        col_min, col_max = st.columns(2)
        col_min.caption(f"Нижняя граница: **{fmt_price(price_min)}**")
        col_max.caption(f"Верхняя граница: **{fmt_price(price_max)}**")

        # --- Geo-features indicator ---
        _info = predictor.model_info
        if _info.get("geo_enabled"):
            if features.get("latitude") is not None and city in ("Москва", "Санкт-Петербург"):
                st.success(
                    "Геопризнаки использованы: расстояния до метро, школ, парков и т.д. "
                    "рассчитаны по указанным координатам.",
                    icon="📍",
                )
            else:
                st.info(
                    "Координаты не заданы (или город без данных OSM) — прогноз сделан без "
                    "геопризнаков, `has_coordinates = 0`.",
                    icon="📍",
                )

        # --- Segment-reliability warning ---
        if result.get("prediction_reliability") == "limited_data":
            seg = result.get("segment_support") or {}
            st.warning(
                f"**Ограниченные данные для категории «{property_category_label}».** "
                f"{seg.get('reason', '')} "
                f"(holdout R² ≈ {seg.get('holdout_r2', 'н/д')}, "
                f"объектов на holdout: {seg.get('n_holdout', 'н/д')}). "
                "Оценка менее надёжна, чем для обычной квартиры.",
                icon="⚠️",
            )

        # --- Input summary ---
        st.markdown('<div class="section-header">Параметры запроса</div>', unsafe_allow_html=True)
        rooms_display = "Студия" if rooms == 0 else f"{rooms}-комнатная"
        _coord_row = (
            f"<tr><td>Координаты</td><td><b>{features['latitude']:.4f}, "
            f"{features['longitude']:.4f}</b></td></tr>"
            if features.get("latitude") is not None
            else "<tr><td>Координаты</td><td><b>не указаны</b></td></tr>"
        )
        summary_html = f"""
        <div class="result-card">
        <table class="info-table">
          <tr><td>Город</td><td><b>{city}</b></td></tr>
          <tr><td>Тип квартиры</td><td><b>{rooms_display}</b></td></tr>
          <tr><td>Площадь</td><td><b>{total_area} м²</b></td></tr>
          <tr><td>Этаж</td><td><b>{floor} из {floors_total}</b></td></tr>
          <tr><td>Тип здания</td><td><b>{building_type_val or "не указан"}</b></td></tr>
          <tr><td>Год постройки</td><td><b>{year_built if year_built else "не указан"}</b></td></tr>
          {_coord_row}
        </table>
        </div>
        """
        st.markdown(summary_html, unsafe_allow_html=True)

        if mode == "demo":
            st.info(
                "Оценка выполнена в **DEMO-режиме** (эвристика по средним ценам города, не ML-модель). "
                "Для ML-прогноза поместите обученный артефакт в папку `models/`.",
                icon="ℹ️",
            )
        elif predictor.model_info.get("is_synthetic"):
            st.warning(
                "Модель обучена на **синтетических демо-данных** (200 искусственно "
                "сгенерированных записей, не реальные объявления CIAN — см. DATA_CARD.md). "
                "Эта оценка не отражает реальный рынок недвижимости.",
                icon="⚠️",
            )

        st.caption(
            "Модель не заменяет профессиональную оценку недвижимости и не является "
            "офертой или гарантией цены сделки."
        )
    else:
        # Placeholder before first calculation
        st.info("Заполните параметры в боковой панели и нажмите **Рассчитать**.", icon="👈")


# ---------------------------------------------------------------------------
# Analytics page
# ---------------------------------------------------------------------------


def render_analytics_page() -> None:
    st.title("📊 Аналитика рынка недвижимости")
    st.divider()

    df, is_real = load_analytics_data()
    if is_real:
        st.success(
            f"Данные загружены из `data/processed/` — {len(df):,} объявлений.",
            icon="✅",
        )
    else:
        st.warning(
            "Используются **синтетические демо-данные**. "
            "Поместите обработанный CSV в `data/processed/` для реальной аналитики.",
            icon="⚠️",
        )

    # --- Sidebar filter ---
    with st.sidebar:
        st.markdown('<div class="section-header">Фильтры</div>', unsafe_allow_html=True)
        selected_cities = st.multiselect(
            "Города",
            options=CITIES,
            default=CITIES,
        )
        room_filter = st.multiselect(
            "Комнат",
            options=[0, 1, 2, 3, 4],
            default=[0, 1, 2, 3, 4],
            format_func=lambda x: "Студия" if x == 0 else str(x),
        )

    filtered = df[df["city"].isin(selected_cities) & df["rooms"].isin(room_filter)]

    if filtered.empty:
        st.warning("Нет данных для выбранных фильтров. Измените параметры в боковой панели.")
        return

    # --- Summary metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Объявлений", len(filtered))
    c2.metric("Ср. цена", fmt_price(filtered["price"].mean()))
    c3.metric("Ср. площадь", f"{filtered['total_area'].mean():.1f} м²")
    c4.metric("Ср. цена/м²", f"{fmt_number(filtered['price_per_sqm'].mean())} ₽")

    st.divider()

    # --- Row 1: Bar + Scatter ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Медианная цена по городам")
        city_stats = (
            filtered.groupby("city")["price"]
            .median()
            .reset_index()
            .sort_values("price", ascending=False)
        )
        city_stats.columns = ["Город", "Медианная цена"]
        fig_bar = px.bar(
            city_stats,
            x="Город",
            y="Медианная цена",
            color="Город",
            color_discrete_sequence=_CHART_COLORS,
            labels={"Медианная цена": "Цена, ₽"},
            text_auto=".3s",
        )
        fig_bar.update_traces(textfont_size=11, textangle=0)
        fig_bar.update_layout(
            showlegend=False,
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.subheader("Площадь vs Цена")
        rooms_map = {0: "Студия", 1: "1 комн.", 2: "2 комн.", 3: "3 комн.", 4: "4+ комн."}
        scatter_df = filtered.copy()
        scatter_df["Комнат"] = scatter_df["rooms"].map(rooms_map)
        fig_scatter = px.scatter(
            scatter_df,
            x="total_area",
            y="price",
            color="Комнат",
            hover_data=["city", "floor", "floors_total"],
            labels={"total_area": "Площадь, м²", "price": "Цена, ₽"},
            color_discrete_sequence=_CHART_COLORS,
            opacity=0.7,
        )
        fig_scatter.update_traces(marker_size=8)
        fig_scatter.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Row 2: Histogram + Box plot ---
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("Распределение цен")
        fig_hist = px.histogram(
            filtered,
            x="price",
            nbins=30,
            labels={"price": "Цена, ₽"},
            color_discrete_sequence=[_PRIMARY],
        )
        fig_hist.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            bargap=0.05,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Количество",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right2:
        st.subheader("Цены по количеству комнат")
        rooms_order = ["Студия", "1 комн.", "2 комн.", "3 комн.", "4+ комн."]
        scatter_df2 = filtered.copy()
        scatter_df2["Комнат"] = scatter_df2["rooms"].map(rooms_map)
        fig_box = px.box(
            scatter_df2,
            x="Комнат",
            y="price",
            category_orders={"Комнат": rooms_order},
            labels={"price": "Цена, ₽"},
            color="Комнат",
            color_discrete_sequence=_CHART_COLORS,
        )
        fig_box.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # --- Price per sqm by city (heatmap-style table) ---
    st.subheader("Средняя цена за м² по городам")
    ppsqm_stats = (
        filtered.groupby("city")["price_per_sqm"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
    )
    ppsqm_stats.columns = ["Город", "Среднее", "Медиана", "Мин", "Макс", "Объявлений"]
    for col in ["Среднее", "Медиана", "Мин", "Макс"]:
        ppsqm_stats[col] = ppsqm_stats[col].apply(lambda x: f"{x:,.0f} ₽".replace(",", "\u202f"))
    st.dataframe(ppsqm_stats, use_container_width=True, hide_index=True)

    if not is_real:
        st.caption("⚠️ Все данные синтетические и предназначены только для демонстрации интерфейса.")


# ---------------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------------


def render_about_page() -> None:
    st.title("ℹ️ О проекте")
    st.divider()

    # --- Project description ---
    st.markdown(
        """
        ## Прогноз цен на вторичном рынке жилья России

        Проект представляет собой сквозной пайплайн машинного обучения:
        от сбора данных с сайтов агрегаторов до интерактивного веб-интерфейса
        для оценки стоимости квартир.

        ### Ключевые возможности
        - Оценка стоимости квартиры по основным характеристикам
        - Геопризнаки из OpenStreetMap (расстояния до метро, школ, парков и т.д.)
          по координатам объявления — для Москвы и Санкт-Петербурга
        - Оценочный диапазон прогноза (не формальный доверительный интервал)
        - REST API для интеграции с внешними системами
        - DEMO-режим без обученной модели (эвристические цены)

        > **Учебный проект первого года обучения.** Не заменяет профессиональную
        > оценку недвижимости. См. `DATA_CARD.md` и `MODEL_CARD.md` для честного
        > описания происхождения данных и ограничений модели.
        """
    )

    # --- Technology stack ---
    # The model name is read from the live Predictor, not hardcoded, so it
    # stays correct if the current model changes.
    predictor = load_predictor()
    info = predictor.model_info
    _model_label = (
        str(info.get("model_type", "N/A")).replace("_", " ")
        if predictor.is_ready()
        else "эвристика (DEMO, без модели)"
    )

    st.subheader("Технологический стек")
    tech_data = {
        "Компонент": [
            "Модель МО",
            "API-сервер",
            "Дашборд",
            "Данные",
            "Оркестровка",
            "Контейнеризация",
        ],
        "Технология": [
            f"{_model_label} / scikit-learn",
            "FastAPI + Uvicorn",
            "Streamlit",
            "pandas / numpy",
            "Python 3.11",
            "Docker / docker-compose",
        ],
        "Описание": [
            "Регрессия на log-цену с feature engineering",
            "Асинхронный REST-сервис с Pydantic-схемами",
            "Интерактивный веб-интерфейс с Plotly-графиками",
            "Обработка и агрегация обучающих данных",
            "Единая кодовая база проекта",
            "Воспроизводимое развёртывание",
        ],
    }
    st.table(pd.DataFrame(tech_data))

    # --- Architecture diagram ---
    # Model box label filled from the same live model_type as the table
    # above, for the same reason (no hardcoded algorithm name to drift).
    st.subheader("Архитектура системы")
    # Fixed-width box column (13 chars, matching the ASCII template below) —
    # truncate long model names (e.g. "linear regression" is longer than
    # "XGBoost" ever was) so the box drawing doesn't visually break.
    _box_width = 13
    _raw_box_text = f"({_model_label})" if predictor.is_ready() else "(нет модели)"
    _model_box_label = (
        _raw_box_text
        if len(_raw_box_text) <= _box_width
        else _raw_box_text[: _box_width - 2] + "…)"
    )
    st.code(
        f"""
┌─────────────────────────────────────────────────────────┐
│                    Пользователь / Browser               │
└────────────────┬───────────────────────┬────────────────┘
                 │                       │
        ┌────────▼────────┐   ┌──────────▼──────────┐
        │  Streamlit UI   │   │   FastAPI REST API   │
        │  (port 8501)    │   │   (port 8000)        │
        └────────┬────────┘   └──────────┬───────────┘
                 │                       │
         ┌───────▼───────────────────────▼────────┐
         │           src/inference/Predictor       │
         │  ┌──────────────┐  ┌────────────────┐  │
         │  │  ML Model    │  │  DEMO heuristic│  │
         │  │  {_model_box_label:<13}│  │  (city prices) │  │
         │  └──────┬───────┘  └───────┬────────┘  │
         └─────────┼──────────────────┼────────────┘
                   │                  │
              ┌────▼──────────────────▼────┐
              │     models/*.pkl           │
              │     (artefact storage)     │
              └────────────────────────────┘
        """,
        language="text",
    )

    # --- Data & model provenance (honest status, not marketing) ---
    st.subheader("Происхождение данных и модели")
    metrics: dict = info.get("metrics", {})

    if info.get("is_synthetic"):
        st.error(
            "**Текущая модель обучена на синтетических демо-данных, а не на реальных "
            "объявлениях CIAN.** 200 записей сгенерированы скриптом "
            "`scripts/run_feature_engineering_real.py` (фейковые адреса «ул. Примерная», "
            "последовательные тестовые URL). Метрики ниже описывают качество модели ТОЛЬКО "
            "на этих синтетических данных и не говорят ничего о точности на реальном рынке. "
            "Подробности — в `DATA_CARD.md` и `MODEL_CARD.md`.",
            icon="🚫",
        )
    elif not predictor.is_ready():
        st.info("Модель не загружена — активен DEMO-режим (эвристика, не ML-модель).")
    else:
        st.success(
            "Загруженный артефакт не помечен как синтетический "
            f"(data_source: {info.get('data_source', 'N/A')})."
        )

    st.caption(
        f"feature_schema_version: {info.get('feature_schema_version', 'N/A')} · "
        f"dataset_rows: {info.get('dataset_rows', 'N/A')} · "
        f"dataset_sha256: {str(info.get('dataset_sha256'))[:12]}…"
        if info.get("dataset_sha256")
        else f"feature_schema_version: {info.get('feature_schema_version', 'N/A')}"
    )

    # --- Model metrics ---
    st.subheader("Метрики модели")
    if metrics:
        metric_rows = [{"Метрика": k.upper(), "Значение": f"{v:.4f}"} for k, v in metrics.items()]
        st.table(pd.DataFrame(metric_rows))
        interval = info.get("prediction_interval") or {}
        if interval:
            st.caption(
                f"Оценочный диапазон: целевое покрытие "
                f"{interval.get('coverage_target', 0):.0%}, наблюдаемое покрытие на "
                f"отложенной выборке {interval.get('observed_coverage', 0):.0%} "
                f"(n={interval.get('n_holdout', 'N/A')}). Не формальный доверительный интервал — "
                f"см. MODEL_CARD.md."
            )
    else:
        st.info(
            "Модель не загружена (DEMO-режим) — метрики недоступны, потому что ML-модель "
            "не используется. Никакие приблизительные/декоративные числа не показываются, "
            "чтобы не создавать ложное впечатление о качестве модели."
        )

    # --- Features ---
    st.subheader("Используемые признаки")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown(
            """
            **Базовые**
            - `rooms` — количество комнат
            - `total_area` — общая площадь, м²
            - `floor` — этаж
            - `floors_total` — этажность дома
            - `city` — город (fixed-category one-hot, 8 городов)
            - `building_type` — тип здания (fixed-category one-hot, 4 типа;
              «монолитно-кирпичный»/«другое» обрабатываются как «тип не указан»)
            """
        )
    with col_f2:
        st.markdown(
            """
            **Производные**
            - `building_age` — возраст здания (относительно даты объявления)
            - `floor_ratio` — этаж / этажность
            - `is_first_floor` / `is_top_floor`
            - `log_area` — логарифм площади
            - `room_density` — комнат / площадь
            - `rooms_x_area` — произведение

            **Макроэкономические**
            - `key_rate` — ключевая ставка ЦБ РФ
            - `usd_rate` — курс доллара
            - `inflation_rate` — инфляция
            - `rate_change_6m` — изменение ставки за 6 мес.
            """
        )

    # --- Data sources ---
    st.subheader("Источники данных")
    st.markdown(
        """
        | Источник | Тип | Статус |
        |---|---|---|
        | ЦИАН (cian.ru) | Листинги | Скрапер реализован и покрыт тестами; **массовый сбор не запускался** в текущей сборке — см. DATA_CARD.md |
        | Банк России (cbr.ru) | Макро (ключевая ставка, курсы валют) | API-клиент реализован (`use_api=True`); текущая модель использует offline-константы (`use_api=False`) |
        """
    )

    # --- Future improvements ---
    st.subheader("Планируемые улучшения")
    st.markdown(
        """
        - [ ] Загрузка и обновление данных в режиме реального времени
        - [ ] Поддержка новостроек (первичный рынок)
        - [ ] Геокодирование и тепловые карты цен
        - [ ] Ансамблирование нескольких моделей
        - [ ] A/B-тестирование моделей через API
        - [ ] История запросов и кэширование предсказаний
        - [ ] Специализированная mobile-first доработка интерфейса (крупные
              элементы управления, компактные графики для маленьких экранов) —
              за пределами стандартного адаптивного поведения Streamlit
              (сворачивание боковой панели, реflow), которое работает "из
              коробки" без дополнительного кода в этом приложении
        """
    )


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------


def main() -> None:
    page = render_sidebar()

    if page == "Прогноз":
        render_prediction_page()
    elif page == "Аналитика":
        render_analytics_page()
    elif page == "О проекте":
        render_about_page()


if __name__ == "__main__":
    main()

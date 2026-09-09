"""
Inference module for Real Estate Price Prediction.

Provides a unified ``Predictor`` class used by both the REST API and the
Streamlit dashboard.  Falls back to a heuristic DEMO mode when no trained
model is available.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.feature_engineering import (
    BUILDING_TYPE_ALIASES,
    BUILDING_TYPE_SLUGS,
    CITY_SLUGS,
    FEATURE_SCHEMA_VERSION,
    FeatureEngineer,
    one_hot_fixed_categories,
)
from src.features.geo_features import GEO_FEATURE_COLUMNS, GeoFeatureBuilder
from src.preprocessing.imputer import GroupMedianImputer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# City-level base price per sqm (RUB) used in DEMO / heuristic mode only.
# This is a simple rule-based heuristic, unrelated to the ML model's learned
# coefficients — kept separate on purpose so DEMO mode never pretends to be
# the trained model.
# ---------------------------------------------------------------------------

_CITY_BASE_PRICE_PER_SQM: Dict[str, float] = {
    "москва": 280_000.0,
    "москва ": 280_000.0,  # trailing space guard
    "санкт-петербург": 180_000.0,
    "спб": 180_000.0,
    "екатеринбург": 90_000.0,
    "новосибирск": 85_000.0,
    "казань": 80_000.0,
    "нижний новгород": 75_000.0,
    "самара": 72_000.0,
    "краснодар": 78_000.0,
}

_DEFAULT_PRICE_PER_SQM = 70_000.0

# Heuristic estimated-range width in DEMO mode (±15%). NOT a statistical
# confidence interval — see PREDICTION_INTERVAL notes in _predict_demo/
# _predict_with_model docstrings and reports/MODEL_EXPERIMENTS.md.
_DEMO_CI_FRACTION = 0.15

# ---------------------------------------------------------------------------
# Feature order expected by the model. This is derived from
# ``FeatureEngineer`` (the single source of truth for feature construction)
# rather than duplicated as an independent literal, to remove the drift risk:
# previously this
# list and the trainer's feature list were two separately maintained
# constants guarded only by a regression test. NOTE: an artefact's own
# ``feature_names`` (persisted at training time) still takes precedence when
# a model is loaded — this constant is only the fallback used when an older
# artefact does not carry its own feature list.
# ---------------------------------------------------------------------------
_MODEL_FEATURES: List[str] = FeatureEngineer()._build_default_feature_list()

# Macro defaults used when the model expects economic features but none are provided
_MACRO_DEFAULTS: Dict[str, float] = {
    "key_rate": 21.0,
    "usd_rate": 90.0,
    "inflation_rate": 8.5,
    "rate_change_6m": 0.0,
}

# ---------------------------------------------------------------------------
# Segment reliability.
#
# The model does NOT take property category as an input feature at all — it
# only sees rooms/area/floor/city/building_type/macro. For most categories
# (flats) that's fine. For `cottages_sale` and `room_sale`, restate.ru
# structurally does not publish `rooms` (both) or `floor`/`floors_total`
# (cottages) — not a scraping gap, the source site does not have these
# fields for these listing types. Those rows'
# rooms/floor/floors_total are filled from the ordinary-flat population's
# city medians, which is a poor proxy for a house or a single room, so the
# model measurably underperforms there:
#   cottages_sale: n=158 (42 holdout), holdout R²≈0.22
#   room_sale:     n=198 (38 holdout), holdout R²≈-0.80 (worse than predicting the mean)
# vs R²>0.7 for the ordinary flat categories. A larger scrape would not fix
# this — the missing fields are absent at the source regardless of volume.
# `property_category` is an optional, honesty-only input: it does not change
# the numeric prediction (the model has no such feature to use), only the
# `prediction_reliability`/`segment_support` metadata attached to the
# response, so a caller is not misled into thinking a cottage/room estimate
# carries the same confidence as an ordinary flat.
# ---------------------------------------------------------------------------
LOW_SUPPORT_SEGMENTS: Dict[str, Dict[str, Any]] = {
    "cottages_sale": {
        "n_train": 116,
        "n_holdout": 42,
        "holdout_r2": 0.22,
        "reason": (
            "rooms/floor/floors_total are not published by the data source for houses "
            "(100% missing) and are backfilled from the ordinary-flat city median, which "
            "is not a meaningful proxy for a house."
        ),
    },
    "room_sale": {
        "n_train": 160,
        "n_holdout": 38,
        "holdout_r2": -0.80,
        "reason": (
            "rooms is not published by the data source for single-room listings (100% "
            "missing) and is backfilled from the ordinary-flat city median."
        ),
    },
}


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class Predictor:
    """Load a trained model and produce price predictions.

    If no model is available the predictor falls back to a rule-based DEMO
    mode so the application remains functional during development.

    Args:
        model_path: Directory that contains serialised ``.pkl`` model artefacts
                    (default: ``'models'``).
    """

    def __init__(self, model_path: str = "models") -> None:
        self._model_path = Path(model_path)
        self._model = None
        self._scaler = None
        self._model_type: Optional[str] = None
        self._trained_at: Optional[str] = None
        self._metrics: Dict[str, float] = {}
        self._feature_names: List[str] = []
        self._artifact_metadata: Dict[str, Any] = {}
        self._is_loaded: bool = False
        self._imputer: Optional[GroupMedianImputer] = None
        self._geo_builder: Optional[GeoFeatureBuilder] = None
        self._geo_unavailable: bool = False

        logger.info("Predictor initialised | model_path=%s", self._model_path)

    @property
    def _expects_geo_features(self) -> bool:
        """True when the loaded model's feature list contains OSM geo columns."""
        return any(c in self._feature_names for c in GEO_FEATURE_COLUMNS)

    def _check_geo_dependency(self) -> None:
        """If the loaded model expects OSM geo features, the local
        ``osm_poi.csv`` must be present — detected here at load time, not
        mysteriously on the first ``/predict``. A pre-geo model needs no
        OSM data and this is a no-op for it.
        """
        self._geo_unavailable = False
        if not self._expects_geo_features:
            return
        self._geo_builder = GeoFeatureBuilder()
        if not self._geo_builder.available:
            self._geo_unavailable = True
            logger.error(
                "Loaded model uses OSM geo features but the POI table (%s) is missing or "
                "empty. /predict will return a clear error until it is built: "
                "run `python scripts/prepare_osm_poi.py`.",
                self._geo_builder.poi_csv,
            )

    def _geo_feature_dict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Compute the geo feature columns for a single live request.

        Uses ``latitude``/``longitude`` from the request if present, matched
        against the local ``osm_poi.csv`` the same way the training pipeline
        did. With no coordinates (or no POI table) every listing falls to the
        sentinel/zero row with ``has_coordinates=0`` — identical to how
        coordinate-less listings were treated at training time.
        """
        if self._geo_builder is None:
            self._geo_builder = GeoFeatureBuilder()
        row = {
            "city": features.get("city", ""),
            "latitude": features.get("latitude"),
            "longitude": features.get("longitude"),
        }
        out = self._geo_builder.transform(pd.DataFrame([row])).iloc[0]
        return {col: float(out[col]) for col in GEO_FEATURE_COLUMNS}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load the "current" model artefact from :attr:`model_path`.

        Selection order:
        1. ``{model_path}/current_model.json`` manifest (written by
           ``ModelTrainer.save()``) — explicit, not dependent on filesystem
           mtime, which is fragile (e.g. restoring/copying files can reorder
           mtimes and silently switch which artefact gets served).
        2. Fallback: most recently modified ``*.pkl`` in the directory, for
           artefacts saved before the manifest existed, or if the manifest is
           missing/corrupt/points at a file that no longer exists — logged
           explicitly as a fallback, not silent.

        Returns:
            ``True`` if a model was loaded successfully, ``False`` otherwise.
        """
        target = self._resolve_current_artifact_path()
        if target is None:
            logger.warning(
                "No .pkl model files found in '%s'. Running in DEMO mode.", self._model_path
            )
            return False

        latest = target
        try:
            import joblib

            artefact = joblib.load(latest)
            self._model = artefact.get("model")
            self._scaler = artefact.get("scaler")
            self._model_type = artefact.get("model_type", "unknown")
            self._trained_at = artefact.get("trained_at")
            self._metrics = artefact.get("metrics", {})
            self._feature_names = artefact.get("feature_names", _MODEL_FEATURES)
            self._artifact_metadata = {
                k: v for k, v in artefact.items() if k not in ("model", "scaler")
            }
            self._imputer = GroupMedianImputer.from_dict(artefact.get("imputer"))
            self._is_loaded = True
            self._warn_on_sklearn_version_mismatch()
            self._check_geo_dependency()
            logger.info(
                "Model loaded from %s | type=%s | trained_at=%s | geo=%s",
                latest,
                self._model_type,
                self._trained_at,
                self._expects_geo_features,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", latest, exc, exc_info=True)
            return False

    def _resolve_current_artifact_path(self) -> Optional[Path]:
        """Return the artefact path to load, per the manifest-first order
        documented in :meth:`load`."""
        manifest_path = self._model_path / "current_model.json"
        if manifest_path.is_file():
            try:
                import json

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                candidate = self._model_path / manifest["filename"]
                if candidate.is_file():
                    logger.info("Using current_model.json manifest -> %s", candidate.name)
                    return candidate
                logger.warning(
                    "current_model.json points at '%s' which does not exist under '%s' "
                    "— falling back to mtime-latest .pkl.",
                    manifest.get("filename"),
                    self._model_path,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to read/parse current_model.json (%s) — falling back to "
                    "mtime-latest .pkl.",
                    exc,
                )

        pkl_files = sorted(self._model_path.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
        return pkl_files[-1] if pkl_files else None

    # Fallback expected sklearn version for artefacts that don't carry their
    # own `library_versions.sklearn` metadata (pre-provenance artefacts).
    # Artefacts trained by scripts/run_model_training_real.py after
    # provenance tracking was added record their own training-time version,
    # which takes precedence over this constant.
    _EXPECTED_SKLEARN_VERSION = "1.3.2"

    def _warn_on_sklearn_version_mismatch(self) -> None:
        """Log a clear, non-blocking warning on scikit-learn version mismatch.

        Unpickling a model trained under a different scikit-learn version than the
        one installed raises sklearn's own ``InconsistentVersionWarning`` via the
        ``warnings`` module, which is easy to miss in deployment logs. This surfaces
        the same risk through our own logger with an explicit expected-vs-installed
        message. A mismatch does not necessarily break predictions but is not
        guaranteed to be safe across arbitrary version gaps.
        """
        try:
            import sklearn

            installed = sklearn.__version__
        except Exception:
            return
        expected = (
            self._artifact_metadata.get("library_versions", {}).get("sklearn")
            or self._EXPECTED_SKLEARN_VERSION
        )
        if installed != expected:
            logger.warning(
                "scikit-learn version mismatch: installed=%s, but this artefact "
                "was trained under scikit-learn==%s. Predictions should still "
                "work, but for guaranteed compatibility recreate the "
                "environment from requirements.txt or retrain the model.",
                installed,
                expected,
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the apartment price from a feature dictionary.

        Accepted input keys:
        - ``rooms`` (int)
        - ``total_area`` (float, sqm)
        - ``floor`` (int)
        - ``floors_total`` (int)
        - ``city`` (str, Russian name or slug)
        - ``building_type`` (str, optional)
        - ``year_built`` (int, optional)
        - ``key_rate``, ``usd_rate`` (float, optional — macro overrides)
        - ``property_category`` (str, optional — reliability metadata only,
          see LOW_SUPPORT_SEGMENTS; does not affect the numeric prediction)

        Returns:
            Dict with keys:
            - ``price``       — point estimate (RUB)
            - ``price_min``   — lower bound of the *estimated range* (RUB) —
              NOT a statistical confidence interval, see ``interval_method``
            - ``price_max``   — upper bound of the estimated range (RUB)
            - ``confidence``  — model R² (ML mode) or a fixed heuristic value
              (DEMO mode) in [0, 1] — not a prediction probability
            - ``interval_method`` — how price_min/price_max were derived
              (``"residual_quantile_holdout"``, ``"heuristic_fixed_fraction_uncalibrated"``,
              or ``"demo_heuristic_fixed_fraction"``)
            - ``mode``        — ``'model'`` or ``'demo'``
        """
        if self._is_loaded and self._geo_unavailable:
            raise RuntimeError(
                "The loaded model uses OSM geo features, but the POI table "
                "(data/external/osm_poi.csv) is missing. Build it with "
                "`python scripts/prepare_osm_poi.py`, or point current_model.json "
                "at a pre-geo artefact."
            )
        if self._is_loaded and self._model is not None:
            result = self._predict_with_model(features)
        else:
            result = self._predict_demo(features)
        result.update(self.segment_reliability(features.get("property_category")))
        return result

    @staticmethod
    def segment_reliability(property_category: Optional[str]) -> Dict[str, Any]:
        """Return honest reliability metadata for *property_category*.

        Does NOT influence the numeric prediction — the model has no
        property-category feature to condition on (see LOW_SUPPORT_SEGMENTS
        docstring above). This purely flags, for the handful of structurally
        underrepresented categories, that the estimate should not be trusted
        to the same degree as an ordinary flat.
        """
        if not property_category:
            return {"prediction_reliability": "standard", "segment_support": None}
        key = str(property_category).strip().lower()
        info = LOW_SUPPORT_SEGMENTS.get(key)
        if info is None:
            return {"prediction_reliability": "standard", "segment_support": None}
        return {
            "prediction_reliability": "limited_data",
            "segment_support": {
                "category": key,
                "n_train": info["n_train"],
                "n_holdout": info["n_holdout"],
                "holdout_r2": info["holdout_r2"],
                "reason": info["reason"],
            },
        }

    # ------------------------------------------------------------------
    # Model-based prediction
    # ------------------------------------------------------------------

    def _predict_with_model(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Run the loaded ML model.

        The returned ``price_min``/``price_max`` are an *estimated range*
        (оценочный диапазон), not a statistical confidence interval: they
        come from ``prediction_interval`` persisted in the model artefact
        (residual quantiles measured on the training pipeline's holdout
        set — see ``compute_residual_quantile_interval`` in
        ``src/models/trainer.py``). If an older artefact has no persisted
        interval, a clearly-flagged ±10% heuristic fallback is used instead.
        """
        try:
            X = self._build_feature_vector(features)
            feature_list = self._feature_names or _MODEL_FEATURES

            # Reindex to match training columns; fill missing with 0
            X_df = pd.DataFrame([X]).reindex(columns=feature_list, fill_value=0)

            if self._scaler is not None:
                X_scaled = self._scaler.transform(X_df)
            else:
                X_scaled = X_df.values

            log_price = float(self._model.predict(X_scaled)[0])
            price = float(np.expm1(log_price))

            interval = self._artifact_metadata.get("prediction_interval") or {}
            if interval and "lower_offset" in interval and "upper_offset" in interval:
                price_min = price + interval["lower_offset"]
                price_max = price + interval["upper_offset"]
                interval_method = interval.get("method", "residual_quantile_holdout")
            else:
                # Fallback for artefacts saved before the interval was
                # persisted — explicitly an untested heuristic, not a
                # calibrated range.
                ci = 0.10
                price_min = price * (1 - ci)
                price_max = price * (1 + ci)
                interval_method = "heuristic_fixed_fraction_uncalibrated"

            price_min = max(price_min, 0.0)
            confidence = min(self._metrics.get("r2", 0.85), 1.0)

            logger.debug(
                "Model prediction: %.0f RUB (estimated range: %.0f - %.0f, method=%s)",
                price,
                price_min,
                price_max,
                interval_method,
            )
            return {
                "price": round(price),
                "price_min": round(price_min),
                "price_max": round(price_max),
                "confidence": round(confidence, 4),
                "interval_method": interval_method,
                "mode": "model",
            }
        except Exception as exc:
            logger.error("Model prediction failed: %s. Falling back to DEMO.", exc, exc_info=True)
            return self._predict_demo(features)

    _HARDCODED_DEFAULTS: Dict[str, float] = {
        "rooms": 2.0,
        "total_area": 50.0,
        "floor": 3.0,
        "floors_total": 9.0,
    }

    def _fill_missing_listing_fields(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Return rooms/total_area/floor/floors_total, filling any field
        absent from *features* the same way offline training would have.

        Runs the raw values through the loaded artefact's persisted
        train-fit ``GroupMedianImputer.transform()`` (city-aware, same
        fallback chain as ``DataCleaner``/``run_feature_engineering_real.py``
        used offline — see ``src/preprocessing/imputer.py``) rather than a
        single hardcoded literal, so a live request that happens to omit one
        of these fields is filled consistently with how the model's own
        training data was. Falls back to fixed literals for artefacts saved
        before the imputer was persisted (e.g. the current synthetic
        production model) — DEMO mode never reaches this method.
        """
        raw = {
            "rooms": features.get("rooms"),
            "total_area": features.get("total_area"),
            "floor": features.get("floor"),
            "floors_total": features.get("floors_total"),
            "city": features.get("city", ""),
        }
        if self._imputer is not None and self._imputer.fitted_:
            filled = self._imputer.transform(pd.DataFrame([raw])).iloc[0]
            return {
                "rooms": float(filled["rooms"]),
                "total_area": float(filled["total_area"]),
                "floor": float(filled["floor"]),
                "floors_total": float(filled["floors_total"]),
            }
        return {
            field: float(raw[field]) if raw[field] is not None else self._HARDCODED_DEFAULTS[field]
            for field in ("rooms", "total_area", "floor", "floors_total")
        }

    def _build_feature_vector(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Construct a complete numeric feature dict from raw input.

        ``building_age`` intentionally uses the *current* year here (unlike
        historical training rows, which use listing year — see
        ``DataCleaner._engineer_derived``): this method values a NEW
        prediction request "as of today", so "now" is the correct reference
        year, not a bug.
        """
        filled = self._fill_missing_listing_fields(features)
        rooms = filled["rooms"]
        total_area = filled["total_area"]
        floor = filled["floor"]
        floors_total = filled["floors_total"]
        year_built = features.get("year_built")

        # Derived features
        safe_area = total_area if total_area != 0 else 1.0
        safe_floors = floors_total if floors_total != 0 else 1.0

        building_age = (
            max(float(datetime.now().year - int(year_built)), 0.0)
            if year_built is not None
            else 0.0
        )
        # Rounding below matches src.features.feature_engineering.FeatureEngineer's
        # training-time formulas exactly (_make_floor_features/_make_room_density/
        # _make_interaction_features) so the same listing produces a bit-identical
        # feature vector whether it goes through the offline training pipeline or
        # this live inference path -- see tests/test_inference_parity.py.
        floor_ratio = round(floor / safe_floors, 4)
        is_first_floor = int(floor == 1)
        is_top_floor = int(floor == floors_total)
        log_area = float(np.log1p(total_area))
        room_density = round(rooms / safe_area, 6)
        rooms_x_area = round(rooms * total_area, 2)

        city_ohe = (
            one_hot_fixed_categories(
                pd.Series([str(features.get("city", ""))]), CITY_SLUGS, prefix="city"
            )
            .iloc[0]
            .to_dict()
        )
        building_type_ohe = (
            one_hot_fixed_categories(
                pd.Series([str(features.get("building_type", ""))]),
                BUILDING_TYPE_SLUGS,
                prefix="building_type",
                aliases=BUILDING_TYPE_ALIASES,
            )
            .iloc[0]
            .to_dict()
        )

        macro = {k: features.get(k, _MACRO_DEFAULTS[k]) for k in _MACRO_DEFAULTS}

        # Geo features are only emitted when the loaded model actually expects
        # them — otherwise this stays a no-op so the vector matches exactly
        # what a pre-geo model was trained on (and keeps inference-parity with
        # FeatureEngineer, which likewise only selects geo columns when the
        # training frame carried them).
        geo = self._geo_feature_dict(features) if self._expects_geo_features else {}

        return {
            "rooms": rooms,
            "total_area": total_area,
            "floor": floor,
            "floors_total": floors_total,
            "building_age": building_age,
            "floor_ratio": floor_ratio,
            "is_top_floor": is_top_floor,
            "is_first_floor": is_first_floor,
            "log_area": log_area,
            "room_density": room_density,
            "rooms_x_area": rooms_x_area,
            **city_ohe,
            **building_type_ohe,
            **macro,
            **geo,
        }

    # ------------------------------------------------------------------
    # DEMO / heuristic prediction
    # ------------------------------------------------------------------

    def _predict_demo(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Simple heuristic price estimate for DEMO mode."""
        total_area = float(features.get("total_area", 50.0))
        floor = int(features.get("floor", 3))
        floors_total = int(features.get("floors_total", 9))
        city = str(features.get("city", "")).strip().lower()

        base_ppsqm = _CITY_BASE_PRICE_PER_SQM.get(city, _DEFAULT_PRICE_PER_SQM)
        price = base_ppsqm * total_area

        # Floor adjustments
        if floor == 1:
            price *= 0.95  # first floor -5 %
        elif floor == floors_total:
            price *= 0.97  # top floor -3 %

        price_min = max(price * (1 - _DEMO_CI_FRACTION), 0.0)
        price_max = price * (1 + _DEMO_CI_FRACTION)

        logger.debug(
            "DEMO prediction for city='%s', area=%.1f sqm: %.0f RUB", city, total_area, price
        )
        return {
            "price": round(price),
            "price_min": round(price_min),
            "price_max": round(price_max),
            "confidence": 0.60,
            "interval_method": "demo_heuristic_fixed_fraction",
            "mode": "demo",
        }

    # ------------------------------------------------------------------
    # Status / metadata
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return ``True`` if a trained model is loaded and ready."""
        return self._is_loaded and self._model is not None

    @property
    def model_info(self) -> Dict[str, Any]:
        """Return metadata about the currently loaded model.

        Returns:
            Dict with keys ``model_type``, ``trained_at``, ``metrics``,
            ``is_demo``, ``model_path``, ``data_source``, ``is_synthetic``,
            ``feature_schema_version``, ``prediction_interval`` (coverage
            target/observed + method), ``dataset_rows``. Fields that are
            absent from an older artefact fall back to ``"N/A"``/``None`` —
            callers (API ``/health``, ``/model-info``, dashboard "О модели")
            must handle that explicitly rather than assume every artefact
            carries full provenance.
        """
        meta = self._artifact_metadata
        return {
            "model_type": self._model_type or "N/A",
            "trained_at": self._trained_at or "N/A",
            "metrics": self._metrics,
            "is_demo": not self._is_loaded,
            "model_path": str(self._model_path),
            "data_source": meta.get("data_source", "N/A" if self._is_loaded else "demo-heuristic"),
            "is_synthetic": meta.get("is_synthetic"),
            "feature_schema_version": meta.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
            "dataset_rows": meta.get("dataset_rows"),
            "dataset_sha256": meta.get("dataset_sha256"),
            "prediction_interval": meta.get("prediction_interval", {}),
            "geo_enabled": self._expects_geo_features,
            "geo_poi_available": (not self._geo_unavailable) if self._expects_geo_features else None,
            "coordinate_coverage": meta.get("coordinate_coverage"),
            "osm_poi_manifest": meta.get("osm_poi_manifest"),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_price_per_sqm(city: Any) -> float:
        """Return the heuristic price-per-sqm for *city*."""
        city_str = str(city).strip().lower()
        return _CITY_BASE_PRICE_PER_SQM.get(city_str, _DEFAULT_PRICE_PER_SQM)

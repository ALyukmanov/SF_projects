"""
Feature engineering module for the Real Estate Price Prediction project.

Transforms a cleaned listings DataFrame into a feature matrix ready for
machine-learning training or inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature schema version — bump whenever the *set* or *meaning* of model
# input columns changes (used in model-artifact metadata so an artifact can
# be checked against the code that would currently produce its features).
# v2 replaced the single fit-before-split `LabelEncoder`-based `city_encoded`
# integer column with fixed-category one-hot columns for `city` and
# `building_type` (see CITY_SLUGS / BUILDING_TYPE_SLUGS below). This removes
# the "LabelEncoder as pseudo-ordinal scale" risk, makes `building_type`
# actually reach the model (it was previously accepted by the API/UI and
# silently discarded), and needs no fitted encoder object to be persisted
# because the categories are fixed constants, not fit from data.
# ---------------------------------------------------------------------------
FEATURE_SCHEMA_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Fixed, versioned category lists for one-hot encoding.
#
# These are intentionally *hardcoded constants*, not fit from the training
# data, so that:
#   - the same encoding is trivially reproducible at inference time without
#     persisting/deserialising an encoder object;
#   - unseen categories at inference time map to an all-zero row instead of
#     raising or silently guessing an integer code (safe "unknown" handling);
#   - encoding cannot leak information from the training split (there is
#     nothing to "fit").
# Slugs match the city keys already used in configs/config.yaml.
# ---------------------------------------------------------------------------
CITY_SLUGS: Dict[str, str] = {
    "москва": "moskva",
    "санкт-петербург": "spb",
    "екатеринбург": "ekaterinburg",
    "новосибирск": "novosibirsk",
    "казань": "kazan",
    "нижний новгород": "nizhniy_novgorod",
    "самара": "samara",
    "краснодар": "krasnodar",
}

BUILDING_TYPE_SLUGS: Dict[str, str] = {
    "panel": "panel",
    "brick": "brick",
    "monolith": "monolith",
    "block": "block",
}

# The training data (scripts/run_feature_engineering_real.py's synthetic
# generator, and any real scraped CIAN data normalised the same way) stores
# `building_type` as the English keys above. The public-facing API/dashboard
# vocabulary (api/schemas.py::BuildingTypeEnum) is Russian, for readability.
# These aliases translate the Russian labels to the same canonical keys so
# one encoding function serves both without duplicating category logic.
# "монолитно-кирпичный" and "другое" ("other") have NO dedicated training
# category (the 200-row synthetic generator only ever produces the four
# canonical types) — they are intentionally left unmapped so they fall
# through to the safe "unknown category" all-zero-row path rather than
# being guessed into a category the training data never actually contained.
BUILDING_TYPE_ALIASES: Dict[str, str] = {
    "панельный": "panel",
    "кирпичный": "brick",
    "монолитный": "monolith",
    "блочный": "block",
}

CITY_ONEHOT_COLUMNS: List[str] = [f"city_{slug}" for slug in CITY_SLUGS.values()]
BUILDING_TYPE_ONEHOT_COLUMNS: List[str] = [
    f"building_type_{slug}" for slug in BUILDING_TYPE_SLUGS.values()
]

# ---------------------------------------------------------------------------
# Default ML feature list
# ---------------------------------------------------------------------------

_BASE_NUMERIC_FEATURES: List[str] = (
    [
        "rooms",
        "total_area",
        "floor",
        "floors_total",
        "building_age",
        "floor_ratio",
        "is_top_floor",
        "is_first_floor",
        "log_area",
        "room_density",
        "rooms_x_area",
    ]
    + CITY_ONEHOT_COLUMNS
    + BUILDING_TYPE_ONEHOT_COLUMNS
)

_OPTIONAL_NUMERIC_FEATURES: List[str] = [
    "key_rate",
    "usd_rate",
    "inflation_rate",
    "rate_change_6m",
    # NOTE: price_per_sqm intentionally excluded — it is derived from the
    # target (price / area) and would constitute data leakage in training.
    # It is computed and kept in the DataFrame for EDA only.
]

_TARGET_COLUMN = "price"


def _normalize_categorical(series: pd.Series) -> pd.Series:
    """Lowercase/strip a categorical column, filling missing values with ''."""
    return series.fillna("").astype(str).str.strip().str.lower()


def one_hot_fixed_categories(
    series: pd.Series,
    category_to_slug: Dict[str, str],
    prefix: str,
    aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """One-hot encode *series* against a fixed, known set of categories.

    Unlike ``sklearn.OneHotEncoder``, nothing is *fit*: the category list is
    a hardcoded constant, so this function is fold-safe by construction (it
    cannot leak information from a train/test split) and handles unseen
    categories safely by producing an all-zero row instead of raising.

    Args:
        series: Raw categorical column (any case/whitespace).
        category_to_slug: Mapping of canonical category value (lowercased) to
            the column-name-safe slug used in the output column names.
        prefix: Column name prefix, e.g. ``"city"`` -> ``city_moskva``.
        aliases: Optional mapping of alternative labels (e.g. a different
            language) to a canonical key in *category_to_slug*. Values not
            found in either *category_to_slug* or *aliases* map to an
            all-zero row (safe "unknown category" handling).

    Returns:
        DataFrame with one ``{prefix}_{slug}`` int column (0/1) per category,
        indexed like *series*.
    """
    normalized = _normalize_categorical(series)
    if aliases:
        normalized = normalized.replace(aliases)
    out = pd.DataFrame(index=series.index)
    for category, slug in category_to_slug.items():
        out[f"{prefix}_{slug}"] = (normalized == category).astype(int)
    return out


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """Transform cleaned listings into an ML-ready feature matrix.

    Args:
        config: Optional configuration dict (currently unused but reserved
                for future feature-selection overrides).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._feature_list: Optional[List[str]] = None
        logger.info(
            "FeatureEngineer initialised (feature_schema_version=%s).", FEATURE_SCHEMA_VERSION
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature columns in *df* and return the augmented DataFrame.

        New columns created:
        - ``log_price``    — log1p(price)
        - ``log_area``     — log1p(total_area)
        - ``price_per_sqm`` — price / total_area  (if not already present)
        - ``room_density`` — rooms / total_area
        - ``floor_ratio``  — floor / floors_total  (if not already present)
        - ``is_first_floor``, ``is_top_floor`` (int 0/1; if not already present)
        - ``building_age`` (0 if year_built absent and not already present)
        - ``rooms_x_area`` — rooms × total_area interaction
        - ``city_<slug>`` — one fixed-category one-hot column per known city
        - ``building_type_<slug>`` — one fixed-category one-hot column per
          known building type

        Args:
            df: Cleaned DataFrame (output of :class:`~src.preprocessing.cleaner.DataCleaner`).

        Returns:
            DataFrame with additional feature columns.
        """
        df = df.copy()

        df = self._make_log_features(df)
        df = self._make_price_per_sqm(df)
        df = self._make_room_density(df)
        df = self._make_floor_features(df)
        df = self._make_building_age(df)
        df = self._make_interaction_features(df)
        df = self._encode_categoricals(df)

        logger.info("Feature creation done. DataFrame shape: %s", df.shape)
        return df

    def get_feature_list(self) -> List[str]:
        """Return the list of column names used as ML input features.

        The list is populated after the first call to
        :meth:`prepare_for_training` or can be built from
        :meth:`create_features` output columns.

        Returns:
            List of feature column names.
        """
        if self._feature_list is not None:
            return self._feature_list
        return self._build_default_feature_list()

    def prepare_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply feature engineering and split into X, y.

        Args:
            df: Cleaned DataFrame containing a ``price`` column.

        Returns:
            Tuple ``(X, y)`` where ``X`` is the feature matrix and ``y`` is
            the log-transformed price target.

        Raises:
            ValueError: If the ``price`` column is absent.
        """
        if _TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{_TARGET_COLUMN}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        df = self.create_features(df)

        # Drop rows where target is missing
        df = df.dropna(subset=[_TARGET_COLUMN])

        # Build final feature list from what actually exists
        feature_cols = self._select_available_features(df)
        self._feature_list = feature_cols

        X = df[feature_cols].copy()
        y = np.log1p(df[_TARGET_COLUMN])  # predict log-price; back-transform at inference

        # Fill any remaining NaN with a constant 0 — NOT a per-column median
        # computed from whatever DataFrame happens to be passed in here.
        # An earlier version computed a full-dataset median at this point,
        # which is a leakage bug when `df` spans both the train and
        # test/holdout split:
        # rooms/total_area/floor/floors_total must already be complete by
        # the time they reach here, via a
        # `~src.preprocessing.imputer.GroupMedianImputer` fit on TRAIN ONLY
        # and applied to this df before `prepare_for_training` is called.
        # Any NaN still present at this point means a feature column is
        # missing for every row in this dataset (e.g. `building_age` when
        # `year_built` was unavailable at the source) — a per-df median
        # would be undefined (NaN) in that case anyway. For a numeric
        # feature with zero variance across the whole dataset, any constant
        # fill is equivalent (a linear model's coefficient on it is
        # unidentifiable; a tree can't split on it) — 0 is chosen for
        # readability, not because it's a plausible true value.
        for col in X.columns:
            if X[col].isna().any():
                logger.warning(
                    "Feature '%s' still has NaN after upstream imputation — filling with 0. "
                    "This should only happen for a column missing for every row (e.g. no "
                    "'year_built' data at all), not for rooms/total_area/floor/floors_total, "
                    "which must be imputed upstream by GroupMedianImputer before this call.",
                    col,
                )
                X[col] = X[col].fillna(0)

        logger.info(
            "prepare_for_training: X=%s, y shape=%s, features=%s",
            X.shape,
            y.shape,
            feature_cols,
        )
        return X, y

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------

    def _make_log_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if _TARGET_COLUMN in df.columns:
            df["log_price"] = np.log1p(pd.to_numeric(df[_TARGET_COLUMN], errors="coerce"))
        if "total_area" in df.columns:
            df["log_area"] = np.log1p(pd.to_numeric(df["total_area"], errors="coerce"))
        return df

    def _make_price_per_sqm(self, df: pd.DataFrame) -> pd.DataFrame:
        if "price_per_sqm" not in df.columns:
            if "price" in df.columns and "total_area" in df.columns:
                safe_area = pd.to_numeric(df["total_area"], errors="coerce").replace(0, np.nan)
                price = pd.to_numeric(df["price"], errors="coerce")
                df["price_per_sqm"] = (price / safe_area).round(2)
        return df

    def _make_room_density(self, df: pd.DataFrame) -> pd.DataFrame:
        if "rooms" in df.columns and "total_area" in df.columns:
            rooms = pd.to_numeric(df["rooms"], errors="coerce")
            safe_area = pd.to_numeric(df["total_area"], errors="coerce").replace(0, np.nan)
            df["room_density"] = (rooms / safe_area).round(6)
        else:
            df["room_density"] = 0.0
        return df

    def _make_floor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        floor = pd.to_numeric(df.get("floor"), errors="coerce") if "floor" in df.columns else None
        floors_total = (
            pd.to_numeric(df.get("floors_total"), errors="coerce")
            if "floors_total" in df.columns
            else None
        )

        if "floor_ratio" not in df.columns:
            if floor is not None and floors_total is not None:
                safe_floors = floors_total.replace(0, np.nan)
                df["floor_ratio"] = (floor / safe_floors).round(4)
            else:
                df["floor_ratio"] = 0.5

        if "is_first_floor" not in df.columns:
            df["is_first_floor"] = (floor == 1).astype(int) if floor is not None else 0

        if "is_top_floor" not in df.columns:
            if floor is not None and floors_total is not None:
                df["is_top_floor"] = (floor == floors_total).astype(int)
            else:
                df["is_top_floor"] = 0

        return df

    def _make_building_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ``building_age`` relative to the *listing* year, not "now".

        For a historical listing, age should be measured as of when it was
        published (``date_published``), so the feature's meaning does not
        silently drift depending on when the pipeline happens to run. If
        ``date_published`` is unavailable, the current year is used as an
        explicit, logged fallback (this is the correct behaviour for a *new*
        prediction request being valued "as of today" — see
        ``Predictor._build_feature_vector``, which intentionally uses the
        current year for exactly that reason).

        In practice this method is usually a no-op: ``DataCleaner`` already
        computes ``building_age`` from listing-year in
        ``_engineer_derived()`` before this runs.
        """
        if "building_age" not in df.columns:
            if "year_built" in df.columns:
                if (
                    "date_published" in df.columns
                    and pd.to_datetime(df["date_published"], errors="coerce").notna().any()
                ):
                    listing_year = pd.to_datetime(df["date_published"], errors="coerce").dt.year
                    listing_year = listing_year.fillna(pd.Timestamp.now().year)
                else:
                    logger.warning(
                        "'date_published' unavailable — building_age computed "
                        "using the current year as a fallback, not the "
                        "listing year. This is expected for a live prediction "
                        "request, but would understate age for historical "
                        "training data missing dates."
                    )
                    listing_year = pd.Series(pd.Timestamp.now().year, index=df.index)

                df["building_age"] = listing_year - pd.to_numeric(df["year_built"], errors="coerce")
                df["building_age"] = df["building_age"].clip(lower=0).fillna(0)
            else:
                df["building_age"] = 0
        return df

    def _make_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        rooms = pd.to_numeric(df.get("rooms", pd.Series(dtype=float)), errors="coerce").fillna(0)
        area = pd.to_numeric(df.get("total_area", pd.Series(dtype=float)), errors="coerce").fillna(
            0
        )
        df["rooms_x_area"] = (rooms * area).round(2)
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode ``city`` and ``building_type`` against fixed categories.

        See :data:`CITY_SLUGS` / :data:`BUILDING_TYPE_SLUGS` for why these
        are hardcoded constants rather than a fitted encoder: it makes the
        encoding trivially fold-safe and reproducible at inference time
        without persisting an encoder object, and unseen categories map to
        an all-zero row instead of an arbitrary integer code.

        Idempotent by construction: any pre-existing ``city_*``/
        ``building_type_*`` one-hot columns (e.g. from loading a CSV that
        was already run through ``create_features`` once, such as a saved
        ``*.engineered.csv``) are dropped and recomputed from the raw
        ``city``/``building_type`` columns rather than concatenated
        alongside them — otherwise ``pd.concat`` would produce duplicate
        column names and any later ``df[some_onehot_col]`` lookup would
        return a DataFrame instead of a Series.
        """
        df = df.drop(
            columns=[
                c for c in CITY_ONEHOT_COLUMNS + BUILDING_TYPE_ONEHOT_COLUMNS if c in df.columns
            ]
        )

        if "city" in df.columns:
            city_ohe = one_hot_fixed_categories(df["city"], CITY_SLUGS, prefix="city")
            df = pd.concat([df, city_ohe], axis=1)
        else:
            for col in CITY_ONEHOT_COLUMNS:
                df[col] = 0

        if "building_type" in df.columns:
            bt_ohe = one_hot_fixed_categories(
                df["building_type"],
                BUILDING_TYPE_SLUGS,
                prefix="building_type",
                aliases=BUILDING_TYPE_ALIASES,
            )
            df = pd.concat([df, bt_ohe], axis=1)
        else:
            for col in BUILDING_TYPE_ONEHOT_COLUMNS:
                df[col] = 0

        return df

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _build_default_feature_list(self) -> List[str]:
        return _BASE_NUMERIC_FEATURES + _OPTIONAL_NUMERIC_FEATURES

    def _select_available_features(self, df: pd.DataFrame) -> List[str]:
        """Return those features from the default list that exist in *df*."""
        all_features = _BASE_NUMERIC_FEATURES + _OPTIONAL_NUMERIC_FEATURES
        available = [f for f in all_features if f in df.columns]
        missing = [f for f in all_features if f not in df.columns]
        if missing:
            logger.debug("Features absent from DataFrame (skipped): %s", missing)
        return available

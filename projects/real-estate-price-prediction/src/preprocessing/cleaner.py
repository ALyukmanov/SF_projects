"""
Data cleaning module for raw CIAN real-estate listings.

Removes duplicates, validates numeric ranges, imputes missing values and
engineers basic derived columns ready for feature engineering.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "min_price": 500_000,
    "max_price": 200_000_000,
    "min_area": 10,
    "max_area": 500,
    "min_rooms": 0,
    "max_rooms": 10,
    "min_floor": 1,
    "max_floor": 100,
}


# ---------------------------------------------------------------------------
# DataCleaner
# ---------------------------------------------------------------------------


class DataCleaner:
    """Clean raw real-estate DataFrames scraped from CIAN.

    Args:
        config: Optional dict overriding default bounds.  Keys match
                :data:`_DEFAULT_CONFIG`.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._cfg = {**_DEFAULT_CONFIG, **(config or {})}
        logger.info("DataCleaner initialised with config: %s", self._cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline on *df* and return a clean copy.

        Steps:
        1. Log initial shape & missing values.
        2. Remove duplicate URLs.
        3. Cast numeric columns to correct types.
        4. Drop rows outside valid price / area / floor ranges.
        5. Engineer derived columns that do NOT depend on rooms/floor/
           floors_total/total_area (price_per_sqm, building_age).
        6. Log final shape & summary stats.

        Deliberately does NOT impute missing ``rooms``/``total_area``/
        ``floor``/``floors_total`` and does NOT compute ``floor_ratio``/
        ``is_top_floor``/``is_first_floor`` here (an earlier version did
        both, using medians computed on the FULL dataset — a leak that was
        later fixed). Missing
        values in those four columns are left as NaN for the caller to fill
        via :class:`~src.preprocessing.imputer.GroupMedianImputer`, fit on
        a TRAIN split only, *after* the train/test split is made — then
        :class:`~src.features.feature_engineering.FeatureEngineer` computes
        ``floor_ratio``/flags/etc. from the now-complete columns.

        Args:
            df: Raw DataFrame (typically from a CIAN/restate scraper).

        Returns:
            Cleaned DataFrame (new object, original untouched). May still
            contain NaN in ``rooms``/``total_area``/``floor``/``floors_total``
            for genuinely missing source values.
        """
        df = df.copy()
        self._log_initial_stats(df)

        df = self._drop_duplicates(df)
        df = self._cast_types(df)
        df = self._filter_price(df)
        df = self._filter_area(df)
        df = self._filter_floor(df)
        df = self._engineer_derived(df)

        self._log_final_stats(df)
        return df

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        if "url" in df.columns:
            df = df.drop_duplicates(subset=["url"], keep="first")
        else:
            df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            logger.info("Removed %d duplicate rows.", removed)
        return df.reset_index(drop=True)

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce columns to their expected types; set unparseable values to NaN."""
        numeric_float = ["price", "total_area", "price_per_sqm"]
        numeric_int = ["rooms", "floor", "floors_total"]

        for col in numeric_float:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        for col in numeric_int:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Keep as float here to allow NaN; will cast after imputation
                df[col] = df[col].where(df[col].notna(), other=np.nan)

        # Parse date_published
        if "date_published" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["date_published"]
        ):
            df["date_published"] = pd.to_datetime(df["date_published"], errors="coerce")

        logger.debug("Type casting done.")
        return df

    def _filter_price(self, df: pd.DataFrame) -> pd.DataFrame:
        if "price" not in df.columns:
            return df
        before = len(df)
        mask = df["price"].between(self._cfg["min_price"], self._cfg["max_price"])
        df = df[mask | df["price"].isna()]
        removed = before - len(df)
        if removed:
            logger.info(
                "Removed %d rows outside price range [%s, %s].",
                removed,
                self._cfg["min_price"],
                self._cfg["max_price"],
            )
        return df.reset_index(drop=True)

    def _filter_area(self, df: pd.DataFrame) -> pd.DataFrame:
        if "total_area" not in df.columns:
            return df
        before = len(df)
        mask = df["total_area"].between(self._cfg["min_area"], self._cfg["max_area"])
        df = df[mask | df["total_area"].isna()]
        removed = before - len(df)
        if removed:
            logger.info(
                "Removed %d rows outside area range [%s, %s] sqm.",
                removed,
                self._cfg["min_area"],
                self._cfg["max_area"],
            )
        return df.reset_index(drop=True)

    def _filter_floor(self, df: pd.DataFrame) -> pd.DataFrame:
        if "floor" not in df.columns:
            return df
        before = len(df)
        mask = df["floor"].between(self._cfg["min_floor"], self._cfg["max_floor"])
        df = df[mask | df["floor"].isna()]
        removed = before - len(df)
        if removed:
            logger.info(
                "Removed %d rows with floor outside [%s, %s].",
                removed,
                self._cfg["min_floor"],
                self._cfg["max_floor"],
            )

        # Fix floors_total < floor
        if "floors_total" in df.columns:
            bad = (
                df["floors_total"].notna()
                & df["floor"].notna()
                & (df["floors_total"] < df["floor"])
            )
            if bad.any():
                df.loc[bad, "floors_total"] = df.loc[bad, "floor"]
                logger.info("Fixed %d rows where floors_total < floor.", bad.sum())

        return df.reset_index(drop=True)

    def _engineer_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns that do not depend on rooms/floor/floors_total/
        total_area imputation: ``price_per_sqm``, ``building_age``.

        ``floor_ratio``/``is_top_floor``/``is_first_floor`` (and the other
        rooms/area-derived features) are intentionally NOT computed here —
        see :meth:`clean`'s docstring. They are computed by
        :class:`~src.features.feature_engineering.FeatureEngineer` after a
        train-fit imputer has filled any missing floor/floors_total values,
        so they are never derived from a NaN.
        """
        # price_per_sqm (EDA-only column; excluded from the ML feature list
        # as a target-derived quantity, so leaving NaN here where total_area
        # is itself missing is not a leakage concern).
        if "price" in df.columns and "total_area" in df.columns:
            safe_area = df["total_area"].replace(0, np.nan)
            df["price_per_sqm"] = (df["price"] / safe_area).round(2)
            logger.debug("Computed 'price_per_sqm'.")

        # building_age — measured relative to the LISTING year
        # (date_published), not "now". Using the current wall-clock year here
        # would make the feature's meaning drift depending on when this
        # pipeline happens to run, which is wrong for historical training
        # data (a listing published in 2023 should have its age computed as
        # of 2023). If date_published is missing, we fall back to the
        # current year and log a warning so the approximation is visible.
        if "year_built" in df.columns:
            if "date_published" in df.columns and df["date_published"].notna().any():
                listing_year = pd.to_datetime(df["date_published"], errors="coerce").dt.year
                listing_year = listing_year.fillna(pd.Timestamp.now().year)
            else:
                logger.warning(
                    "'date_published' missing or entirely null — building_age "
                    "computed using the current year as a fallback, not the "
                    "listing year. Provenance metadata should record this."
                )
                listing_year = pd.Timestamp.now().year
            df["building_age"] = listing_year - pd.to_numeric(df["year_built"], errors="coerce")
            df["building_age"] = df["building_age"].clip(lower=0)
            logger.debug("Computed 'building_age' relative to listing year.")
        elif "building_age" not in df.columns:
            df["building_age"] = 0

        return df

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_initial_stats(self, df: pd.DataFrame) -> None:
        logger.info("=== DataCleaner: Initial dataset ===")
        logger.info("Shape: %d rows × %d columns", *df.shape)
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            logger.info("Missing values:\n%s", missing.to_string())
        for col in ["price", "total_area", "rooms"]:
            if col in df.columns:
                logger.info(
                    "%s: min=%.0f, max=%.0f, mean=%.0f",
                    col,
                    df[col].min() if df[col].notna().any() else float("nan"),
                    df[col].max() if df[col].notna().any() else float("nan"),
                    df[col].mean() if df[col].notna().any() else float("nan"),
                )

    def _log_final_stats(self, df: pd.DataFrame) -> None:
        logger.info("=== DataCleaner: Cleaned dataset ===")
        logger.info("Shape: %d rows × %d columns", *df.shape)
        remaining_missing = df.isnull().sum().sum()
        logger.info("Total remaining missing values: %d", remaining_missing)
        for col in ["price", "total_area", "price_per_sqm"]:
            if col in df.columns and df[col].notna().any():
                logger.info(
                    "%s: min=%.0f, max=%.0f, mean=%.0f, median=%.0f",
                    col,
                    df[col].min(),
                    df[col].max(),
                    df[col].mean(),
                    df[col].median(),
                )

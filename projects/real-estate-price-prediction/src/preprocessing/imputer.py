"""
Train-fit-only median imputer for the four listing fields that can be
missing at the raw-data stage: ``rooms``, ``total_area``, ``floor``,
``floors_total``.

An earlier version of this pipeline computed these medians on the FULL
engineered dataset before any train/test split existed — a real,
if small-magnitude, leak: holdout-row values informed the fill value used
for train rows (and vice versa). This module fixes that by making the
statistics an explicit fit/transform object: ``fit()`` must only ever be
called on a TRAIN split, and the exact same fitted values are then reused
for validation/holdout transforms and, once persisted in the saved model
artefact, for live inference — never recomputed from whatever data happens
to be passed in.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_IMPUTED_COLUMNS = ("rooms", "total_area", "floor", "floors_total")


class GroupMedianImputer:
    """Fit-on-train, apply-anywhere median imputer for listing fields.

    Mirrors the group-median fallback chain previously hardcoded inside
    ``DataCleaner._impute_missing`` (city -> global median for
    rooms/floor/floors_total; (city, rooms) -> city -> global for
    total_area), but as a stateful object: all statistics are computed once
    by :meth:`fit` on a caller-supplied DataFrame (which must be the TRAIN
    split only) and stored; :meth:`transform` never recomputes anything from
    its input, so it is safe to call on test/holdout data or a single live
    inference row without leaking that row's own value back into the fill
    value.
    """

    def __init__(self) -> None:
        self.fitted_: bool = False
        self.global_median_: Dict[str, float] = {}
        self.city_median_: Dict[str, Dict[str, float]] = {}
        self.city_rooms_median_total_area_: Dict[Any, float] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "GroupMedianImputer":
        """Compute fill statistics from *df*. Call on the TRAIN split only."""
        for col in _IMPUTED_COLUMNS:
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            self.global_median_[col] = float(values.median()) if values.notna().any() else 0.0
            if "city" in df.columns:
                by_city = values.groupby(df["city"]).median()
                self.city_median_[col] = {
                    str(city): float(v) for city, v in by_city.items() if pd.notna(v)
                }

        if "total_area" in df.columns and "city" in df.columns and "rooms" in df.columns:
            area = pd.to_numeric(df["total_area"], errors="coerce")
            # Round rooms to int for the group key so it matches exactly how
            # `transform()` looks this dict up (rooms is always an integer
            # count by the time total_area is filled — see _fill_total_area,
            # which fills rooms first). Using the raw (possibly float/NaN)
            # rooms column as the group key here would silently never match
            # at lookup time, degrading this to the coarser city-only
            # fallback without either crashing or leaking anything.
            rooms_int = pd.to_numeric(df["rooms"], errors="coerce").round()
            by_city_rooms = area.groupby([df["city"].astype(str), rooms_int]).median()
            self.city_rooms_median_total_area_ = {
                (city, int(rooms)): float(v)
                for (city, rooms), v in by_city_rooms.items()
                if pd.notna(v) and pd.notna(rooms)
            }

        self.fitted_ = True
        logger.info(
            "GroupMedianImputer fit on %d train rows. Global medians: %s",
            len(df),
            self.global_median_,
        )
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in *df* using statistics fitted on TRAIN.

        Safe to call on train (idempotent), test, holdout, or a single-row
        DataFrame built from a live prediction request — the fill values
        never depend on *df* itself, only on what :meth:`fit` recorded.
        """
        if not self.fitted_:
            raise RuntimeError("GroupMedianImputer.transform() called before fit().")

        df = df.copy()

        if "rooms" in df.columns:
            df["rooms"] = self._fill_series(df, "rooms")
            df["rooms"] = df["rooms"].round().astype("Int64")

        if "total_area" in df.columns:
            df["total_area"] = self._fill_total_area(df)

        if "floor" in df.columns:
            df["floor"] = self._fill_series(df, "floor")
            df["floor"] = df["floor"].round().astype("Int64")

        if "floors_total" in df.columns:
            df["floors_total"] = self._fill_series(df, "floors_total")
            df["floors_total"] = df["floors_total"].round().astype("Int64")

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_series(self, df: pd.DataFrame, col: str) -> pd.Series:
        values = pd.to_numeric(df[col], errors="coerce")
        if not values.isna().any():
            return values

        global_fallback = self.global_median_.get(col, 0.0)
        if "city" in df.columns and col in self.city_median_:
            city_fill = df["city"].astype(str).map(self.city_median_[col])
            values = values.fillna(city_fill)
        values = values.fillna(global_fallback)
        return values

    def _fill_total_area(self, df: pd.DataFrame) -> pd.Series:
        values = pd.to_numeric(df["total_area"], errors="coerce")
        if not values.isna().any():
            return values

        if "city" in df.columns and "rooms" in df.columns and self.city_rooms_median_total_area_:
            # Round rooms to int here too, to match fit()'s group key exactly
            # (see the comment in fit()).
            rooms_int = pd.to_numeric(df["rooms"], errors="coerce").round()
            keys = [
                (city, int(rooms)) if pd.notna(rooms) else None
                for city, rooms in zip(df["city"].astype(str), rooms_int)
            ]
            group_fill = pd.Series(
                [
                    self.city_rooms_median_total_area_.get(k) if k is not None else None
                    for k in keys
                ],
                index=df.index,
                dtype="float64",
            )
            values = values.fillna(group_fill)

        if "city" in df.columns and "total_area" in self.city_median_:
            city_fill = df["city"].astype(str).map(self.city_median_["total_area"])
            values = values.fillna(city_fill)

        values = values.fillna(self.global_median_.get("total_area", 0.0))
        return values

    # ------------------------------------------------------------------
    # Serialisation (persisted inside the model artefact for inference)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fitted": self.fitted_,
            "global_median": self.global_median_,
            "city_median": self.city_median_,
            "city_rooms_median_total_area": {
                # JSON needs string keys; tuple (city, rooms) -> "city||rooms"
                f"{city}||{rooms}": v
                for (city, rooms), v in self.city_rooms_median_total_area_.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GroupMedianImputer":
        obj = cls()
        if not data:
            return obj
        obj.fitted_ = bool(data.get("fitted", False))
        obj.global_median_ = dict(data.get("global_median", {}))
        obj.city_median_ = {k: dict(v) for k, v in data.get("city_median", {}).items()}
        obj.city_rooms_median_total_area_ = {}
        for key, v in data.get("city_rooms_median_total_area", {}).items():
            city, _, rooms = key.rpartition("||")
            # rooms is always stored as a plain int by fit() (see fit()'s
            # comment on why the group key is rounded), so this must always
            # parse cleanly — a ValueError here indicates the artefact was
            # written by an earlier, incompatible version of this class.
            obj.city_rooms_median_total_area_[(city, int(rooms))] = v
        return obj

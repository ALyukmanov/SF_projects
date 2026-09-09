"""
Geographic features from listing coordinates + OpenStreetMap POI.

For every listing that has ``latitude``/``longitude`` this module computes, per
city, the distance to the nearest POI of each kind and a few counts of POI
within a fixed radius. POI come from ``data/external/osm_poi.csv`` (built by
``scripts/prepare_osm_poi.py`` from OSM extracts).

Design notes
------------
* **Not a fitted transform.** The POI table is external reference data, like
  the macro indicators in ``economic_features.py``: there is nothing learned
  from the listings, so running this before or after the train/test split is
  equivalent and leakage-free. It is run once in
  ``run_feature_engineering_real.py`` so the coordinates never have to be
  re-joined later.

* **City-scoped.** A Moscow listing is only ever matched against Moscow POI
  and a Saint Petersburg listing only against Saint Petersburg POI — the two
  cities' POI sets are kept in separate trees, so a listing near the edge of
  one city can't accidentally match a POI from the other.

* **Distances are geographic.** Nearest-neighbour search runs on a
  ``BallTree`` with the haversine metric (coordinates in radians), then the
  angular distance is converted to metres with the mean Earth radius. No
  degree-space Euclidean distance anywhere.

* **The ``metro`` category is deduplicated to stations.** In OSM the raw
  ``metro`` rows are a mix of station nodes and (mostly unnamed) subway
  entrances/platform objects — ~1556 rows for Moscow's ~260 stations, ~353
  for Saint Petersburg's ~72. Every category is spatially clustered (points
  within a small category-specific radius are collapsed to one) so a station
  with eight mapped entrances counts once; the effect is largest for
  ``metro`` and ``supermarket``. The distance feature is therefore
  "distance to the nearest metro station", not "…nearest entrance".

* **Listings without coordinates keep the feature columns** (≈8% of the
  dataset). Distances are filled with a large sentinel and counts with 0, and
  ``has_coordinates`` is set to 0 so a model can tell "genuinely far from
  everything" apart from "location unknown". Nothing is dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_EARTH_RADIUS_M = 6_371_000.0

# Maps the ``city`` column value (as it appears in the listings + POI table)
# to a short internal key. Only these two cities have POI coverage today.
CITY_KEYS: Dict[str, str] = {
    "москва": "moscow",
    "санкт-петербург": "spb",
}

# The seven POI kinds carried in osm_poi.csv.
POI_CATEGORIES = (
    "metro",
    "rail_station",
    "school",
    "kindergarten",
    "hospital",
    "supermarket",
    "park",
)

# Points of the same category within this many metres of each other are
# treated as one object. Tuned to collapse OSM mapping artefacts (a station
# with many entrance nodes, a supermarket mapped as node + building polygon)
# without merging genuinely distinct neighbours.
_DEDUPE_RADIUS_M: Dict[str, float] = {
    "metro": 250.0,
    "rail_station": 150.0,
    "supermarket": 60.0,
    "school": 60.0,
    "kindergarten": 60.0,
    "hospital": 80.0,
    "park": 100.0,
}

# Distance value used for a listing we can't place (no coordinates, or a city
# with no POI table). Kept above the largest real nearest-POI distance seen in
# the data (~45 km for a far-suburb listing to the nearest metro station) so
# "location unknown" always sorts as at least as far as any placed listing,
# and pairs with has_coordinates=0.
NEAREST_SENTINEL_M = 50_000.0

# ---------------------------------------------------------------------------
# Feature names produced by GeoFeatureBuilder.transform().
# Kept as module constants so feature_engineering.py / tests can reference the
# exact set without hard-coding string lists in several places.
# ---------------------------------------------------------------------------
NEAREST_DISTANCE_FEATURES = [
    "nearest_metro_station_distance_m",
    "nearest_rail_station_distance_m",
    "nearest_school_distance_m",
    "nearest_kindergarten_distance_m",
    "nearest_hospital_distance_m",
    "nearest_supermarket_distance_m",
    "nearest_park_distance_m",
]

# category -> its nearest-distance column name
_DISTANCE_COL: Dict[str, str] = {
    "metro": "nearest_metro_station_distance_m",
    "rail_station": "nearest_rail_station_distance_m",
    "school": "nearest_school_distance_m",
    "kindergarten": "nearest_kindergarten_distance_m",
    "hospital": "nearest_hospital_distance_m",
    "supermarket": "nearest_supermarket_distance_m",
    "park": "nearest_park_distance_m",
}

# (category, radius_m, column name) for the count features. Deliberately a
# short, explainable set rather than every-category × every-radius.
_COUNT_SPECS = [
    ("supermarket", 500, "supermarkets_500m"),
    ("school", 500, "schools_500m"),
    ("kindergarten", 500, "kindergartens_500m"),
    ("hospital", 500, "hospitals_500m"),
    ("supermarket", 1000, "supermarkets_1000m"),
    ("school", 1000, "schools_1000m"),
    ("kindergarten", 1000, "kindergartens_1000m"),
    ("hospital", 1000, "hospitals_1000m"),
    ("park", 1000, "parks_1000m"),
    ("metro", 1000, "metro_stations_1000m"),
]
COUNT_FEATURES = [name for _, _, name in _COUNT_SPECS]

GEO_FEATURE_COLUMNS = ["has_coordinates"] + NEAREST_DISTANCE_FEATURES + COUNT_FEATURES

DEFAULT_POI_CSV = Path(__file__).resolve().parents[2] / "data" / "external" / "osm_poi.csv"


def _dedupe_points(coords_rad: np.ndarray, radius_m: float) -> np.ndarray:
    """Greedy spatial de-duplication.

    Walks the points in a fixed order and keeps a point only if no
    already-kept point lies within *radius_m*. Order is fixed (by latitude
    then longitude) so the result is deterministic.

    Args:
        coords_rad: (n, 2) array of [lat, lon] in radians.
        radius_m: merge radius in metres.

    Returns:
        (m, 2) array of the kept points, m <= n.
    """
    from sklearn.neighbors import BallTree

    if len(coords_rad) <= 1:
        return coords_rad
    order = np.lexsort((coords_rad[:, 1], coords_rad[:, 0]))
    tree = BallTree(coords_rad, metric="haversine")
    r = radius_m / _EARTH_RADIUS_M
    alive = np.ones(len(coords_rad), dtype=bool)
    for i in order:
        if not alive[i]:
            continue
        neighbours = tree.query_radius(coords_rad[i : i + 1], r=r)[0]
        alive[neighbours] = False
        alive[i] = True  # keep the representative itself
    return coords_rad[alive]


class _CityPOIIndex:
    """Per-category BallTree indexes for one city's POI."""

    def __init__(self, poi: pd.DataFrame, city_key: str) -> None:
        from sklearn.neighbors import BallTree

        self.city_key = city_key
        self._trees: Dict[str, BallTree] = {}
        self.counts: Dict[str, int] = {}
        for category in POI_CATEGORIES:
            sub = poi[poi["category"] == category]
            if sub.empty:
                continue
            coords = np.radians(sub[["lat", "lon"]].to_numpy(dtype=float))
            coords = _dedupe_points(coords, _DEDUPE_RADIUS_M.get(category, 50.0))
            self._trees[category] = BallTree(coords, metric="haversine")
            self.counts[category] = len(coords)

    def nearest_distances_m(self, coords_rad: np.ndarray, category: str) -> np.ndarray:
        tree = self._trees.get(category)
        if tree is None:
            return np.full(len(coords_rad), NEAREST_SENTINEL_M)
        dist, _ = tree.query(coords_rad, k=1)
        return dist[:, 0] * _EARTH_RADIUS_M

    def count_within_m(self, coords_rad: np.ndarray, category: str, radius_m: float) -> np.ndarray:
        tree = self._trees.get(category)
        if tree is None:
            return np.zeros(len(coords_rad), dtype=int)
        return tree.query_radius(
            coords_rad, r=radius_m / _EARTH_RADIUS_M, count_only=True
        )


class GeoFeatureBuilder:
    """Attach geo features to a listings DataFrame.

    Args:
        poi_csv: path to ``osm_poi.csv``. Defaults to the repo's
            ``data/external/osm_poi.csv``.
    """

    def __init__(self, poi_csv: Optional[Path | str] = None) -> None:
        self.poi_csv = Path(poi_csv) if poi_csv is not None else DEFAULT_POI_CSV
        self._indexes: Dict[str, _CityPOIIndex] = {}
        self._available = False
        if self.poi_csv.is_file():
            self._build_indexes()
        else:
            logger.warning(
                "POI table %s not found — geo features will be sentinel/zero only. "
                "Run scripts/prepare_osm_poi.py to build it.",
                self.poi_csv,
            )

    @property
    def available(self) -> bool:
        """True when a POI table was loaded and at least one city is indexed."""
        return self._available

    def poi_summary(self) -> Dict[str, Dict[str, int]]:
        """{city_key: {category: n_deduped_points}} — for sanity reporting."""
        return {k: dict(v.counts) for k, v in self._indexes.items()}

    def _build_indexes(self) -> None:
        poi = pd.read_csv(self.poi_csv)
        poi = poi.dropna(subset=["lat", "lon"])
        city_col = poi["city"].fillna("").astype(str).str.strip().str.lower()
        for raw_city, key in CITY_KEYS.items():
            city_poi = poi[city_col == raw_city]
            if city_poi.empty:
                continue
            self._indexes[key] = _CityPOIIndex(city_poi, key)
        self._available = bool(self._indexes)
        if self._available:
            logger.info(
                "GeoFeatureBuilder ready | cities=%s | deduped POI counts=%s",
                sorted(self._indexes),
                self.poi_summary(),
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return *df* with the geo feature columns added / overwritten.

        Requires ``city`` and (for a non-sentinel result) ``latitude`` /
        ``longitude`` columns. Rows whose city has no POI index, or whose
        coordinates are missing / out of range, get sentinel distances,
        zero counts and ``has_coordinates = 0``.
        """
        df = df.copy()
        n = len(df)

        # Start every geo column at its "unknown" value, then fill the rows we
        # can actually place.
        df["has_coordinates"] = 0
        for col in NEAREST_DISTANCE_FEATURES:
            df[col] = NEAREST_SENTINEL_M
        for col in COUNT_FEATURES:
            df[col] = 0

        if "latitude" not in df.columns or "longitude" not in df.columns:
            logger.warning("No latitude/longitude columns — all %d rows get sentinel geo features.", n)
            return self._finalise_dtypes(df)

        lat = pd.to_numeric(df["latitude"], errors="coerce")
        lon = pd.to_numeric(df["longitude"], errors="coerce")
        has_coord = lat.notna() & lon.notna() & lat.between(-90, 90) & lon.between(-180, 180)
        city_key = (
            df.get("city", pd.Series("", index=df.index))
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .map(CITY_KEYS)
        )

        placed = 0
        for key, index in self._indexes.items():
            mask = has_coord & (city_key == key)
            if not mask.any():
                continue
            rows = df.index[mask]
            coords_rad = np.radians(
                np.c_[lat.loc[rows].to_numpy(float), lon.loc[rows].to_numpy(float)]
            )
            df.loc[rows, "has_coordinates"] = 1
            for category, col in _DISTANCE_COL.items():
                df.loc[rows, col] = index.nearest_distances_m(coords_rad, category)
            for category, radius_m, col in _COUNT_SPECS:
                df.loc[rows, col] = index.count_within_m(coords_rad, category, float(radius_m))
            placed += len(rows)

        logger.info(
            "Geo features: %d/%d listings placed against a city POI index (%d without usable "
            "coordinates / unsupported city).",
            placed,
            n,
            n - placed,
        )
        return self._finalise_dtypes(df)

    @staticmethod
    def _finalise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        df["has_coordinates"] = df["has_coordinates"].astype(int)
        for col in NEAREST_DISTANCE_FEATURES:
            df[col] = df[col].astype(float).round(1)
        for col in COUNT_FEATURES:
            df[col] = df[col].astype(int)
        return df


def add_geo_features(
    df: pd.DataFrame,
    poi_csv: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Convenience wrapper: build a :class:`GeoFeatureBuilder` and transform *df*."""
    return GeoFeatureBuilder(poi_csv=poi_csv).transform(df)


def coverage_report(df: pd.DataFrame, by: Iterable[str] = ("city",)) -> pd.DataFrame:
    """Small sanity table: per-group coverage + distance quantiles.

    Expects *df* to already carry the geo columns.
    """
    rows = []
    group_cols = [c for c in by if c in df.columns]
    grouped = df.groupby(group_cols) if group_cols else [("all", df)]
    for key, g in grouped:
        rec = {"group": key, "n": len(g), "with_coords": int(g["has_coordinates"].sum())}
        placed = g[g["has_coordinates"] == 1]
        for col in NEAREST_DISTANCE_FEATURES:
            vals = placed[col]
            rec[f"{col}__median"] = float(vals.median()) if len(vals) else float("nan")
            rec[f"{col}__p90"] = float(vals.quantile(0.9)) if len(vals) else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)

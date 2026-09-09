"""Tests for src/data/split_pipeline.py — the 'location' split strategy and
the group_holdout_indices helper used by the tuning pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.split_pipeline import (  # noqa: E402
    group_holdout_indices,
    split_impute_featurize,
)


def _synthetic_listings(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # ~n/4 buildings, 1-6 listings each, each building one coordinate
    buildings = []
    bid = 0
    while sum(len(b) for b in buildings) < n:
        k = int(rng.integers(1, 7))
        lat = round(55.5 + rng.random() * 0.4, 4)
        lon = round(37.4 + rng.random() * 0.5, 4)
        rows = []
        for _ in range(k):
            area = float(rng.uniform(30, 110))
            rows.append(
                {
                    "url": f"u{bid}_{_}",
                    "address": f"bld {bid}",
                    "city": "Москва",
                    "rooms": int(rng.integers(1, 4)),
                    "floor": int(rng.integers(1, 20)),
                    "floors_total": 22,
                    "total_area": round(area, 1),
                    "year_built": 2005,
                    "building_type": "panel",
                    "price": round(area * rng.uniform(250_000, 400_000), -3),
                    "latitude": lat,
                    "longitude": lon,
                }
            )
        buildings.append(rows)
        bid += 1
    flat = [r for b in buildings for r in b][:n]
    return pd.DataFrame(flat)


class TestLocationSplitStrategy:
    def test_no_building_straddles_the_holdout(self):
        df = _synthetic_listings()
        res = split_impute_featurize(df, split_strategy="location", random_state=42)
        train_keys = set(zip(res.train_raw["latitude"], res.train_raw["longitude"]))
        test_keys = set(zip(res.test_raw["latitude"], res.test_raw["longitude"]))
        assert not (train_keys & test_keys)
        assert res.split_strategy == "location_grouped_80_20_random_state_42"

    def test_group_split_is_more_permissive_than_location(self):
        """The near-dup 'group' split lets same-building flats straddle; the
        'location' split must not."""
        df = _synthetic_listings()
        grp = split_impute_featurize(df, split_strategy="group", random_state=42)
        grp_train = set(zip(grp.train_raw["latitude"], grp.train_raw["longitude"]))
        grp_test = set(zip(grp.test_raw["latitude"], grp.test_raw["longitude"]))
        # with many multi-listing buildings, the plain group split almost
        # certainly straddles at least one building
        assert grp_train & grp_test

    def test_group_holdout_indices_matches_split_impute_featurize(self):
        df = _synthetic_listings().reset_index(drop=True)
        tr, te, groups = group_holdout_indices(df, "location", test_size=0.2, random_state=42)
        res = split_impute_featurize(df, split_strategy="location", random_state=42)
        assert set(df.iloc[te]["url"]) == set(res.test_raw["url"])
        # groups vector aligns with df and never crosses
        g = pd.Series(groups)
        assert not (set(g.iloc[tr]) & set(g.iloc[te]))

    def test_random_strategy_returns_none_groups(self):
        df = _synthetic_listings()
        tr, te, groups = group_holdout_indices(df, "random", random_state=42)
        assert groups is None
        assert len(tr) + len(te) == len(df)


class TestLocationSplitOnCleanedCsvShape:
    """Smoke test against the real cleaned CSV if it is present."""

    def test_real_cleaned_csv_location_split_has_geo_features(self):
        csv = _PROJECT_ROOT / "data" / "processed" / "real_estate_cleaned.csv"
        if not csv.is_file():
            pytest.skip("real_estate_cleaned.csv not present")
        df = pd.read_csv(csv)
        res = split_impute_featurize(df, split_strategy="location", random_state=42)
        assert "has_coordinates" in res.feature_names
        assert "nearest_metro_station_distance_m" in res.feature_names
        assert len(res.X_train) + len(res.X_test) <= len(df)
        # no building straddles
        tr = set(zip(res.train_raw["latitude"].round(4), res.train_raw["longitude"].round(4)))
        te = set(zip(res.test_raw["latitude"].round(4), res.test_raw["longitude"].round(4)))
        tr.discard((np.nan, np.nan))
        te.discard((np.nan, np.nan))
        assert not (tr & te)

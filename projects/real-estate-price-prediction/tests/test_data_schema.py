"""Tests for src/data/schema.py — the data contract validator."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.schema import (  # noqa: E402
    SchemaValidationError,
    build_split_groups,
    find_near_duplicate_candidates,
    validate_listings_df,
)


def _valid_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [5_000_000, 8_000_000, 12_000_000][:n],
            "total_area": [40.0, 60.0, 90.0][:n],
            "floor": [3, 5, 10][:n],
            "floors_total": [9, 12, 16][:n],
            "city": ["москва", "самара", "казань"][:n],
            "rooms": [1, 2, 3][:n],
            "year_built": [2005, 1998, 2015][:n],
            "building_type": ["panel", "brick", "monolith"][:n],
            "url": ["https://cian.ru/1/", "https://cian.ru/2/", "https://cian.ru/3/"][:n],
        }
    )


class TestValidData:
    def test_valid_df_has_no_errors(self):
        report = validate_listings_df(_valid_df())
        assert report.is_valid
        assert report.errors == []

    def test_missing_required_column_is_an_error(self):
        df = _valid_df().drop(columns=["floor"])
        report = validate_listings_df(df)
        assert not report.is_valid
        assert any("floor" in e for e in report.errors)


class TestNumericRangeGuards:
    def test_price_must_be_positive(self):
        df = _valid_df()
        df.loc[0, "price"] = -100
        report = validate_listings_df(df)
        assert not report.is_valid

    def test_total_area_must_be_positive(self):
        df = _valid_df()
        df.loc[0, "total_area"] = 0
        report = validate_listings_df(df)
        assert not report.is_valid

    def test_floor_below_one_is_an_error(self):
        df = _valid_df()
        df.loc[0, "floor"] = 0
        report = validate_listings_df(df)
        assert not report.is_valid

    def test_floors_total_less_than_floor_is_an_error(self):
        df = _valid_df()
        df.loc[0, "floors_total"] = 1
        df.loc[0, "floor"] = 5
        report = validate_listings_df(df)
        assert not report.is_valid
        assert any("floors_total" in e for e in report.errors)

    def test_year_built_out_of_range_is_an_error(self):
        df = _valid_df()
        df.loc[0, "year_built"] = 1500
        report = validate_listings_df(df)
        assert not report.is_valid

    def test_rooms_out_of_range_is_an_error(self):
        df = _valid_df()
        df.loc[0, "rooms"] = 99
        report = validate_listings_df(df)
        assert not report.is_valid


class TestReferentialChecks:
    def test_unknown_city_is_a_warning_not_an_error(self):
        df = _valid_df()
        df.loc[0, "city"] = "атлантида"
        report = validate_listings_df(df)
        assert report.is_valid, "Unknown city should be a warning, not a hard error."
        assert any("city" in w for w in report.warnings)

    def test_duplicate_url_is_an_error(self):
        df = _valid_df()
        df.loc[1, "url"] = df.loc[0, "url"]
        report = validate_listings_df(df)
        assert not report.is_valid
        assert any("duplicate" in e.lower() for e in report.errors)

    def test_anomalous_price_per_sqm_is_a_warning(self):
        df = _valid_df()
        df.loc[0, "price"] = 100_000
        df.loc[0, "total_area"] = 500.0  # 200 RUB/sqm — absurdly cheap
        report = validate_listings_df(df)
        assert any("price_per_sqm" in w for w in report.warnings)


class TestNearDuplicateCandidates:
    """Data Quality Gate 2.0 -- near-duplicate detection (see schema.py's
    find_near_duplicate_candidates docstring for the matching rule). These
    NEVER auto-delete rows -- only ever report candidates."""

    def _df(self, **overrides) -> pd.DataFrame:
        base = {
            "url": [f"https://www.restate.ru/base/{i}.html" for i in range(4)],
            "address": ["ул. Тестовая, д. 1"] * 4,
            "rooms": [1, 1, 1, 1],
            "floor": [5, 5, 5, 5],
            "price": [10_000_000, 10_100_000, 10_000_000, 10_000_000],
            "total_area": [40.0, 40.0, 40.0, 40.0],
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def test_same_address_rooms_floor_close_price_area_is_flagged(self):
        df = self._df(price=[10_000_000, 10_100_000, 30_000_000, 30_000_000])
        cands = find_near_duplicate_candidates(df)
        # Rows 0,1 (within 2% price) form one group; rows 2,3 (identical) form another.
        assert set(cands["group_id"]) == {1, 2}
        assert len(cands) == 4

    def test_same_address_but_different_price_is_not_flagged(self):
        df = self._df(price=[10_000_000, 20_000_000, 5_000_000, 40_000_000])
        cands = find_near_duplicate_candidates(df)
        assert cands.empty

    def test_same_address_text_but_different_rooms_is_not_flagged(self):
        # Every row has a distinct rooms value, so no two rows share a
        # (address, rooms, floor) group key despite identical address text.
        df = self._df(rooms=[1, 2, 3, 4])
        cands = find_near_duplicate_candidates(df)
        # Address alone is not enough -- rooms differ, so no pair matches.
        assert cands.empty

    def test_price_tolerance_boundary_respected(self):
        # Exactly at 2% is inside tolerance; just past it is not.
        cands_inside = find_near_duplicate_candidates(
            pd.DataFrame(
                {
                    "url": ["a", "b"],
                    "address": ["ул. Тестовая, д. 1"] * 2,
                    "rooms": [1, 1],
                    "floor": [5, 5],
                    "price": [10_000_000, 10_200_000],
                    "total_area": [40.0, 40.0],
                }
            )
        )
        assert len(cands_inside) == 2
        cands_outside = find_near_duplicate_candidates(
            pd.DataFrame(
                {
                    "url": ["a", "b"],
                    "address": ["ул. Тестовая, д. 1"] * 2,
                    "rooms": [1, 1],
                    "floor": [5, 5],
                    "price": [10_000_000, 10_300_000],
                    "total_area": [40.0, 40.0],
                }
            )
        )
        assert cands_outside.empty

    def test_missing_required_column_returns_empty_not_error(self):
        df = self._df().drop(columns=["address"])
        cands = find_near_duplicate_candidates(df)
        assert cands.empty

    def test_empty_dataframe_returns_empty(self):
        cands = find_near_duplicate_candidates(pd.DataFrame())
        assert cands.empty

    def test_distinct_listings_produce_no_candidates(self):
        df = pd.DataFrame(
            {
                "url": ["https://x/1", "https://x/2"],
                "address": ["ул. Первая, д. 1", "ул. Вторая, д. 2"],
                "rooms": [1, 3],
                "floor": [2, 9],
                "price": [8_000_000, 25_000_000],
                "total_area": [35.0, 90.0],
            }
        )
        cands = find_near_duplicate_candidates(df)
        assert cands.empty

    def test_three_way_mutual_group_all_clustered_together(self):
        df = pd.DataFrame(
            {
                "url": ["a", "b", "c"],
                "address": ["ул. Тестовая, д. 1"] * 3,
                "rooms": [1, 1, 1],
                "floor": [5, 5, 5],
                "price": [10_000_000, 10_050_000, 10_100_000],
                "total_area": [40.0, 40.0, 40.0],
            }
        )
        cands = find_near_duplicate_candidates(df)
        assert len(cands) == 3
        assert cands["group_id"].nunique() == 1

    def test_rows_with_nan_price_or_area_are_skipped_not_crashed(self):
        df = pd.DataFrame(
            {
                "url": ["a", "b", "c"],
                "address": ["ул. Тестовая, д. 1"] * 3,
                "rooms": [1, 1, 1],
                "floor": [5, 5, 5],
                "price": [10_000_000, None, 10_050_000],
                "total_area": [40.0, 40.0, None],
            }
        )
        # Must not raise, and rows with a NaN price/area must never be
        # reported as matching (there is nothing to compare).
        cands = find_near_duplicate_candidates(df)
        assert set(cands["url"]) <= {"a", "c"}

    def test_validate_listings_df_surfaces_near_duplicate_warning(self):
        df = self._df(price=[10_000_000, 10_100_000, 10_000_000, 10_100_000])
        df["floors_total"] = 9
        df["city"] = "москва"
        report = validate_listings_df(df)
        assert any("NEAR-DUPLICATE" in w for w in report.warnings)
        assert report.is_valid, "Near-duplicate candidates are a warning, not an error."


class TestBuildSplitGroups:
    """build_split_groups() -- the grouping key for GroupShuffleSplit/
    GroupKFold, so near-duplicate listings never split across train/test."""

    def _df(self, **overrides) -> pd.DataFrame:
        base = {
            "url": [f"https://www.restate.ru/base/{i}.html" for i in range(6)],
            "address": ["ул. Тестовая, д. 1"] * 4 + ["ул. Другая, д. 2"] * 2,
            "rooms": [1, 1, 1, 1, 2, 2],
            "floor": [5, 5, 5, 5, 3, 3],
            "price": [10_000_000, 10_100_000, 30_000_000, 30_000_000, 8_000_000, 20_000_000],
            "total_area": [40.0, 40.0, 60.0, 60.0, 35.0, 35.0],
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def test_near_duplicate_rows_share_a_group_id(self):
        df = self._df()
        groups = build_split_groups(df)
        # Rows 0,1 are near-duplicates (same address/rooms/floor, price within 2%).
        assert groups.iloc[0] == groups.iloc[1]

    def test_non_duplicate_rows_get_distinct_group_ids(self):
        df = self._df()
        groups = build_split_groups(df)
        # Rows 4,5 share address/rooms/floor but price differs by >2% -- not a match.
        assert groups.iloc[4] != groups.iloc[5]

    def test_group_count_matches_near_duplicate_structure(self):
        df = self._df()
        groups = build_split_groups(df)
        # Expect: {row0,row1} one group, {row2,row3} one group, row4 own group, row5 own group = 4 unique groups.
        assert groups.nunique() == 4

    def test_no_near_duplicates_returns_all_unique_groups(self):
        df = pd.DataFrame(
            {
                "url": ["a", "b", "c"],
                "address": ["ул. Первая, 1", "ул. Вторая, 2", "ул. Третья, 3"],
                "rooms": [1, 2, 3],
                "floor": [2, 5, 9],
                "price": [8_000_000, 15_000_000, 25_000_000],
                "total_area": [35.0, 55.0, 90.0],
            }
        )
        groups = build_split_groups(df)
        assert groups.nunique() == 3

    def test_missing_url_column_falls_back_to_positional_index(self):
        df = self._df().drop(columns=["url"])
        groups = build_split_groups(df)
        assert len(groups) == len(df)
        assert groups.nunique() == 4  # same grouping structure, just no url-based labels

    def test_groupshufflesplit_never_splits_a_near_duplicate_group(self):
        from sklearn.model_selection import GroupShuffleSplit

        df = self._df()
        groups = build_split_groups(df)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.34, random_state=0)
        train_idx, test_idx = next(gss.split(df, groups=groups))
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert not (train_groups & test_groups)


class TestBuildLocationGroups:
    """build_location_groups() -- the *building-level* grouping key: rows in
    the same building (same rounded coordinate) OR near-duplicate listings
    must share a group so they never straddle a train/holdout split."""

    def _df(self, **overrides) -> pd.DataFrame:
        base = {
            "url": [f"u{i}" for i in range(6)],
            "address": ["A"] * 3 + ["B"] * 2 + ["C"],
            "rooms": [1, 2, 3, 1, 2, 1],
            "floor": [2, 4, 6, 3, 5, 7],
            "price": [8e6, 12e6, 20e6, 9e6, 15e6, 7e6],
            "total_area": [35.0, 50.0, 80.0, 38.0, 55.0, 33.0],
            # rows 0,1,2 share a coordinate (one building); 3,4 another; 5 alone
            "latitude": [55.7501, 55.7501, 55.75012, 55.8000, 55.8000, 55.9000],
            "longitude": [37.6200, 37.6200, 37.62001, 37.7000, 37.7000, 37.8000],
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def test_same_coordinate_rows_share_a_group(self):
        from src.data.schema import build_location_groups

        g = build_location_groups(self._df())
        assert g.iloc[0] == g.iloc[1] == g.iloc[2]  # building A
        assert g.iloc[3] == g.iloc[4]  # building B
        assert g.iloc[5] not in {g.iloc[0], g.iloc[3]}  # lone building C

    def test_group_count_is_building_level(self):
        from src.data.schema import build_location_groups

        g = build_location_groups(self._df())
        assert g.nunique() == 3

    def test_rows_without_coordinates_fall_back_to_near_dup(self):
        from src.data.schema import build_location_groups

        df = self._df().drop(columns=["latitude", "longitude"])
        g = build_location_groups(df)
        # no coords, no near-dups here -> every row its own group
        assert g.nunique() == len(df)

    def test_near_duplicate_without_coords_still_grouped(self):
        from src.data.schema import build_location_groups

        df = pd.DataFrame(
            {
                "url": ["a", "b", "c"],
                "address": ["ул. Тест, 1", "ул. Тест, 1", "ул. Иная, 2"],
                "rooms": [2, 2, 3],
                "floor": [5, 5, 9],
                "price": [10_000_000, 10_050_000, 25_000_000],
                "total_area": [50.0, 50.0, 90.0],
                "latitude": [None, None, None],
                "longitude": [None, None, None],
            }
        )
        g = build_location_groups(df)
        assert g.iloc[0] == g.iloc[1]  # near-dup pair
        assert g.iloc[2] != g.iloc[0]

    def test_groupshufflesplit_never_splits_a_building(self):
        from sklearn.model_selection import GroupShuffleSplit

        from src.data.schema import build_location_groups

        df = self._df()
        g = build_location_groups(df)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=1)
        tr, te = next(gss.split(df, groups=g))
        assert not (set(g.iloc[tr]) & set(g.iloc[te]))


class TestStrictMode:
    def test_strict_raises_on_error(self):
        df = _valid_df()
        df.loc[0, "price"] = -1
        with pytest.raises(SchemaValidationError):
            validate_listings_df(df, strict=True)

    def test_strict_does_not_raise_on_warning_only(self):
        df = _valid_df()
        df.loc[0, "city"] = "атлантида"
        # Should not raise — unknown city is a warning only.
        report = validate_listings_df(df, strict=True)
        assert report.is_valid

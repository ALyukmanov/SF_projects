"""Tests for data preprocessing modules."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path (conftest also does this, but be safe)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.feature_engineering import CITY_ONEHOT_COLUMNS, FeatureEngineer
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.economic_features import EconomicFeatureEngineer

# ---------------------------------------------------------------------------
# Helper: minimal valid row
# ---------------------------------------------------------------------------


def _make_valid_row(
    price: float = 5_000_000,
    rooms: int = 2,
    total_area: float = 55.0,
    floor: int = 5,
    floors_total: int = 10,
    city: str = "москва",
    url: str = "https://cian.ru/sale/flat/1/",
    year_built: int = 2010,
    building_type: str = "panel",
    date_published: str = "2024-06-01",
) -> dict:
    return {
        "price": price,
        "rooms": rooms,
        "total_area": total_area,
        "floor": floor,
        "floors_total": floors_total,
        "city": city,
        "url": url,
        "year_built": year_built,
        "building_type": building_type,
        "date_published": date_published,
        "scraped_at": date_published,
    }


# ===========================================================================
# DataCleaner tests
# ===========================================================================


class TestDataCleaner:
    def test_cleaner_removes_duplicates(self):
        """Rows with the same URL should be deduplicated; keep first."""
        rows = [
            _make_valid_row(price=5_000_000, url="https://cian.ru/1/"),
            _make_valid_row(price=5_500_000, url="https://cian.ru/1/"),  # duplicate URL
            _make_valid_row(price=6_000_000, url="https://cian.ru/2/"),
        ]
        df = pd.DataFrame(rows)
        cleaner = DataCleaner()
        result = cleaner.clean(df)

        assert len(result) == 2, "Duplicate URL row should have been removed."
        # The first occurrence is kept
        assert result.loc[result["url"] == "https://cian.ru/1/", "price"].iloc[0] == 5_000_000

    def test_cleaner_price_range(self):
        """Rows with prices outside [500 000, 200 000 000] should be dropped."""
        rows = [
            _make_valid_row(price=100),  # too low
            _make_valid_row(price=5_000_000, url="https://cian.ru/ok/"),  # valid
            _make_valid_row(price=999_999_999, url="https://cian.ru/hi/"),  # too high
        ]
        df = pd.DataFrame(rows)
        cleaner = DataCleaner()
        result = cleaner.clean(df)

        assert len(result) == 1, "Only one row should survive price range filter."
        assert result.iloc[0]["price"] == 5_000_000

    def test_cleaner_derived_columns(self, sample_raw_df):
        """DataCleaner must produce price_per_sqm and building_age.

        floor_ratio/is_first_floor/is_top_floor are deliberately NOT produced
        by DataCleaner: they depend on rooms/floor/
        floors_total, which may still be NaN at this stage for genuinely
        missing source rows, and must only be derived AFTER a train-fit
        GroupMedianImputer has filled them (see src/preprocessing/imputer.py).
        FeatureEngineer.create_features computes them instead — see
        TestFeatureEngineer.test_feature_engineer_creates_features.
        """
        cleaner = DataCleaner()
        result = cleaner.clean(sample_raw_df)

        assert "price_per_sqm" in result.columns
        assert "building_age" in result.columns
        for col in ("floor_ratio", "is_first_floor", "is_top_floor"):
            assert col not in result.columns, (
                f"DataCleaner must not compute '{col}' — it depends on rooms/floor/"
                "floors_total, which may be unimputed NaN at this stage."
            )

        # price_per_sqm must be positive where price and area are valid
        valid = result.dropna(subset=["price_per_sqm"])
        assert (valid["price_per_sqm"] > 0).all(), "price_per_sqm should be positive."

    def test_cleaner_does_not_impute_missing_rooms_floor(self):
        """DataCleaner must leave rooms/total_area/floor/floors_total as NaN
        for genuinely missing source rows — imputing them here (the earlier
        behaviour) computes a median from the full dataset before any
        train/test split exists, which leaks holdout rows into the value
        used to fill train rows (and vice versa). See
        src/preprocessing/imputer.py::GroupMedianImputer, which must be fit
        on a train split only, downstream of this method.
        """
        rows = [
            _make_valid_row(url="https://cian.ru/a/"),
            _make_valid_row(url="https://cian.ru/b/", rooms=None, floor=None, floors_total=None),
        ]
        df = pd.DataFrame(rows)
        result = DataCleaner().clean(df)

        missing_row = result[result["url"] == "https://cian.ru/b/"].iloc[0]
        assert pd.isna(missing_row["rooms"])
        assert pd.isna(missing_row["floor"])
        assert pd.isna(missing_row["floors_total"])


# ===========================================================================
# EconomicFeatureEngineer tests
# ===========================================================================


class TestEconomicFeatureEngineer:
    def test_economic_features_fallback(self, sample_raw_df):
        """EconomicFeatureEngineer(use_api=False) must add key_rate, usd_rate, inflation_rate."""
        cleaner = DataCleaner()
        df = cleaner.clean(sample_raw_df)

        eng = EconomicFeatureEngineer(use_api=False)
        result = eng.add_economic_features(df)

        for col in ("key_rate", "usd_rate", "inflation_rate", "rate_change_6m"):
            assert col in result.columns, f"Expected economic column '{col}' not found."

        # Values should be non-null and sensible
        assert result["key_rate"].notna().all(), "key_rate should have no NaN."
        assert result["usd_rate"].notna().all(), "usd_rate should have no NaN."
        assert result["inflation_rate"].notna().all(), "inflation_rate should have no NaN."
        assert (result["key_rate"] > 0).all(), "key_rate should be positive."
        assert (result["usd_rate"] > 0).all(), "usd_rate should be positive."
        assert (result["inflation_rate"] > 0).all(), "inflation_rate should be positive."

    def test_economic_features_no_api_calls(self, sample_raw_df, monkeypatch):
        """use_api=False must not call the CBR endpoints."""
        import requests as _requests

        called = []

        def _fake_get(*args, **kwargs):
            called.append(args)
            raise AssertionError("HTTP request was made despite use_api=False")

        monkeypatch.setattr(_requests, "get", _fake_get)

        cleaner = DataCleaner()
        df = cleaner.clean(sample_raw_df)
        eng = EconomicFeatureEngineer(use_api=False)
        eng.add_economic_features(df)  # should not raise

        assert len(called) == 0, "No HTTP calls should be made when use_api=False."

    def test_economic_features_without_date_column(self):
        """When date_published is absent, the current year's rates should be used."""
        df = pd.DataFrame(
            {
                "price": [5_000_000, 8_000_000],
                "rooms": [2, 3],
                "total_area": [55.0, 80.0],
                "city": ["москва", "санкт-петербург"],
            }
        )
        eng = EconomicFeatureEngineer(use_api=False)
        result = eng.add_economic_features(df)

        assert "key_rate" in result.columns
        assert result["key_rate"].notna().all()


# ===========================================================================
# FeatureEngineer tests
# ===========================================================================


class TestFeatureEngineer:
    def test_feature_engineer_creates_features(self, sample_cleaned_df):
        """FeatureEngineer.create_features must produce the expected columns."""
        eng = FeatureEngineer()
        result = eng.create_features(sample_cleaned_df)

        expected_cols = [
            "log_area",
            "room_density",
            "rooms_x_area",
            "floor_ratio",
            "is_first_floor",
            "is_top_floor",
            "building_age",
        ] + CITY_ONEHOT_COLUMNS
        for col in expected_cols:
            assert col in result.columns, f"Expected feature column '{col}' not found."

    def test_feature_engineer_log_area_positive(self, sample_cleaned_df):
        """log_area should be finite and positive for valid areas."""
        eng = FeatureEngineer()
        result = eng.create_features(sample_cleaned_df)
        valid = result.dropna(subset=["log_area"])
        assert (valid["log_area"] > 0).all(), "log_area should be positive for area > 0."

    def test_feature_engineer_city_onehot_is_binary_and_exclusive(self, sample_cleaned_df):
        """Each row's city one-hot columns should sum to exactly 1 (city is always known)."""
        eng = FeatureEngineer()
        result = eng.create_features(sample_cleaned_df)
        for col in CITY_ONEHOT_COLUMNS:
            assert col in result.columns, f"Expected one-hot column '{col}'."
            assert result[col].isin([0, 1]).all(), f"{col} should be binary."
        assert (
            result[CITY_ONEHOT_COLUMNS].sum(axis=1) == 1
        ).all(), "Each row should have exactly one city one-hot column set to 1."

    def test_feature_engineer_create_features_is_idempotent(self, sample_cleaned_df):
        """Running create_features twice (e.g. loading an already-engineered CSV
        and re-running the pipeline on it) must not duplicate one-hot columns.

        Regression test: a bug in _encode_categoricals used to pd.concat new
        city_*/building_type_* one-hot columns onto a DataFrame that already
        had them (from a prior create_features call), producing duplicate
        column names. Any later `df[some_onehot_col]` lookup then returned a
        DataFrame instead of a Series, breaking `prepare_for_training` with
        `ValueError: The truth value of a Series is ambiguous` — this is
        exactly what happens when scripts/run_model_training_real.py loads a
        CSV that scripts/run_feature_engineering_real.py already engineered.
        """
        eng = FeatureEngineer()
        once = eng.create_features(sample_cleaned_df)
        twice = eng.create_features(once)
        assert not twice.columns.duplicated().any(), (
            f"Duplicate columns after re-running create_features: "
            f"{list(twice.columns[twice.columns.duplicated()])}"
        )
        for col in CITY_ONEHOT_COLUMNS:
            assert (twice[col] == once[col]).all()

    def test_feature_engineer_rooms_x_area(self, sample_cleaned_df):
        """rooms_x_area should equal rooms * total_area (approximately)."""
        eng = FeatureEngineer()
        result = eng.create_features(sample_cleaned_df)
        expected = (
            pd.to_numeric(result["rooms"], errors="coerce").fillna(0)
            * pd.to_numeric(result["total_area"], errors="coerce").fillna(0)
        ).round(2)
        pd.testing.assert_series_equal(
            result["rooms_x_area"].round(2).reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_prepare_for_training_returns_xy(self, sample_cleaned_df):
        """prepare_for_training must return (X, y) with matching lengths."""
        eng = FeatureEngineer()
        X, y = eng.prepare_for_training(sample_cleaned_df)
        assert len(X) == len(y), "X and y must have the same number of rows."
        assert X.shape[1] > 0, "Feature matrix must have at least one column."
        assert (y > 0).all(), "Log-transformed prices should be positive (price > 0)."

    def test_prepare_for_training_handles_all_nan_feature_column(self, sample_cleaned_df):
        """A feature that's entirely missing for the whole dataset (e.g.
        `building_age` when `year_built` was never collected -- restate.ru's
        search-results pages never carry it) must not crash training.

        Regression test for a real bug found on real scraped data: a
        per-column median-fill (`X[col].fillna(X[col].median())`) is a
        no-op when the whole column is NaN (median of all-NaN is NaN too),
        leaving NaN in X and crashing sklearn's fit() with
        `ValueError: Input X contains NaN`. Fixed by falling back to 0 for
        exactly this case.
        """
        from src.preprocessing.cleaner import DataCleaner

        df = sample_cleaned_df.copy()
        df["year_built"] = None
        df = DataCleaner().clean(df)
        assert df["building_age"].isna().all(), "Test setup: building_age should be all-NaN here."

        # Note: this project's logger sets propagate=False (see
        # src/utils/logger.py), so pytest's `caplog` cannot observe its
        # warnings -- the actual warning was verified by inspection/manual
        # runs rather than here.
        eng = FeatureEngineer()
        X, y = eng.prepare_for_training(df)

        assert not X.isna().any().any(), "No NaN should survive into the feature matrix."
        assert (X["building_age"] == 0).all()

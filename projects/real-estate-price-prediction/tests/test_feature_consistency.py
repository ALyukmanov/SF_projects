"""
Regression tests guarding the contract between the trained-model feature
order used in ``src.inference.predictor`` and the live feature-construction
logic in ``src.features.feature_engineering``.

As of feature_schema_version 2.0, ``predictor._MODEL_FEATURES`` is built
directly from ``FeatureEngineer()._build_default_feature_list()`` at import
time (single source of truth) instead of being duplicated as an independent
hardcoded literal, which used to drift out of sync. These tests also cover the "unknown category"
safety requirement for the fixed-category one-hot encoding (unseen city /
building_type values must not raise and must not be silently mapped to an
arbitrary known category).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.feature_engineering import (  # noqa: E402
    BUILDING_TYPE_ONEHOT_COLUMNS,
    BUILDING_TYPE_SLUGS,
    CITY_ONEHOT_COLUMNS,
    CITY_SLUGS,
    FeatureEngineer,
    one_hot_fixed_categories,
)
from src.inference.predictor import _MODEL_FEATURES  # noqa: E402


def test_model_features_match_feature_engineering():
    """``predictor._MODEL_FEATURES`` must match the live default feature list.

    If this ever fails, it means one of the two modules was edited without
    re-deriving the other (e.g. ``_MODEL_FEATURES`` was hardcoded again).
    """
    engineer = FeatureEngineer()
    assert _MODEL_FEATURES == engineer._build_default_feature_list(), (
        "predictor._MODEL_FEATURES no longer matches "
        "FeatureEngineer()._build_default_feature_list()."
    )


def test_model_features_include_all_onehot_columns():
    """Every fixed city/building_type one-hot column must be a model feature."""
    for col in CITY_ONEHOT_COLUMNS + BUILDING_TYPE_ONEHOT_COLUMNS:
        assert col in _MODEL_FEATURES, f"Expected one-hot column '{col}' in _MODEL_FEATURES."


class TestUnknownCategorySafety:
    """'unseen categories должны обрабатываться безопасно' — never raise,
    never silently alias to a real known category; produce an all-zero row.
    """

    def test_unknown_city_yields_all_zero_row(self):
        series = pd.Series(["Атлантида"])  # not in CITY_SLUGS
        result = one_hot_fixed_categories(series, CITY_SLUGS, prefix="city")
        assert list(result.columns) == CITY_ONEHOT_COLUMNS
        assert result.iloc[0].sum() == 0, "Unknown city should map to an all-zero one-hot row."

    def test_unknown_building_type_yields_all_zero_row(self):
        series = pd.Series(["glass_tower"])  # not in BUILDING_TYPE_SLUGS
        result = one_hot_fixed_categories(series, BUILDING_TYPE_SLUGS, prefix="building_type")
        assert list(result.columns) == BUILDING_TYPE_ONEHOT_COLUMNS
        assert result.iloc[0].sum() == 0

    def test_known_city_yields_exactly_one_hot(self):
        series = pd.Series(["Москва"])  # different case/whitespace than the stored key
        result = one_hot_fixed_categories(series, CITY_SLUGS, prefix="city")
        assert result.iloc[0].sum() == 1
        assert result.iloc[0]["city_moskva"] == 1

    def test_case_and_whitespace_insensitive(self):
        variants = pd.Series(["  МОСКВА  ", "москва", "Москва"])
        result = one_hot_fixed_categories(variants, CITY_SLUGS, prefix="city")
        assert (result["city_moskva"] == 1).all()


@pytest.mark.parametrize("city", list(CITY_SLUGS.keys()))
def test_every_known_city_has_a_dedicated_column(city):
    series = pd.Series([city])
    result = one_hot_fixed_categories(series, CITY_SLUGS, prefix="city")
    assert result.iloc[0].sum() == 1

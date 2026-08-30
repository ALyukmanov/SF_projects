"""Tests for src/preprocessing/imputer.py::GroupMedianImputer and
src/data/split_pipeline.py.

Central regression coverage for the train/test leakage fix: rooms/total_area/
floor/floors_total medians must be fit on a TRAIN split only, and must never
change no matter what the holdout rows contain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.imputer import GroupMedianImputer


def _make_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cities = rng.choice(["москва", "санкт-петербург"], size=n)
    rooms = rng.integers(1, 5, size=n).astype(float)
    total_area = rng.uniform(30, 120, size=n).round(1)
    floors_total = rng.integers(5, 20, size=n).astype(float)
    floor = np.array([rng.integers(1, ft + 1) for ft in floors_total], dtype=float)
    df = pd.DataFrame(
        {
            "city": cities,
            "rooms": rooms,
            "total_area": total_area,
            "floor": floor,
            "floors_total": floors_total,
        }
    )
    # Introduce missingness in a known subset of rows.
    df.loc[df.index[:5], "rooms"] = np.nan
    df.loc[df.index[5:8], "floor"] = np.nan
    df.loc[df.index[5:8], "floors_total"] = np.nan
    df.loc[df.index[8:10], "total_area"] = np.nan
    return df


class TestGroupMedianImputerFitTransform:
    def test_transform_fills_all_nan(self):
        df = _make_df()
        imputer = GroupMedianImputer().fit(df)
        result = imputer.transform(df)
        for col in ("rooms", "total_area", "floor", "floors_total"):
            assert not result[col].isna().any(), f"{col} should have no NaN after transform."

    def test_transform_before_fit_raises(self):
        df = _make_df()
        with pytest.raises(RuntimeError):
            GroupMedianImputer().transform(df)

    def test_transform_does_not_alter_already_present_values(self):
        """Rows with a real (non-missing) value must pass through unchanged."""
        df = _make_df()
        imputer = GroupMedianImputer().fit(df)
        result = imputer.transform(df)
        present_mask = df["rooms"].notna()
        assert (
            result.loc[present_mask, "rooms"].astype(float) == df.loc[present_mask, "rooms"]
        ).all()


class TestGroupMedianImputerTrainOnlyFitScope:
    """The central regression coverage for the leakage fix: fitted fill
    values must be fully determined by the TRAIN split and invariant to
    whatever the holdout rows contain.
    """

    def test_fit_on_train_is_unaffected_by_mutating_holdout_rows(self):
        df = _make_df(n=60, seed=1)
        train_idx = df.index[:40]
        holdout_idx = df.index[40:]

        train_df = df.loc[train_idx].reset_index(drop=True)
        imputer_baseline = GroupMedianImputer().fit(train_df)

        # Mutate a COPY of the full dataset's holdout rows to extreme,
        # obviously-distinguishable values, then re-derive the same train
        # slice from that mutated dataset and re-fit.
        mutated = df.copy()
        mutated.loc[holdout_idx, "rooms"] = 999.0
        mutated.loc[holdout_idx, "total_area"] = 999_999.0
        mutated.loc[holdout_idx, "floor"] = 999.0
        mutated.loc[holdout_idx, "floors_total"] = 999.0
        mutated_train_df = mutated.loc[train_idx].reset_index(drop=True)
        imputer_after_mutation = GroupMedianImputer().fit(mutated_train_df)

        assert imputer_baseline.global_median_ == imputer_after_mutation.global_median_
        assert imputer_baseline.city_median_ == imputer_after_mutation.city_median_
        assert (
            imputer_baseline.city_rooms_median_total_area_
            == imputer_after_mutation.city_rooms_median_total_area_
        )

    def test_fit_on_train_only_can_differ_from_fit_on_full_dataset(self):
        """Sanity check that the two fitting scopes are not trivially
        identical by construction — otherwise the invariance test above
        would be vacuous. Uses a bimodal dataset engineered so the train
        slice's median visibly differs from the full dataset's (a single
        outlier is not enough to move a median — a whole second mode is).
        """
        n = 40
        # First half (train) is uniformly 50 sqm; second half ("holdout")
        # is uniformly 200 sqm — the full-dataset median sits between the
        # two modes, while the train-only median stays exactly at 50.
        df = pd.DataFrame(
            {
                "city": ["москва"] * n,
                "rooms": [2.0] * n,
                "total_area": [50.0] * (n // 2) + [200.0] * (n // 2),
                "floor": [5.0] * n,
                "floors_total": [10.0] * n,
            }
        )
        train_df = df.iloc[: n // 2].reset_index(drop=True)

        imputer_train_only = GroupMedianImputer().fit(train_df)
        imputer_full = GroupMedianImputer().fit(df)

        assert imputer_train_only.global_median_["total_area"] == 50.0
        assert (
            imputer_train_only.global_median_["total_area"]
            != imputer_full.global_median_["total_area"]
        )

    def test_holdout_transform_uses_train_fitted_values_not_its_own(self):
        """transform() on a holdout row with a missing value must fill it
        with the TRAIN median, never a statistic derived from the holdout
        set itself (which would still be a leak, just a same-split one).
        """
        train_df = pd.DataFrame(
            {
                "city": ["москва"] * 10,
                "rooms": [2.0] * 10,
                "total_area": [40.0] * 10,
                "floor": [5.0] * 10,
                "floors_total": [10.0] * 10,
            }
        )
        imputer = GroupMedianImputer().fit(train_df)

        holdout_df = pd.DataFrame(
            {
                "city": ["москва"] * 3,
                "rooms": [np.nan, 8.0, 8.0],  # holdout's own median would be 8, not 2
                "total_area": [40.0] * 3,
                "floor": [5.0] * 3,
                "floors_total": [10.0] * 3,
            }
        )
        result = imputer.transform(holdout_df)
        assert result["rooms"].iloc[0] == 2, "Must use the TRAIN median (2), not holdout's own (8)."


class TestSplitImputeFeaturizeTrainOnlyFit:
    """End-to-end regression coverage for src/data/split_pipeline.py::
    split_impute_featurize() itself (not just GroupMedianImputer in
    isolation) — catches a future regression back to the old order
    (featurize the whole dataset, THEN split), which was easy to miss
    without coverage at the pipeline-function level.
    """

    @staticmethod
    def _make_cleaned_style_df(n: int = 80, seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        cities = rng.choice(["москва", "санкт-петербург"], size=n)
        rooms = rng.integers(1, 5, size=n).astype(float)
        total_area = rng.uniform(30, 120, size=n).round(1)
        floors_total = rng.integers(5, 20, size=n).astype(float)
        floor = np.array([rng.integers(1, ft + 1) for ft in floors_total], dtype=float)
        price = total_area * rng.uniform(80_000, 300_000, size=n)
        df = pd.DataFrame(
            {
                "price": price,
                "city": cities,
                "rooms": rooms,
                "total_area": total_area,
                "floor": floor,
                "floors_total": floors_total,
                "building_type": rng.choice(["panel", "brick", "monolith", "block"], size=n),
                "url": [f"https://example.test/{i}" for i in range(n)],
            }
        )
        # Genuine missingness, scattered across the dataset.
        df.loc[df.index[:6], "rooms"] = np.nan
        df.loc[df.index[6:10], "floor"] = np.nan
        df.loc[df.index[6:10], "floors_total"] = np.nan
        return df

    def test_train_medians_unaffected_by_mutating_test_split_rows(self):
        from src.data.split_pipeline import split_impute_featurize

        df = self._make_cleaned_style_df()
        baseline = split_impute_featurize(df, split_strategy="group", random_state=42)
        test_urls = set(baseline.test_raw["url"])

        mutated = df.copy()
        mutate_mask = mutated["url"].isin(test_urls)
        mutated.loc[mutate_mask, "rooms"] = 999.0
        mutated.loc[mutate_mask, "total_area"] = 999_999.0
        mutated.loc[mutate_mask, "floor"] = 999.0
        mutated.loc[mutate_mask, "floors_total"] = 999.0

        after_mutation = split_impute_featurize(mutated, split_strategy="group", random_state=42)

        assert baseline.imputer.global_median_ == after_mutation.imputer.global_median_, (
            "split_impute_featurize's fitted imputer must not change when only "
            "test-split rows are mutated -- a regression back to fitting on the "
            "full (pre-split) dataframe would make this fail."
        )
        # And the resulting TRAIN feature matrix itself must be identical too,
        # not just the imputer's stored statistics.
        pd.testing.assert_frame_equal(
            baseline.X_train.reset_index(drop=True), after_mutation.X_train.reset_index(drop=True)
        )

    def test_train_medians_differ_from_a_full_dataset_fit(self):
        """Sanity check that this dataset is actually capable of exposing the
        leak (train-only vs full-dataset medians must be able to differ) --
        otherwise the invariance test above would be vacuous."""
        from src.data.split_pipeline import split_impute_featurize
        from src.preprocessing.imputer import GroupMedianImputer

        df = self._make_cleaned_style_df()
        split_result = split_impute_featurize(df, split_strategy="group", random_state=42)
        full_dataset_imputer = GroupMedianImputer().fit(df)

        assert (
            split_result.imputer.global_median_["total_area"]
            != full_dataset_imputer.global_median_["total_area"]
            or split_result.imputer.global_median_["rooms"]
            != full_dataset_imputer.global_median_["rooms"]
        ), "Test dataset should be capable of exposing a train-vs-full-dataset difference."


class TestGroupMedianImputerSerialization:
    def test_to_dict_from_dict_roundtrip_produces_identical_transform(self):
        df = _make_df(seed=3)
        imputer = GroupMedianImputer().fit(df)
        restored = GroupMedianImputer.from_dict(imputer.to_dict())

        original = imputer.transform(df)
        via_restored = restored.transform(df)
        for col in ("rooms", "total_area", "floor", "floors_total"):
            assert (original[col].astype(float) == via_restored[col].astype(float)).all()

    def test_from_dict_none_returns_unfitted_imputer(self):
        imputer = GroupMedianImputer.from_dict(None)
        assert imputer.fitted_ is False

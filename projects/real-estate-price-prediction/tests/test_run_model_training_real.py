"""Tests for scripts/run_model_training_real.py's data_source provenance
resolution -- see _resolve_data_source()'s docstring for why a mixed
data_source column must not silently collapse to a majority label -- and
for _build_lineage_metadata()'s artefact provenance completeness."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from scripts.run_model_training_real import _build_lineage_metadata, _resolve_data_source


class TestResolveDataSource:
    def test_synthetic_flag_wins_regardless_of_column(self):
        df = pd.DataFrame({"data_source": ["restate"] * 5})
        assert _resolve_data_source(is_synthetic=True, df=df) == "synthetic_demo_generator"

    def test_single_real_source_returned_as_is(self):
        df = pd.DataFrame({"data_source": ["restate"] * 10})
        assert _resolve_data_source(is_synthetic=False, df=df) == "restate"

    def test_missing_data_source_column_falls_back_to_legacy_cian_label(self):
        df = pd.DataFrame({"price": [1, 2, 3]})
        assert _resolve_data_source(is_synthetic=False, df=df) == "cian_scraper_manual_run"

    def test_all_null_data_source_column_falls_back_to_legacy_cian_label(self):
        df = pd.DataFrame({"data_source": [None, None, None]})
        assert _resolve_data_source(is_synthetic=False, df=df) == "cian_scraper_manual_run"

    def test_mixed_data_source_values_are_not_silently_collapsed(self, caplog):
        df = pd.DataFrame({"data_source": ["restate"] * 8 + ["cian"] * 2})
        result = _resolve_data_source(is_synthetic=False, df=df)
        assert result == "mixed:cian,restate"
        assert "cian" in result and "restate" in result

    def test_mixed_with_nulls_ignores_nulls_when_forming_the_label(self):
        df = pd.DataFrame({"data_source": ["restate", None, "cian", None]})
        result = _resolve_data_source(is_synthetic=False, df=df)
        assert result == "mixed:cian,restate"


class TestBuildLineageMetadata:
    """Lineage completeness: version/timestamp/git revision, dataset
    fingerprint, city/category coverage, feature target+transform, split
    method, seed, hyperparameters, and library versions must all be present
    in every saved artefact."""

    _REQUIRED_KEYS = {
        "data_source",
        "is_synthetic",
        "dataset_path",
        "dataset_sha256",
        "dataset_rows",
        "dataset_columns",
        "city_coverage",
        "category_coverage",
        "target",
        "target_transform",
        "training_start",
        "training_end",
        "split_strategy",
        "random_seed",
        "hyperparameters",
        "tuned_from_study",
        "library_versions",
        "feature_schema_version",
        "git_commit",
    }

    def _build(self, df, **overrides):
        kwargs = dict(
            df=df,
            resolved_data_source="restate",
            is_synthetic=False,
            dataset_path_repo_relative="data/processed/real_estate_engineered.csv",
            dataset_sha256="deadbeef" * 8,
            dataset_rows=len(df),
            dataset_columns=len(df.columns),
            training_start=datetime(2026, 8, 27, 9, 0, 0),
            training_end=datetime(2026, 8, 27, 9, 1, 0),
            split_strategy="group_aware_80_20_random_state_42",
            trainer_params={"n_estimators": 600},
            params_from_study="xgboost",
        )
        kwargs.update(overrides)
        return _build_lineage_metadata(**kwargs)

    def test_all_required_lineage_keys_present(self):
        df = pd.DataFrame(
            {
                "city": ["Москва", "Санкт-Петербург"],
                "source_category": ["1_rooms_flats_sale", "cottages_sale"],
            }
        )
        metadata = self._build(df)
        assert self._REQUIRED_KEYS <= set(metadata.keys())

    def test_city_and_category_coverage_reflect_actual_counts(self):
        df = pd.DataFrame(
            {
                "city": ["Москва", "Москва", "Санкт-Петербург"],
                "source_category": ["1_rooms_flats_sale", "1_rooms_flats_sale", "cottages_sale"],
            }
        )
        metadata = self._build(df)
        assert metadata["city_coverage"] == {"Москва": 2, "Санкт-Петербург": 1}
        assert metadata["category_coverage"] == {"1_rooms_flats_sale": 2, "cottages_sale": 1}

    def test_missing_city_or_category_columns_yield_none_not_a_crash(self):
        df = pd.DataFrame({"price": [1, 2, 3]})
        metadata = self._build(df)
        assert metadata["city_coverage"] is None
        assert metadata["category_coverage"] is None

    def test_target_and_transform_are_explicit(self):
        df = pd.DataFrame({"price": [1, 2, 3]})
        metadata = self._build(df)
        assert metadata["target"] == "price"
        assert metadata["target_transform"] == "log1p"

    def test_untuned_run_records_library_defaults_not_empty_dict(self):
        df = pd.DataFrame({"price": [1, 2, 3]})
        metadata = self._build(df, trainer_params={}, params_from_study=None)
        assert metadata["hyperparameters"] == "library_defaults"
        assert metadata["tuned_from_study"] is None

    def test_library_versions_include_core_dependencies(self):
        df = pd.DataFrame({"price": [1, 2, 3]})
        metadata = self._build(df)
        assert set(metadata["library_versions"].keys()) >= {
            "python",
            "pandas",
            "sklearn",
            "xgboost",
        }

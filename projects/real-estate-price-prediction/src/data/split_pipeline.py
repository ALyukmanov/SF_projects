"""
Single source of truth for the leakage-safe split -> impute -> featurize
pipeline used by every script that trains or evaluates a real-data
candidate (``run_model_training_real.py``, ``verify_split_leakage.py``,
``analyze_final_candidate_real.py``).

Order matters: raw/cleaned listings -> (group-aware or random) split ->
``GroupMedianImputer`` fit on TRAIN ONLY -> ``FeatureEngineer.create_features``
run separately on each split half -> feature/target extraction. Computing
imputation statistics or engineered features on the full dataset BEFORE the
split (an earlier approach) leaks holdout information into train.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.data.schema import build_location_groups, build_split_groups
from src.features.feature_engineering import FeatureEngineer
from src.preprocessing.imputer import GroupMedianImputer

SplitStrategy = Literal["random", "group", "location"]


def _resolve_groups(df: pd.DataFrame, split_strategy: str):
    """Return (group_series_or_None, resolved_strategy_label) for *split_strategy*."""
    if split_strategy == "group":
        return build_split_groups(df), "group_aware_80_20_random_state_42"
    if split_strategy == "location":
        return build_location_groups(df), "location_grouped_80_20_random_state_42"
    if split_strategy == "random":
        return None, "random_holdout_80_20_random_state_42"
    raise ValueError(f"Unknown split_strategy: {split_strategy!r}")


def group_holdout_indices(
    df: pd.DataFrame,
    split_strategy: SplitStrategy = "location",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Just the index split (no impute / featurize).

    Returns ``(train_idx, test_idx, groups)`` where *groups* is the group-id
    Series for the strategy (``None`` for ``"random"``). Use this when a
    caller needs the group vector for its own CV (e.g. tuning does
    ``GroupKFold`` on the train portion) and then rebuilds the *identical*
    holdout via :func:`split_impute_featurize` with the same
    strategy/seed/test_size.
    """
    df = df.reset_index(drop=True)
    groups, _ = _resolve_groups(df, split_strategy)
    if groups is None:
        train_idx, test_idx = train_test_split(
            df.index.to_numpy(), test_size=test_size, random_state=random_state
        )
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, groups=groups))
    return train_idx, test_idx, groups


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    imputer: GroupMedianImputer
    feature_names: list
    split_strategy: str


def split_impute_featurize(
    df: pd.DataFrame,
    split_strategy: SplitStrategy = "group",
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitResult:
    """Split *df* (cleaned, pre-feature-engineering listings), fit the
    imputer on the train half only, and build the final (X, y) matrices for
    both halves.

    Args:
        df: Output of ``DataCleaner.clean()`` + economic features — i.e. the
            "cleaned" dataset, which may still contain NaN in
            rooms/total_area/floor/floors_total. Must NOT already have had
            ``FeatureEngineer.create_features`` applied (this function calls
            it internally, once per split half).
        split_strategy: ``"location"`` (strictest, primary for the geo model)
            keeps every listing in the same building — same rounded
            coordinate — entirely on one side, via
            :func:`src.data.schema.build_location_groups`. ``"group"`` only
            isolates near-duplicate *listings*
            (:func:`src.data.schema.build_split_groups`); it is more
            optimistic once coordinates exist because two different flats in
            one building can still straddle the split. ``"random"`` is a
            plain positional split, kept only as a diagnostic comparison —
            never the strategy behind a promotion candidate.
        test_size: Held-out fraction.
        random_state: Seed for the splitter.

    Returns:
        A :class:`SplitResult` with train/test feature matrices, the fitted
        imputer (persist this in the model artefact for inference), and the
        raw (pre-impute, pre-featurize) train/test slices for diagnostics.
    """
    df = df.reset_index(drop=True)

    groups, resolved_strategy = _resolve_groups(df, split_strategy)
    if groups is None:
        train_idx, test_idx = train_test_split(
            df.index.to_numpy(), test_size=test_size, random_state=random_state
        )
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, groups=groups))

    train_raw = df.iloc[train_idx].reset_index(drop=True)
    test_raw = df.iloc[test_idx].reset_index(drop=True)

    imputer = GroupMedianImputer().fit(train_raw)
    train_imputed = imputer.transform(train_raw)
    test_imputed = imputer.transform(test_raw)

    feat_eng = FeatureEngineer()
    X_train, y_train = feat_eng.prepare_for_training(train_imputed)
    feature_names = list(X_train.columns)

    test_featurized = feat_eng.create_features(test_imputed)
    test_featurized = test_featurized.dropna(subset=["price"])
    X_test = test_featurized.reindex(columns=feature_names, fill_value=0)
    y_test = np.log1p(test_featurized["price"])
    # Any leftover NaN (e.g. a feature entirely absent in this dataset) —
    # same constant-fill rule as FeatureEngineer.prepare_for_training, never
    # a statistic computed from the test set itself.
    X_test = X_test.fillna(0)

    return SplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_raw=train_raw,
        test_raw=test_raw,
        imputer=imputer,
        feature_names=feature_names,
        split_strategy=resolved_strategy,
    )

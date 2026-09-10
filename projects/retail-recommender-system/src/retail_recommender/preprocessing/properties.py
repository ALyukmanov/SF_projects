"""Item metadata из item_properties: categoryid и available, с честным as-of lookup.

Политика leakage-safe:
  item_properties хранят недельные срезы. Для момента предсказания T
  берётся последний срез со snapshot_ts <= T.
  Если срезов до T нет (события 03–09 мая 2015, до первого среза 10 мая) —
  значение возвращается как UNKNOWN (pd.NA). Forward-fill из будущего не делается.
"""
from __future__ import annotations

import pandas as pd

from retail_recommender.config import Config
from retail_recommender.data.load import load_item_properties_subset

META_PROPERTIES = {"categoryid", "available"}


def build_metadata_snapshots(cfg: Config) -> pd.DataFrame:
    """Длинная таблица: itemid, property, snapshot_ts, value (int)."""
    raw = load_item_properties_subset(cfg, META_PROPERTIES)
    raw = raw.rename(columns={"timestamp": "snapshot_ts"})
    raw["snapshot_ts"] = pd.to_datetime(raw["snapshot_ts"], unit="ms", utc=True)
    raw["value_int"] = pd.to_numeric(raw["value"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["value_int"])
    raw = raw[["itemid", "property", "snapshot_ts", "value_int"]].drop_duplicates()
    raw = raw.sort_values(["property", "itemid", "snapshot_ts"]).reset_index(drop=True)
    return raw


def save_metadata_snapshots(df: pd.DataFrame, cfg: Config) -> None:
    cfg.paths.interim_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.paths.interim_dir / "item_metadata_snapshots.parquet", index=False)


def load_metadata_snapshots(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.paths.interim_dir / "item_metadata_snapshots.parquet")


def as_of_lookup(
    snapshots: pd.DataFrame,
    queries: pd.DataFrame,
    property_name: str,
    item_col: str = "itemid",
    time_col: str = "ts",
) -> pd.Series:
    """Для каждой строки queries вернуть значение property на момент queries[time_col].

    queries: DataFrame с колонками [item_col, time_col] (time_col — tz-aware UTC).
    Возвращает Series (Int64) той же длины/индекса, что queries; pd.NA если среза до T нет.
    Гарантия: используется только срез со snapshot_ts <= time_col.
    """
    snap = (
        snapshots[snapshots["property"] == property_name]
        [["itemid", "snapshot_ts", "value_int"]]
        .sort_values("snapshot_ts")
        .reset_index(drop=True)
    )
    snap["itemid"] = snap["itemid"].astype("int64")
    q = queries[[item_col, time_col]].copy()
    # merge_asof (pandas >= 2.2) требует совпадения dtype ключа by; на Windows
    # np.int по умолчанию int32, поэтому нормализуем обе стороны к int64.
    q[item_col] = q[item_col].astype("int64")
    q["_order"] = range(len(q))
    q = q.sort_values(time_col).reset_index(drop=True)

    merged = pd.merge_asof(
        q,
        snap,
        left_on=time_col,
        right_on="snapshot_ts",
        left_by=item_col,
        right_by="itemid",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("_order")
    result = merged["value_int"].astype("Int64")
    result.index = queries.index
    return result

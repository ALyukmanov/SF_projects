"""Построение event-level таблицы взаимодействий из сырых событий.

Сырые события не изменяются. На processed-стадии:
  - убираются полные дубли строк (в данных их 460);
  - добавляется вес события (implicit feedback);
  - помечаются аномально активные визиторы (> порога событий за весь период);
  - проставляется метка split по дате события.
Временная последовательность сохраняется — агрегирование до (user, item)
делают уже сами модели, если им это нужно.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from retail_recommender.config import Config
from retail_recommender.data.load import load_events


@dataclass
class InteractionBuildStats:
    raw_rows: int
    full_duplicates_removed: int
    processed_rows: int
    n_visitors: int
    n_items: int
    n_bots: int
    bot_event_share: float
    split_counts: dict[str, int]


def _assign_split(ts: pd.Series, cfg: Config) -> pd.Series:
    s = cfg.split
    d = ts.dt.tz_convert("UTC").dt.normalize()
    out = pd.Series(pd.NA, index=ts.index, dtype="string")
    to_ts = lambda x: pd.Timestamp(x, tz="UTC")
    out[(d >= to_ts(s.train_start)) & (d <= to_ts(s.train_end))] = "train"
    out[(d >= to_ts(s.val_start)) & (d <= to_ts(s.val_end))] = "val"
    out[(d >= to_ts(s.test_start)) & (d <= to_ts(s.test_end))] = "test"
    return out


def build_interactions(cfg: Config) -> tuple[pd.DataFrame, InteractionBuildStats]:
    raw = load_events(cfg)
    raw_rows = len(raw)

    df = raw.drop_duplicates().reset_index(drop=True)
    full_dups = raw_rows - len(df)

    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["event"] = df["event"].astype("string")
    df["weight"] = df["event"].map(cfg.event_weights).astype("int64")

    events_per_visitor = df.groupby("visitorid")["timestamp"].transform("size")
    df["is_bot"] = events_per_visitor > cfg.bot_event_threshold

    df["split"] = _assign_split(df["ts"], cfg)

    df = df.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
    df = df[["visitorid", "itemid", "timestamp", "ts", "event", "weight",
             "transactionid", "is_bot", "split"]]

    bot_rows = int(df["is_bot"].sum())
    stats = InteractionBuildStats(
        raw_rows=raw_rows,
        full_duplicates_removed=full_dups,
        processed_rows=len(df),
        n_visitors=int(df["visitorid"].nunique()),
        n_items=int(df["itemid"].nunique()),
        n_bots=int(df.loc[df["is_bot"], "visitorid"].nunique()),
        bot_event_share=round(bot_rows / len(df), 5),
        split_counts={k: int(v) for k, v in df["split"].value_counts(dropna=False).items()},
    )
    return df, stats


def save_interactions(df: pd.DataFrame, cfg: Config) -> None:
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.paths.processed_dir / "interactions.parquet", index=False)


def load_interactions(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.paths.processed_dir / "interactions.parquet")

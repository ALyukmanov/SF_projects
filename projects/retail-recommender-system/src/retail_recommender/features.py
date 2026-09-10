"""Небольшие функции для подготовки признаков и целевой переменной (неделя 2).

Один принцип на весь модуль: всё, что относится к признакам, считается только
из событий с ``ts < cutoff``. Целевая переменная смотрит в окно после cutoff.

Здесь лежит логика, которую удобно проверять тестами (as-of категория, счётчики
событий, целевая переменная). Остальная часть подготовки признаков —
арифметика долей и давностей — видна прямо в ``notebooks/02_features.ipynb``.
"""
from __future__ import annotations

import pandas as pd

from retail_recommender.config import Config

EVENT_TYPES = ("view", "addtocart", "transaction")


def cutoff_ts(cfg: Config) -> pd.Timestamp:
    """Момент, в который модель делает предсказание (граница train / target)."""
    return pd.Timestamp(cfg.feature_engineering["cutoff"], tz="UTC")


def label_window_end_ts(cfg: Config) -> pd.Timestamp:
    """Правая граница целевого окна (exclusive): конец дня label_window_end."""
    end = pd.Timestamp(cfg.feature_engineering["label_window_end"], tz="UTC")
    return end + pd.Timedelta(days=1)


def pre_cutoff(interactions: pd.DataFrame, cutoff: pd.Timestamp,
               drop_bots: bool = True) -> pd.DataFrame:
    """События строго до cutoff — единственный источник данных для признаков."""
    df = interactions
    if drop_bots and "is_bot" in df.columns:
        df = df[~df["is_bot"]]
    return df[df["ts"] < cutoff].copy()


def days_before(cutoff: pd.Timestamp, ts: pd.Series) -> pd.Series:
    """Сколько дней прошло от каждого момента ts до cutoff."""
    return (cutoff - ts).dt.total_seconds() / 86400.0


def counts_by_event(events: pd.DataFrame, key) -> pd.DataFrame:
    """Счётчики событий по типам (view / addtocart / transaction) в разрезе key."""
    table = events.pivot_table(index=key, columns="event", values="ts",
                               aggfunc="size", fill_value=0)
    return table.reindex(columns=list(EVENT_TYPES), fill_value=0).rename_axis(columns=None)


def category_as_of(snapshots: pd.DataFrame, at: pd.Timestamp) -> pd.Series:
    """itemid -> categoryid по последнему недельному срезу свойств с snapshot_ts <= at.

    Товары без среза до момента at в результат не попадают (категория неизвестна) —
    значение из будущего не подставляется.
    """
    snap = snapshots[(snapshots["property"] == "categoryid")
                     & (snapshots["snapshot_ts"] <= at)]
    if snap.empty:
        return pd.Series(dtype="Int64", name="categoryid")
    latest = snap.sort_values("snapshot_ts").groupby("itemid")["value_int"].last()
    latest.name = "categoryid"
    latest.index.name = "itemid"
    return latest.astype("Int64")


def strong_events_after(interactions: pd.DataFrame, cutoff: pd.Timestamp,
                        window_end: pd.Timestamp, strong_events) -> pd.DataFrame:
    """Пары (visitorid, itemid) с сильным событием в целевом окне [cutoff, window_end)."""
    df = interactions
    if "is_bot" in df.columns:
        df = df[~df["is_bot"]]
    mask = ((df["ts"] >= cutoff) & (df["ts"] < window_end)
            & df["event"].isin(list(strong_events)))
    return df.loc[mask, ["visitorid", "itemid"]].drop_duplicates()


def build_target(pairs: pd.DataFrame, strong_pairs: pd.DataFrame) -> pd.DataFrame:
    """target = 1 для пары из pairs, если та же пара есть в strong_pairs.

    pairs — исторические пары (visitorid, itemid), наблюдавшиеся до cutoff.
    strong_pairs — пары с сильным событием в целевом окне (см. strong_events_after).
    """
    marked = strong_pairs[["visitorid", "itemid"]].drop_duplicates().assign(target=1)
    out = pairs[["visitorid", "itemid"]].merge(marked, on=["visitorid", "itemid"], how="left")
    out["target"] = out["target"].fillna(0).astype("int64")
    return out

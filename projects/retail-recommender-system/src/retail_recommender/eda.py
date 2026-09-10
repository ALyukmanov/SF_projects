"""Агрегации для EDA. Только вычисления, без графиков — графики строит notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd

from retail_recommender.config import Config
from retail_recommender.split.temporal import slice_window, windows


def raw_overview(events_raw: pd.DataFrame) -> pd.DataFrame:
    """Обзор сырого events.csv: размер, ключи, диапазон дат, дубли, пропуски."""
    dt = pd.to_datetime(events_raw["timestamp"], unit="ms")
    rows = [
        ("строк", len(events_raw)),
        ("уникальных visitorid", events_raw["visitorid"].nunique()),
        ("уникальных itemid", events_raw["itemid"].nunique()),
        ("уникальных transactionid", int(events_raw["transactionid"].nunique())),
        ("дата: min", dt.min()),
        ("дата: max", dt.max()),
        ("период, дней", (dt.max() - dt.min()).days),
        ("полных дубликатов строк", int(events_raw.duplicated().sum())),
        ("пропусков timestamp", int(events_raw["timestamp"].isna().sum())),
        ("пропусков visitorid", int(events_raw["visitorid"].isna().sum())),
        ("пропусков itemid", int(events_raw["itemid"].isna().sum())),
        ("transactionid заполнен (только у transaction)", int(events_raw["transactionid"].notna().sum())),
    ]
    return pd.DataFrame(rows, columns=["метрика", "значение"])


def user_activity_distribution(interactions: pd.DataFrame) -> pd.DataFrame:
    """Сколько визиторов имеют 1 / 2 / 3 / 4 / 5+ взаимодействий."""
    epv = interactions.groupby("visitorid").size()
    buckets = pd.cut(epv, bins=[0, 1, 2, 3, 4, 10, 100, np.inf],
                     labels=["1", "2", "3", "4", "5-10", "11-100", "100+"])
    tab = buckets.value_counts().sort_index().rename("visitors").to_frame()
    tab["share"] = (tab["visitors"] / len(epv)).round(4)
    return tab.reset_index(names="n_interactions")


def top_users(interactions: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    df = interactions.assign(_is_txn=(interactions["event"] == "transaction").astype("int64"))
    g = df.groupby("visitorid")
    out = pd.DataFrame({
        "events": g.size(),
        "items": g["itemid"].nunique(),
        "transactions": g["_is_txn"].sum(),
    }).sort_values("events", ascending=False).head(k)
    return out.reset_index()


def item_popularity_buckets(interactions: pd.DataFrame) -> pd.DataFrame:
    pop = interactions.groupby("itemid").size()
    buckets = pd.cut(pop, bins=[0, 1, 2, 5, 10, 50, np.inf],
                     labels=["1", "2", "3-5", "6-10", "11-50", "50+"])
    tab = buckets.value_counts().sort_index().rename("items").to_frame()
    tab["share"] = (tab["items"] / len(pop)).round(4)
    return tab.reset_index(names="n_interactions")


def transactions_over_time(interactions: pd.DataFrame, freq: str = "W") -> pd.Series:
    tx = interactions[interactions["event"] == "transaction"]
    s = tx.set_index("ts").resample(freq).size()
    s.index = s.index.tz_convert("UTC").tz_localize(None)
    return s.rename("transactions")


def basket_stats(interactions: pd.DataFrame) -> pd.DataFrame:
    tx = interactions[interactions["event"] == "transaction"]
    size = tx.groupby("transactionid").size()
    return pd.DataFrame([{
        "n_baskets": int(tx["transactionid"].nunique()),
        "items_per_basket_mean": round(float(size.mean()), 3),
        "items_per_basket_median": float(size.median()),
        "items_per_basket_max": int(size.max()),
        "single_item_baskets_share": round(float((size == 1).mean()), 4),
    }])


def events_per_day(interactions: pd.DataFrame) -> pd.Series:
    s = interactions.set_index("ts").resample("D").size()
    # pandas timeseries-plot не поддерживает tz-aware индекс
    s.index = s.index.tz_convert("UTC").tz_localize(None)
    return s.rename("events")


def event_type_counts(interactions: pd.DataFrame) -> pd.Series:
    return interactions["event"].value_counts()


def events_per_visitor(interactions: pd.DataFrame) -> pd.Series:
    return interactions.groupby("visitorid").size().rename("events_per_visitor")


def unique_items_per_visitor(interactions: pd.DataFrame) -> pd.Series:
    return interactions.groupby("visitorid")["itemid"].nunique().rename("items_per_visitor")


def item_popularity(interactions: pd.DataFrame) -> pd.Series:
    return interactions.groupby("itemid").size().sort_values(ascending=False).rename("events")


def funnel_pair_level(interactions: pd.DataFrame) -> pd.DataFrame:
    piv = interactions.pivot_table(
        index=["visitorid", "itemid"], columns="event", values="timestamp",
        aggfunc="count", fill_value=0,
    )
    for col in ("view", "addtocart", "transaction"):
        if col not in piv:
            piv[col] = 0
    return pd.DataFrame({
        "stage": ["view", "addtocart", "transaction"],
        "pairs": [
            int((piv["view"] > 0).sum()),
            int((piv["addtocart"] > 0).sum()),
            int((piv["transaction"] > 0).sum()),
        ],
    })


def warm_cold_by_window(interactions: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    w = windows(cfg)
    clean = interactions[~interactions["is_bot"]]
    rows = []
    for name in ("val", "test"):
        win = w[name]
        hist_users = set(clean.loc[clean["ts"] < win.start, "visitorid"].unique())
        win_users = set(slice_window(clean, win)["visitorid"].unique())
        warm = len(win_users & hist_users)
        rows.append({
            "window": name,
            "active_users": len(win_users),
            "warm": warm,
            "cold": len(win_users) - warm,
            "cold_share": round((len(win_users) - warm) / max(len(win_users), 1), 3),
        })
    return pd.DataFrame(rows)


def bot_summary(interactions: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    bots = interactions[interactions["is_bot"]]
    return pd.DataFrame([{
        "bot_threshold": cfg.bot_event_threshold,
        "bot_visitors": int(bots["visitorid"].nunique()),
        "bot_events": len(bots),
        "bot_event_share": round(len(bots) / len(interactions), 4),
        "total_visitors": int(interactions["visitorid"].nunique()),
    }])


def property_coverage(interactions: pd.DataFrame, meta_snapshots: pd.DataFrame) -> pd.DataFrame:
    event_items = set(interactions["itemid"].unique())
    cat_items = set(meta_snapshots.loc[meta_snapshots["property"] == "categoryid", "itemid"].unique())
    strong_items = set(interactions.loc[
        interactions["event"].isin(["addtocart", "transaction"]), "itemid"].unique())
    txn_items = set(interactions.loc[interactions["event"] == "transaction", "itemid"].unique())
    def cov(a, b):
        return round(len(a & b) / max(len(a), 1), 4)
    return pd.DataFrame([
        {"subset": "all event items", "n": len(event_items), "with_category": cov(event_items, cat_items)},
        {"subset": "addtocart/transaction items", "n": len(strong_items),
         "with_category": cov(strong_items, cat_items)},
        {"subset": "transaction items", "n": len(txn_items), "with_category": cov(txn_items, cat_items)},
    ])

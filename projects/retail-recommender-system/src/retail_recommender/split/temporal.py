"""Временной split. Вся информация течёт только вперёд: train -> val -> test."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from retail_recommender.config import Config


@dataclass(frozen=True)
class Window:
    name: str
    start: pd.Timestamp  # включительно
    end_exclusive: pd.Timestamp  # исключительно


def windows(cfg: Config) -> dict[str, Window]:
    s = cfg.split
    day = pd.Timedelta(days=1)
    ts = lambda x: pd.Timestamp(x, tz="UTC")
    return {
        "train": Window("train", ts(s.train_start), ts(s.train_end) + day),
        "val": Window("val", ts(s.val_start), ts(s.val_end) + day),
        "test": Window("test", ts(s.test_start), ts(s.test_end) + day),
    }


def slice_window(df: pd.DataFrame, w: Window, time_col: str = "ts") -> pd.DataFrame:
    return df[(df[time_col] >= w.start) & (df[time_col] < w.end_exclusive)]


def history_before(df: pd.DataFrame, w: Window, time_col: str = "ts") -> pd.DataFrame:
    """Все события строго до начала окна w — доступная история для инференса."""
    return df[df[time_col] < w.start]


def assert_forward_only(df: pd.DataFrame, cfg: Config, time_col: str = "ts") -> None:
    """Boundary-инвариант: окна не пересекаются и упорядочены во времени."""
    w = windows(cfg)
    tr = slice_window(df, w["train"], time_col)
    va = slice_window(df, w["val"], time_col)
    te = slice_window(df, w["test"], time_col)
    if len(tr) and len(va):
        assert tr[time_col].max() < va[time_col].min(), "train пересекается с val"
    if len(va) and len(te):
        assert va[time_col].max() < te[time_col].min(), "val пересекается с test"
    if len(tr) and len(te):
        assert tr[time_col].max() < te[time_col].min(), "train пересекается с test"


def describe_windows(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Размеры окон: события, пользователи, товары, сильные события, покупки,
    cold/warm пользователи, новые товары. Боты исключены (как и в оценке)."""
    if "is_bot" in df.columns:
        df = df[~df["is_bot"]]
    w = windows(cfg)
    strong = set(cfg.strong_events)
    txn = set(cfg.transaction_events)
    rows = []
    seen_users: set[int] = set()
    seen_items: set[int] = set()
    for name in ["train", "val", "test"]:
        part = slice_window(df, w[name])
        users = set(part["visitorid"].unique())
        items = set(part["itemid"].unique())
        strong_part = part[part["event"].isin(strong)]
        txn_part = part[part["event"].isin(txn)]
        cold_users = users - seen_users if name != "train" else set()
        new_items = items - seen_items if name != "train" else set()
        rows.append({
            "window": name,
            "start": w[name].start.date().isoformat(),
            "end": (w[name].end_exclusive - pd.Timedelta(days=1)).date().isoformat(),
            "events": len(part),
            "users": len(users),
            "items": len(items),
            "strong_events": len(strong_part),
            "transactions": len(txn_part),
            "strong_users": int(strong_part["visitorid"].nunique()),
            "txn_users": int(txn_part["visitorid"].nunique()),
            "cold_users_vs_past": len(cold_users),
            "warm_users_vs_past": len(users) - len(cold_users) if name != "train" else len(users),
            "new_items_vs_past": len(new_items),
        })
        seen_users |= users
        seen_items |= items
    return pd.DataFrame(rows)

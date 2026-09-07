"""Общие фикстуры: синтетические взаимодействия в реальных окнах split."""
import pandas as pd
import pytest

from second_year.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def make_interactions():
    return _make_interactions


def _make_interactions(rows: list[dict]) -> pd.DataFrame:
    """rows: {visitorid, itemid, date, event, [transactionid], [is_bot]}."""
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["date"], utc=True)
    df["timestamp"] = (df["ts"].astype("int64") // 1_000_000).astype("int64")
    weights = {"view": 1, "addtocart": 3, "transaction": 5}
    df["weight"] = df["event"].map(weights).astype("int64")
    if "transactionid" not in df:
        df["transactionid"] = pd.NA
    if "is_bot" not in df:
        df["is_bot"] = False
    df["split"] = pd.NA  # не используется в этих тестах напрямую
    return df[["visitorid", "itemid", "timestamp", "ts", "event", "weight",
               "transactionid", "is_bot", "split"]].sort_values("timestamp").reset_index(drop=True)

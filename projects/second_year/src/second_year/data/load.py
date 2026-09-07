"""Чтение сырых CSV. Единственное место, где известны имена колонок исходных файлов."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from second_year.config import Config

EVENTS_DTYPES = {
    "timestamp": "int64",
    "visitorid": "int64",
    "event": "string",
    "itemid": "int64",
    "transactionid": "float64",  # пусто у не-transaction, поэтому float
}

PROPERTIES_DTYPES = {
    "timestamp": "int64",
    "itemid": "int64",
    "property": "string",
    "value": "string",
}


def load_events(cfg: Config) -> pd.DataFrame:
    """Сырой events.csv без изменений."""
    return pd.read_csv(cfg.events_csv, dtype=EVENTS_DTYPES)


def load_category_tree(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.category_tree_csv, dtype={"categoryid": "int64", "parentid": "Int64"})
    return df


def iter_item_properties(cfg: Config, chunksize: int = 2_000_000) -> Iterator[pd.DataFrame]:
    """Потоковое чтение обеих частей item_properties (файлы большие)."""
    for path in cfg.item_property_csvs:
        for chunk in pd.read_csv(path, dtype=PROPERTIES_DTYPES, chunksize=chunksize):
            yield chunk


def load_item_properties_subset(cfg: Config, properties: set[str]) -> pd.DataFrame:
    """Только строки с нужными property-кодами (например categoryid, available)."""
    parts = []
    for chunk in iter_item_properties(cfg):
        parts.append(chunk[chunk["property"].isin(properties)])
    return pd.concat(parts, ignore_index=True)

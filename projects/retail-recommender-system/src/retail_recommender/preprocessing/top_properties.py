"""TOP-N самых распространённых property-кодов товаров и их as-of значения.

Зачем: LightFM умеет учитывать side-features товара. Берём несколько самых частых
property-кодов из item_properties как категориальные признаки. Список определяется
ТОЛЬКО по срезам с ``timestamp <= cutoff`` (без заглядывания за момент оценки).

Коды свойств в RetailRocket захешированы (целые числа), кроме двух литералов
``categoryid`` и ``available``. Значения свойств тоже захешированы.
"""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from retail_recommender.config import Config
from retail_recommender.data.load import iter_item_properties

DEFAULT_TOP_N = 20


def _cutoff_ms(cutoff: pd.Timestamp) -> int:
    return int(cutoff.tz_convert("UTC").value // 1_000_000)


def compute_top_property_codes(
    cfg: Config, cutoff: pd.Timestamp, top_n: int = DEFAULT_TOP_N
) -> tuple[list[str], dict[str, int]]:
    """Список TOP-N property-кодов по числу строк среди срезов с snapshot_ts <= cutoff.

    Возвращает (top_codes, full_counts). Порядок — по убыванию частоты; при равенстве
    частот — по коду (детерминированно).
    """
    cut = _cutoff_ms(cutoff)
    counts: Counter[str] = Counter()
    for chunk in iter_item_properties(cfg):
        sub = chunk[chunk["timestamp"] <= cut]
        if sub.empty:
            continue
        for prop, n in sub.groupby("property").size().items():
            counts[str(prop)] += int(n)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_codes = [code for code, _ in ordered[:top_n]]
    return top_codes, dict(counts)


def top_properties_path(cfg: Config):
    return cfg.paths.interim_dir / "top_property_codes.json"


def save_top_property_codes(
    cfg: Config, cutoff: pd.Timestamp, top_codes: list[str], counts: dict[str, int]
) -> None:
    cfg.paths.interim_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cutoff": str(cutoff.date()),
        "top_n": len(top_codes),
        "top_codes": top_codes,
        "top_counts": {c: counts[c] for c in top_codes},
        "n_distinct_codes": len(counts),
    }
    top_properties_path(cfg).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_top_property_codes(cfg: Config) -> dict:
    return json.loads(top_properties_path(cfg).read_text(encoding="utf-8"))


def build_top_property_snapshots(
    cfg: Config, top_codes: list[str]
) -> pd.DataFrame:
    """Длинная таблица itemid, property, snapshot_ts, value (строка) для TOP-кодов.

    Значение свойства оставляем как есть (строкой) — для LightFM это категориальный
    токен. Многотокенные значения не разбиваем: берём как единый токен.
    """
    wanted = set(top_codes)
    parts = []
    for chunk in iter_item_properties(cfg):
        sub = chunk[chunk["property"].isin(wanted)]
        if not sub.empty:
            parts.append(sub[["itemid", "property", "timestamp", "value"]])
    raw = pd.concat(parts, ignore_index=True)
    raw["snapshot_ts"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
    raw = raw.drop(columns=["timestamp"])
    raw["property"] = raw["property"].astype(str)
    raw["value"] = raw["value"].astype(str)
    raw = raw.drop_duplicates(["itemid", "property", "snapshot_ts"])
    return raw.sort_values(["property", "itemid", "snapshot_ts"]).reset_index(drop=True)


def save_top_property_snapshots(df: pd.DataFrame, cfg: Config) -> None:
    cfg.paths.interim_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.paths.interim_dir / "top_property_snapshots.parquet", index=False)


def load_top_property_snapshots(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.paths.interim_dir / "top_property_snapshots.parquet")


def property_values_as_of(
    snapshots: pd.DataFrame, at: pd.Timestamp, code: str
) -> pd.Series:
    """itemid -> значение свойства ``code`` по последнему срезу с snapshot_ts <= at."""
    snap = snapshots[
        (snapshots["property"] == code) & (snapshots["snapshot_ts"] <= at)
    ]
    if snap.empty:
        return pd.Series(dtype="object", name=code)
    latest = snap.sort_values("snapshot_ts").groupby("itemid")["value"].last()
    latest.name = code
    latest.index.name = "itemid"
    return latest

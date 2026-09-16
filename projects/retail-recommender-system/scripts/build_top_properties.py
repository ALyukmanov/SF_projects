"""TOP-20 самых распространённых property-кодов товаров.

Запуск: python scripts/build_top_properties.py

Считаются ДВА списка (метод одинаков — частота строк item_properties среди
срезов с snapshot_ts <= cutoff):

  A. cutoff = 2015-07-01  — зафиксированная точка отсечения генератора факторов
     проекта. Это основной TOP-20.
  B. cutoff = 2015-08-22  — начало validation-окна. Диагностический список для
     train-window экспериментов с моделями. Один список другим не подменяется.

Коды в RetailRocket захешированы, кроме литералов `categoryid` и `available`.

Пишет:
  data/interim/top_property_codes.json          — оба списка
  data/interim/top_property_snapshots.parquet   — as-of значения кодов A + B
"""
from __future__ import annotations

import json

import _bootstrap  # noqa: F401
import pandas as pd

from retail_recommender.config import load_config
from retail_recommender.preprocessing.top_properties import (
    build_top_property_snapshots,
    compute_top_property_codes,
    save_top_property_snapshots,
    top_properties_path,
)

FEATURE_CUTOFF = "2015-07-01"
VAL_WINDOW_START = "2015-08-22"


def main() -> int:
    cfg = load_config()
    cfg.paths.interim_dir.mkdir(parents=True, exist_ok=True)

    a_codes, a_counts = compute_top_property_codes(
        cfg, pd.Timestamp(FEATURE_CUTOFF, tz="UTC"), top_n=20
    )
    b_codes, b_counts = compute_top_property_codes(
        cfg, pd.Timestamp(VAL_WINDOW_START, tz="UTC"), top_n=20
    )

    payload = {
        "primary": {
            "cutoff": FEATURE_CUTOFF,
            "role": "feature-generator TOP-20 (зафиксированный feature cutoff проекта)",
            "top_codes": a_codes,
            "top_counts": {c: a_counts[c] for c in a_codes},
            "n_distinct_codes": len(a_counts),
        },
        "val_window_diagnostic": {
            "cutoff": VAL_WINDOW_START,
            "role": "train-window diagnostic для экспериментов с моделями",
            "top_codes": b_codes,
            "top_counts": {c: b_counts[c] for c in b_codes},
            "n_distinct_codes": len(b_counts),
        },
        "lists_identical": a_codes == b_codes,
        "only_in_primary": [c for c in a_codes if c not in b_codes],
        "only_in_val_window": [c for c in b_codes if c not in a_codes],
        # обратная совместимость: старые потребители читали "top_codes"/"cutoff"
        "cutoff": FEATURE_CUTOFF,
        "top_codes": a_codes,
        "top_counts": {c: a_counts[c] for c in a_codes},
        "top_n": 20,
        "n_distinct_codes": len(a_counts),
    }
    top_properties_path(cfg).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    union = sorted(set(a_codes) | set(b_codes))
    snap = build_top_property_snapshots(cfg, union)
    save_top_property_snapshots(snap, cfg)

    print("A. TOP-20 @", FEATURE_CUTOFF, "(основной, feature cutoff проекта):")
    for i, c in enumerate(a_codes, 1):
        print(f"  {i:2d}. {c:>12}  строк <= cutoff: {a_counts[c]}")
    print("\nB. TOP-20 @", VAL_WINDOW_START, "(диагностический, train-window):")
    for i, c in enumerate(b_codes, 1):
        print(f"  {i:2d}. {c:>12}  строк <= cutoff: {b_counts[c]}")
    print(f"\nсписки идентичны: {payload['lists_identical']}")
    print(f"только в A: {payload['only_in_primary']}   только в B: {payload['only_in_val_window']}")
    print(f"снимки значений (A + B, {len(union)} кодов): {len(snap)} строк, "
          f"{snap['itemid'].nunique()} товаров")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Сборка processed/interim данных из сырых CSV.

Запуск: python scripts/build_dataset.py

Создаёт:
  data/processed/interactions.parquet        — event-level, дедуп, веса, is_bot, split
  data/interim/item_metadata_snapshots.parquet — categoryid/available по неделям
  data/interim/category_paths.parquet        — путь категории до корня
  reports/build_report.md                    — что изменилось от raw к processed
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import pandas as pd

from retail_recommender.config import load_config
from retail_recommender.preprocessing.categories import build_category_paths, save_category_paths
from retail_recommender.preprocessing.interactions import build_interactions, save_interactions
from retail_recommender.preprocessing.properties import (
    build_metadata_snapshots,
    load_metadata_snapshots,
    save_metadata_snapshots,
)
from retail_recommender.split.temporal import assert_forward_only, describe_windows


def main() -> int:
    ap = argparse.ArgumentParser(description="Сборка processed/interim данных.")
    ap.add_argument("--skip-metadata", action="store_true",
                    help="не перечитывать item_properties, взять готовый parquet")
    args = ap.parse_args()

    cfg = load_config()
    cfg.paths.interim_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] interactions ...")
    interactions, stats = build_interactions(cfg)
    save_interactions(interactions, cfg)
    assert_forward_only(interactions, cfg)
    win_df = describe_windows(interactions, cfg)

    if args.skip_metadata:
        print("[2/3] item metadata snapshots — пропуск, читаю готовый parquet ...")
        meta = load_metadata_snapshots(cfg)
    else:
        print("[2/3] item metadata snapshots ...")
        meta = build_metadata_snapshots(cfg)
        save_metadata_snapshots(meta, cfg)

    print("[3/3] category paths ...")
    paths = build_category_paths(cfg)
    save_category_paths(paths, cfg)

    meta_cat = meta[meta["property"] == "categoryid"]
    meta_avail = meta[meta["property"] == "available"]
    snap_times = sorted(meta["snapshot_ts"].dt.date.astype(str).unique().tolist())

    report = []
    report.append("# Build report\n")
    report.append("## interactions.parquet\n")
    report.append(f"- сырых строк events.csv: {stats.raw_rows}")
    report.append(f"- удалено полных дубликатов: {stats.full_duplicates_removed}")
    report.append(f"- строк после дедупа: {stats.processed_rows}")
    report.append(f"- визиторов: {stats.n_visitors}, товаров: {stats.n_items}")
    report.append(f"- аномально активных визиторов (> {cfg.bot_event_threshold} событий): "
                  f"{stats.n_bots}; их доля событий: {stats.bot_event_share:.4%}")
    report.append(f"- события по split: {json.dumps(stats.split_counts, ensure_ascii=False)}")
    report.append("")
    report.append("## Окна временного split\n")
    report.append("```")
    report.append(win_df.to_string(index=False))
    report.append("```")
    report.append("")
    report.append("## item_metadata_snapshots.parquet\n")
    report.append(f"- строк: {len(meta)} (categoryid: {len(meta_cat)}, available: {len(meta_avail)})")
    report.append(f"- товаров с категорией: {meta_cat['itemid'].nunique()}")
    report.append(f"- недельных срезов: {len(snap_times)} ({snap_times[0]} .. {snap_times[-1]})")
    report.append(f"- политика as-of: срезов до {cfg.properties_first_snapshot} нет, "
                  f"для более ранних моментов признак = UNKNOWN (pd.NA), forward-fill не применяется")
    report.append("")
    report.append("## category_paths.parquet\n")
    report.append(f"- категорий: {len(paths)}, корней: {(paths['depth'] == 0).sum()}, "
                  f"макс. глубина: {paths['depth'].max()}")
    report.append("")

    out_path = cfg.paths.reports_dir / "build_report.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nOK. Отчёт: {out_path}")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

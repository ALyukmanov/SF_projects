"""Дерево категорий: путь до корня, глубина, корневая категория."""
from __future__ import annotations

import pandas as pd

from second_year.config import Config
from second_year.data.load import load_category_tree

MAX_DEPTH_GUARD = 50  # защита от неожиданных циклов


def build_category_paths(cfg: Config) -> pd.DataFrame:
    tree = load_category_tree(cfg)
    parent = {int(c): (int(p) if pd.notna(p) else None)
              for c, p in zip(tree["categoryid"], tree["parentid"])}

    rows = []
    for cat in parent:
        path = [cat]
        node = cat
        for _ in range(MAX_DEPTH_GUARD):
            nxt = parent.get(node)
            if nxt is None:
                break
            path.append(nxt)
            node = nxt
        rows.append({
            "categoryid": cat,
            "root": path[-1],
            "depth": len(path) - 1,
            "path": path,
        })
    return pd.DataFrame(rows).sort_values("categoryid").reset_index(drop=True)


def save_category_paths(df: pd.DataFrame, cfg: Config) -> None:
    cfg.paths.interim_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.paths.interim_dir / "category_paths.parquet", index=False)


def load_category_paths(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.paths.interim_dir / "category_paths.parquet")

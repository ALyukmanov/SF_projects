"""Baseline 2 — популярность внутри категории интереса пользователя.

Категория интереса определяется ТОЛЬКО по истории до cutoff:
берётся последний товар в истории и его categoryid на момент того события
(as-of, без заглядывания в будущее). Категория будущего target-товара
не используется.
Внутри категории — популярные train-товары этой категории, затем добор
глобальной популярностью.
"""
from __future__ import annotations

import pandas as pd

from retail_recommender.config import Config
from retail_recommender.evaluation.protocol import EvalTask, UserHistory
from retail_recommender.models.base import backfill, take_top_n, weighted_popularity
from retail_recommender.preprocessing.properties import as_of_lookup, load_metadata_snapshots


class CategoryPopularity:
    name = "category_popularity"

    def __init__(self) -> None:
        self.global_ranked: list[int] = []
        self.category_ranked: dict[int, list[int]] = {}
        self.user_last_category: dict[int, int] = {}
        self._snapshots: pd.DataFrame | None = None

    def fit(self, task: EvalTask, cfg: Config) -> "CategoryPopularity":
        self._snapshots = load_metadata_snapshots(cfg)
        pop = weighted_popularity(task.fit_events)
        self.global_ranked = [int(i) for i in pop.index]

        # категория каждого train-товара на момент конца train-окна (as-of)
        train_end = task.fit_events["ts"].max()
        q = pd.DataFrame({"itemid": pop.index.astype(int)})
        q["ts"] = train_end
        cats = as_of_lookup(self._snapshots, q, "categoryid")
        by_cat: dict[int, list[int]] = {}
        for item, cat in zip(q["itemid"].tolist(), cats.tolist()):
            if pd.isna(cat):
                continue
            by_cat.setdefault(int(cat), []).append(int(item))
        # pop уже отсортирована по убыванию, порядок внутри категории сохраняется
        self.category_ranked = by_cat
        return self

    def prepare(self, task: EvalTask, cfg: Config) -> None:
        assert self._snapshots is not None, "fit должен быть вызван раньше prepare"
        rows = [
            {"user": u, "itemid": h.last_item, "ts": h.last_ts}
            for u, h in task.history.items()
            if h.last_item is not None and h.last_ts is not None
        ]
        if not rows:
            self.user_last_category = {}
            return
        q = pd.DataFrame(rows)
        cats = as_of_lookup(self._snapshots, q, "categoryid")
        self.user_last_category = {
            int(u): int(c)
            for u, c in zip(q["user"].tolist(), cats.tolist())
            if pd.notna(c)
        }

    def recommend(self, user_id: int, history: UserHistory, n: int,
                  exclude: set[int]) -> list[int]:
        cat = self.user_last_category.get(user_id)
        if cat is not None and cat in self.category_ranked:
            primary = take_top_n(self.category_ranked[cat], n, exclude)
            return backfill(primary, self.global_ranked, n, exclude)
        return take_top_n(self.global_ranked, n, exclude)

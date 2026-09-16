"""Baseline 3 — item-item co-occurrence (explainable).

Только train-события. Co-occurrence = совместное появление двух товаров
в истории одного визитора (в пределах train-окна). Похожесть — косинус
по co-occurrence-счётчикам. Хранятся только top-N соседей на товар
(sparse-словарь), плотная матрица item×item не строится.

Для пользователя: берём его последние сильные/недавние товары истории,
суммируем оценки их соседей, исключаем уже виденное, добираем глобальной
популярностью.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from retail_recommender.config import Config
from retail_recommender.evaluation.protocol import EvalTask, UserHistory
from retail_recommender.models.base import backfill, weighted_popularity


class ItemItemCooccurrence:
    name = "item_item_cooccurrence"

    def __init__(self) -> None:
        self.global_ranked: list[int] = []
        self.neighbours: dict[int, list[tuple[int, float]]] = {}
        self._query_items = 20

    def fit(self, task: EvalTask, cfg: Config) -> "ItemItemCooccurrence":
        co = cfg.cooccurrence
        self._query_items = int(co["user_query_items"])
        min_items = int(co["min_user_items"])
        max_items = int(co["max_user_items"])
        top_k = int(co["top_neighbours"])

        pop = weighted_popularity(task.fit_events)
        self.global_ranked = [int(i) for i in pop.index]

        ev = task.fit_events[["visitorid", "itemid"]].drop_duplicates()
        n_items_per_user = ev.groupby("visitorid")["itemid"].transform("size")
        ev = ev[(n_items_per_user >= min_items) & (n_items_per_user <= max_items)]
        if ev.empty:
            return self

        users = ev["visitorid"].astype("category")
        items = ev["itemid"].astype("category")
        item_codes = items.cat.categories.to_numpy().astype(int)
        mat = sparse.coo_matrix(
            (np.ones(len(ev), dtype=np.float32), (users.cat.codes.to_numpy(),
                                                  items.cat.codes.to_numpy())),
            shape=(users.cat.categories.size, items.cat.categories.size),
        ).tocsr()

        cooc = (mat.T @ mat).tocsr()          # item x item co-occurrence counts
        diag = np.asarray(cooc.diagonal(), dtype=np.float64)  # = частота товара среди этих юзеров
        norm = np.sqrt(np.where(diag > 0, diag, 1.0))

        cooc.setdiag(0)
        cooc.eliminate_zeros()

        neighbours: dict[int, list[tuple[int, float]]] = {}
        indptr, indices, data = cooc.indptr, cooc.indices, cooc.data
        for i in range(cooc.shape[0]):
            start, end = indptr[i], indptr[i + 1]
            if start == end:
                continue
            cols = indices[start:end]
            sims = data[start:end] / (norm[i] * norm[cols])
            if len(cols) > top_k:
                sel = np.argpartition(sims, -top_k)[-top_k:]
                cols, sims = cols[sel], sims[sel]
            order = np.argsort(sims)[::-1]
            neighbours[int(item_codes[i])] = [
                (int(item_codes[c]), float(s)) for c, s in zip(cols[order], sims[order])
            ]
        self.neighbours = neighbours
        return self

    def prepare(self, task: EvalTask, cfg: Config) -> None:
        return None

    def recommend(self, user_id: int, history: UserHistory, n: int,
                  exclude: set[int]) -> list[int]:
        if history.length == 0 or not self.neighbours:
            return backfill([], self.global_ranked, n, exclude)

        # запрос: последние товары истории, с приоритетом более сильных событий
        pairs = list(zip(history.items, history.weights))[-self._query_items:]
        scores: dict[int, float] = {}
        for item, w in pairs:
            for nb, sim in self.neighbours.get(item, ()):
                if nb in exclude or nb in history.seen:
                    continue
                scores[nb] = scores.get(nb, 0.0) + sim * w

        ranked = [it for it, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        primary = ranked[:n]
        return backfill(primary, self.global_ranked, n, exclude)

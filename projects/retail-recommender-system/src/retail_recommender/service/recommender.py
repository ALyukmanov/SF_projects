"""Загрузка экспортированной модели и обёртка для инференса.

Алгоритм рекомендаций НЕ переписан: используется
``retail_recommender.models.cooccurrence.ItemItemCooccurrence.recommend`` — та же логика,
что и при обучении/оценке. Здесь только:
  * восстановление обученного состояния из ``model.npz`` (соседи, популярность);
  * подстановка истории визитора из артефакта;
  * определение флага ``fallback``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retail_recommender.evaluation.protocol import UserHistory
from retail_recommender.models.base import backfill
from retail_recommender.models.cooccurrence import ItemItemCooccurrence
from retail_recommender.service.artifact import ArrayNeighbours


@dataclass
class RecommendResult:
    recommendations: list[int]
    fallback: bool
    known_user: bool
    history_size: int


class ServiceRecommender:
    """Инференс-обёртка над экспортированной co-occurrence моделью."""

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        self.meta: dict = json.loads((model_dir / "meta.json").read_text("utf-8"))

        data = np.load(model_dir / "model.npz")
        self._neighbours = ArrayNeighbours(
            data["neighbour_item_ids"],
            data["neighbour_offsets"],
            data["neighbour_ids"],
            data["neighbour_sims"],
        )
        self._global_ranked: list[int] = data["global_ranked"].tolist()
        self._catalog: frozenset[int] = frozenset(int(x) for x in data["catalog"].tolist())

        self._user_ids: np.ndarray = data["user_ids"]
        self._user_offsets: np.ndarray = data["user_offsets"]
        self._hist_items: np.ndarray = data["user_hist_items"]
        self._hist_weights: np.ndarray = data["user_hist_weights"]

        self.query_items = int(self.meta["query_items"])
        self.filter_seen = bool(self.meta["filter_seen"])
        self.default_n = int(self.meta["default_n"])
        self.max_n = int(self.meta["max_n"])

        # тот же класс, что и при обучении — только с загруженным состоянием
        self._model = ItemItemCooccurrence()
        self._model.neighbours = self._neighbours  # type: ignore[assignment]
        self._model.global_ranked = self._global_ranked
        self._model._query_items = self.query_items

    # ------------------------------------------------------------------ history
    def _history(self, visitorid: int) -> tuple[list[int], list[int]] | None:
        idx = int(np.searchsorted(self._user_ids, visitorid))
        if idx >= len(self._user_ids) or int(self._user_ids[idx]) != int(visitorid):
            return None
        s, e = int(self._user_offsets[idx]), int(self._user_offsets[idx + 1])
        return self._hist_items[s:e].tolist(), self._hist_weights[s:e].tolist()

    # ------------------------------------------------------------------ predict
    def recommend(self, visitorid: int, n: int | None = None) -> RecommendResult:
        n = self.default_n if n is None else max(1, min(int(n), self.max_n))
        hist = self._history(visitorid)

        if hist is None:
            user_hist = UserHistory([], [], [], None, None, set())
            known = False
        else:
            items, weights = hist
            user_hist = UserHistory(
                items=list(items),
                weights=list(weights),
                events=[],
                last_item=items[-1] if items else None,
                last_ts=None,
                seen=set(items),
            )
            known = True

        exclude = set(user_hist.seen) if self.filter_seen else set()
        recs = self._model.recommend(visitorid, user_hist, n=n, exclude=exclude)

        # fallback = выдача совпала с чистой популярностью (co-occurrence ничего не добавил)
        pop_only = backfill([], self._global_ranked, n, exclude)
        fallback = recs == pop_only

        return RecommendResult(
            recommendations=[int(x) for x in recs],
            fallback=bool(fallback),
            known_user=known,
            history_size=user_hist.length,
        )

    # ------------------------------------------------------------------ info
    def model_info(self) -> dict:
        m = self.meta
        return {
            "model_type": m["model_type"],
            "version": m["version"],
            "built_at": m["built_at"],
            "default_n": self.default_n,
            "max_n": self.max_n,
            "query_items": self.query_items,
            "filter_seen": self.filter_seen,
            "catalog_size": m["catalog_size"],
            "items_with_neighbours": m["n_items_with_neighbours"],
            "users_with_history": m["n_users_with_history"],
            "history_window_end": m["split"]["history_before"],
            "train_end": m["split"]["train_end"],
            "test_window": m["split"]["test"],
            "test_metrics": m.get("test_metrics", {}),
        }

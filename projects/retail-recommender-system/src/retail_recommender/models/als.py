"""Implicit-факторизации как рекомендатели: ALS и BPR (библиотека ``implicit``).

Матрица взаимодействий строится ТОЛЬКО из train-окна (``task.fit_events``).
Значение ячейки = сумма весов событий пары (visitorid, itemid) с текущими весами
проекта (view/addtocart/transaction). Для ALS это confidence по Hu et al.
(C = 1 + alpha * R), для BPR — вес положительного примера.

Пользователь без train-истории -> глобальная популярность (тот же fallback, что у
baseline). Политика исключения уже виденных train-товаров сохраняется
(``filter_already_liked_items``), как в общем протоколе оценки.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from retail_recommender.config import Config
from retail_recommender.evaluation.protocol import EvalTask, UserHistory
from retail_recommender.models.base import take_top_n, weighted_popularity


class _ImplicitMF:
    """Общая обвязка для моделей из ``implicit`` (ALS / BPR)."""

    name = "implicit_mf"

    def __init__(self, factors: int, iterations: int, random_state: int, alpha: float) -> None:
        self.factors = int(factors)
        self.iterations = int(iterations)
        self.random_state = int(random_state)
        self.alpha = float(alpha)
        self._pop_fallback: list[int] = []
        self._u_pos: dict[int, int] = {}
        self._item_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self._ui: sparse.csr_matrix | None = None
        self._model = None

    # --- переопределяется в наследниках ---
    def _make_model(self):  # pragma: no cover - тривиально
        raise NotImplementedError

    def fit(self, task: EvalTask, cfg: Config) -> "_ImplicitMF":
        ev = task.fit_events
        agg = ev.groupby(["visitorid", "itemid"])["weight"].sum().reset_index()
        user_ids = np.sort(ev["visitorid"].unique())
        item_ids = np.sort(np.fromiter(task.catalog, dtype=np.int64))
        self._u_pos = {int(u): i for i, u in enumerate(user_ids)}
        self._item_ids = item_ids
        i_pos = {int(it): i for i, it in enumerate(item_ids)}

        rows = agg["visitorid"].map(self._u_pos).to_numpy()
        cols = agg["itemid"].map(i_pos).to_numpy()
        keep = ~pd.isna(cols)
        rows = rows[keep].astype(np.int64)
        cols = cols[keep].astype(np.int64)
        vals = 1.0 + self.alpha * agg["weight"].to_numpy()[keep].astype(np.float32)
        self._ui = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(user_ids), len(item_ids))
        )

        self._model = self._make_model()
        try:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(1, "blas"):
                self._model.fit(self._ui, show_progress=False)
        except ImportError:
            self._model.fit(self._ui, show_progress=False)
        self._pop_fallback = [int(i) for i in weighted_popularity(ev).index]
        return self

    def prepare(self, task: EvalTask, cfg: Config) -> None:
        return None

    def recommend(
        self, user_id: int, history: UserHistory, n: int, exclude: set[int]
    ) -> list[int]:
        pos = self._u_pos.get(int(user_id))
        if pos is None or self._model is None or self._ui is None:
            return take_top_n(self._pop_fallback, n, exclude)

        ids, _ = self._model.recommend(
            pos,
            self._ui[pos],
            N=min(n + len(exclude) + 5, len(self._item_ids)),
            filter_already_liked_items=True,
        )
        ordered: list[int] = []
        seen: set[int] = set(exclude)
        for j in np.asarray(ids).ravel():
            it = int(self._item_ids[int(j)])
            if it in seen:
                continue
            seen.add(it)
            ordered.append(it)
            if len(ordered) >= n:
                break
        if len(ordered) < n:
            ordered = _backfill(ordered, self._pop_fallback, n, exclude)
        return ordered[:n]


class ALSRecommender(_ImplicitMF):
    name = "als"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 20,
        alpha: float = 1.0,
        random_state: int = 42,
    ) -> None:
        super().__init__(factors, iterations, random_state, alpha)
        self.regularization = float(regularization)
        self.config_str = f"factors={factors},reg={regularization},iters={iterations}"

    def _make_model(self):
        from implicit.als import AlternatingLeastSquares

        return AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            use_gpu=False,
        )


class BPRRecommender(_ImplicitMF):
    name = "bpr"

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        iterations: int = 100,
        random_state: int = 42,
    ) -> None:
        super().__init__(factors, iterations, random_state, alpha=1.0)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.config_str = (
            f"factors={factors},lr={learning_rate},reg={regularization},iters={iterations}"
        )

    def _make_model(self):
        from implicit.bpr import BayesianPersonalizedRanking

        return BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            use_gpu=False,
            verify_negative_samples=True,
        )


def _backfill(primary: list[int], fallback: list[int], n: int, exclude: set[int]) -> list[int]:
    out = list(primary)
    chosen = set(out) | exclude
    for it in fallback:
        if len(out) >= n:
            break
        if it in chosen:
            continue
        out.append(it)
        chosen.add(it)
    return out[:n]

"""Baseline 1 — глобальная популярность.

Список популярных товаров строится только по train-событиям.
Для холодных пользователей это основной fallback.
"""
from __future__ import annotations

from retail_recommender.config import Config
from retail_recommender.evaluation.protocol import EvalTask, UserHistory
from retail_recommender.models.base import take_top_n, weighted_popularity


class GlobalPopularity:
    name = "global_popularity"

    def __init__(self, time_decay: bool = False, half_life_days: float = 30.0) -> None:
        self.time_decay = time_decay
        self.half_life_days = half_life_days
        self.ranked_items: list[int] = []

    def fit(self, task: EvalTask, cfg: Config) -> "GlobalPopularity":
        ev = task.fit_events
        if self.time_decay:
            t_max = ev["ts"].max()
            age_days = (t_max - ev["ts"]).dt.total_seconds() / 86400.0
            decay = 0.5 ** (age_days / self.half_life_days)
            score = (ev["weight"] * decay).groupby(ev["itemid"]).sum().sort_values(ascending=False)
            self.ranked_items = [int(i) for i in score.index]
        else:
            self.ranked_items = [int(i) for i in weighted_popularity(ev).index]
        return self

    def prepare(self, task: EvalTask, cfg: Config) -> None:
        return None

    def recommend(self, user_id: int, history: UserHistory, n: int,
                  exclude: set[int]) -> list[int]:
        return take_top_n(self.ranked_items, n, exclude)

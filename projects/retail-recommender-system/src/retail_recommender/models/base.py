"""Базовый интерфейс рекомендателя и общий helper популярности."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from retail_recommender.config import Config
from retail_recommender.evaluation.protocol import EvalTask, UserHistory


class Recommender(Protocol):
    name: str

    def fit(self, task: EvalTask, cfg: Config) -> "Recommender": ...

    def prepare(self, task: EvalTask, cfg: Config) -> None:
        """Необязательный хук: батч-подготовка перед инференсом (по умолчанию no-op)."""

    def recommend(self, user_id: int, history: UserHistory, n: int,
                  exclude: set[int]) -> list[int]: ...


def weighted_popularity(fit_events: pd.DataFrame) -> pd.Series:
    """Взвешенная популярность товара по обучающим событиям (по убыванию).

    Считается ТОЛЬКО по переданным событиям (train-окно). Индекс — itemid,
    значение — суммарный вес событий.
    """
    pop = fit_events.groupby("itemid")["weight"].sum().sort_values(ascending=False)
    pop.index = pop.index.astype(int)
    return pop


def take_top_n(ranked_items: list[int], n: int, exclude: set[int]) -> list[int]:
    out: list[int] = []
    for item in ranked_items:
        if item in exclude:
            continue
        out.append(item)
        if len(out) >= n:
            break
    return out


def backfill(primary: list[int], fallback_ranked: list[int], n: int,
             exclude: set[int]) -> list[int]:
    out = list(primary)
    chosen = set(out) | exclude
    for item in fallback_ranked:
        if len(out) >= n:
            break
        if item in chosen:
            continue
        out.append(item)
        chosen.add(item)
    return out[:n]

"""Протокол offline-оценки.

Для каждого target-пользователя:
  история      = его события строго ДО начала evaluation-окна (только не-боты);
  ground truth = его сильные события (addtocart ∪ transaction) внутри окна
                 либо только transaction — в зависимости от режима.

Отчёты строятся в разрезах:
  ALL   — все target-пользователи (для cold-start работает fallback модели);
  WARM  — те, у кого есть хотя бы одно событие в истории до окна.
Дополнительно — по длине истории: 0, 1, 2–4, 5+ событий.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from retail_recommender.config import Config
from retail_recommender.evaluation.metrics import evaluate_recommendations
from retail_recommender.split.temporal import Window, slice_window, windows

TargetMode = str  # "strong" | "transaction"


@dataclass
class UserHistory:
    """Компактное представление истории одного пользователя до окна."""
    items: list[int]          # в хронологическом порядке
    weights: list[int]
    events: list[str]
    last_item: int | None
    last_ts: pd.Timestamp | None
    seen: set[int]

    @property
    def length(self) -> int:
        return len(self.items)


@dataclass
class EvalTask:
    window: Window
    mode: TargetMode
    ground_truth: dict[int, set[int]]
    history: dict[int, UserHistory]
    warm_users: set[int]
    catalog: set[int]           # товары, доступные модели (train fit-set)
    fit_events: pd.DataFrame     # события для обучения модели (train-часть)
    history_events: pd.DataFrame  # все не-бот события до начала окна (для инференса)

    @property
    def target_users(self) -> list[int]:
        return list(self.ground_truth)


HISTORY_SEGMENTS = {
    "hist=0": lambda n: n == 0,
    "hist=1": lambda n: n == 1,
    "hist=2-4": lambda n: 2 <= n <= 4,
    "hist>=5": lambda n: n >= 5,
}


def _target_events(cfg: Config, mode: TargetMode) -> set[str]:
    return set(cfg.strong_events) if mode == "strong" else set(cfg.transaction_events)


def build_eval_task(
    interactions: pd.DataFrame,
    cfg: Config,
    window_name: str,
    mode: TargetMode,
) -> EvalTask:
    if window_name not in ("val", "test"):
        raise ValueError("оценка только на val или test")
    w = windows(cfg)
    win = w[window_name]

    clean = interactions[~interactions["is_bot"]]

    # обучающие события: только train-окно (популярность/co-occurrence — train-derived).
    fit_events = slice_window(clean, w["train"])
    catalog = set(fit_events["itemid"].unique().tolist())

    # история для инференса: все не-бот события строго до начала окна.
    history_events = clean[clean["ts"] < win.start]

    window_events = slice_window(clean, win)
    tgt_types = _target_events(cfg, mode)
    gt_df = window_events[window_events["event"].isin(tgt_types)]
    ground_truth: dict[int, set[int]] = {
        int(u): set(map(int, g["itemid"].unique()))
        for u, g in gt_df.groupby("visitorid")
    }

    target_users = set(ground_truth)
    hist_sub = history_events[history_events["visitorid"].isin(target_users)]
    hist_sub = hist_sub.sort_values(["visitorid", "timestamp"])

    history: dict[int, UserHistory] = {}
    for u, g in hist_sub.groupby("visitorid"):
        items = list(map(int, g["itemid"]))
        history[int(u)] = UserHistory(
            items=items,
            weights=list(map(int, g["weight"])),
            events=list(g["event"].astype(str)),
            last_item=items[-1] if items else None,
            last_ts=g["ts"].iloc[-1] if len(g) else None,
            seen=set(items),
        )
    for u in target_users:
        history.setdefault(int(u), UserHistory([], [], [], None, None, set()))

    warm_users = {u for u, h in history.items() if h.length > 0}

    return EvalTask(
        window=win,
        mode=mode,
        ground_truth=ground_truth,
        history=history,
        warm_users=warm_users,
        catalog=catalog,
        fit_events=fit_events,
        history_events=history_events,
    )


def evaluate_model(model, task: EvalTask, cfg: Config) -> dict[str, dict[str, float]]:
    """Возвращает словарь: имя-среза -> {метрика: значение}."""
    k_values = cfg.evaluation["k_values"]
    max_k = max(k_values)
    filter_seen = bool(cfg.evaluation["filter_seen"])
    catalog_size = len(task.catalog)

    if hasattr(model, "prepare"):
        model.prepare(task, cfg)

    recs: dict[int, list[int]] = {}
    for u in task.target_users:
        h = task.history[u]
        exclude = h.seen if filter_seen else set()
        recs[u] = model.recommend(u, h, n=max_k, exclude=exclude)

    segments: dict[str, list[int]] = {
        "ALL": task.target_users,
        "WARM": [u for u in task.target_users if u in task.warm_users],
    }
    for seg_name, pred in HISTORY_SEGMENTS.items():
        segments[seg_name] = [u for u in task.target_users if pred(task.history[u].length)]

    out: dict[str, dict[str, float]] = {}
    for seg_name, users in segments.items():
        gt = {u: task.ground_truth[u] for u in users}
        rc = {u: recs.get(u, []) for u in users}
        out[seg_name] = evaluate_recommendations(rc, gt, k_values, catalog_size)
    return out


def diagnostic_no_filter(model, task: EvalTask, cfg: Config) -> dict[str, float]:
    """Та же оценка на срезе ALL, но БЕЗ исключения уже виденных train-товаров."""
    k_values = cfg.evaluation["k_values"]
    max_k = max(k_values)
    if hasattr(model, "prepare"):
        model.prepare(task, cfg)
    recs = {
        u: model.recommend(u, task.history[u], n=max_k, exclude=set())
        for u in task.target_users
    }
    return evaluate_recommendations(recs, task.ground_truth, k_values, len(task.catalog))

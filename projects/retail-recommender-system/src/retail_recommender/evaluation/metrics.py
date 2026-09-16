"""Offline-метрики ранжирования для top-N рекомендаций.

Все функции работают со списком рекомендаций (по убыванию релевантности)
и множеством релевантных элементов (ground truth).
Дубликаты в рекомендациях удаляются с сохранением порядка (первое вхождение).
"""
from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence


def _dedup(recommended: Iterable[Hashable]) -> list[Hashable]:
    seen: set[Hashable] = set()
    out: list[Hashable] = []
    for item in recommended:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _prepare(recommended: Iterable[Hashable], relevant: Iterable[Hashable], k: int):
    if k <= 0:
        raise ValueError("k должно быть положительным")
    recs = _dedup(recommended)[:k]
    rel = set(relevant)
    return recs, rel


def recall_at_k(recommended: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    recs, rel = _prepare(recommended, relevant, k)
    if not rel:
        return 0.0
    hits = sum(1 for item in recs if item in rel)
    return hits / len(rel)


def precision_at_k(recommended: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    recs, rel = _prepare(recommended, relevant, k)
    if not rel:
        return 0.0
    hits = sum(1 for item in recs if item in rel)
    return hits / k


def hit_rate_at_k(recommended: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    recs, rel = _prepare(recommended, relevant, k)
    if not rel:
        return 0.0
    return 1.0 if any(item in rel for item in recs) else 0.0


def ndcg_at_k(recommended: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    recs, rel = _prepare(recommended, relevant, k)
    if not rel:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(recs) if item in rel)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    recs, rel = _prepare(recommended, relevant, k)
    if not rel:
        return 0.0
    hits = 0
    score = 0.0
    for i, item in enumerate(recs):
        if item in rel:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(rel), k)


_PER_USER = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "hit_rate": hit_rate_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
}


def evaluate_recommendations(
    recommendations: dict[Hashable, Sequence[Hashable]],
    ground_truth: dict[Hashable, Iterable[Hashable]],
    k_values: Iterable[int],
    catalog_size: int | None = None,
) -> dict[str, float]:
    """Усреднение метрик по пользователям, у которых непустой ground truth.

    Пользователи без ground truth в усреднение не входят.
    Пользователь без рекомендаций считается как нулевой результат.
    """
    k_values = list(k_values)
    users = [u for u, gt in ground_truth.items() if set(gt)]
    result: dict[str, float] = {"n_users_evaluated": float(len(users))}
    if not users:
        return result

    for k in k_values:
        sums = {name: 0.0 for name in _PER_USER}
        rec_union: set[Hashable] = set()
        for u in users:
            recs = recommendations.get(u, [])
            gt = ground_truth[u]
            for name, fn in _PER_USER.items():
                sums[name] += fn(recs, gt, k)
            rec_union.update(_dedup(recs)[:k])
        for name in _PER_USER:
            result[f"{name}@{k}"] = sums[name] / len(users)
        if catalog_size:
            result[f"coverage@{k}"] = len(rec_union) / catalog_size
    return result

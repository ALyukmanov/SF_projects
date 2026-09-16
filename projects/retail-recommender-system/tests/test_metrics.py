"""Ручные примеры для метрик ранжирования."""
import math

import pytest

from retail_recommender.evaluation.metrics import (
    average_precision_at_k,
    evaluate_recommendations,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_no_ground_truth_returns_zero():
    for fn in (recall_at_k, precision_at_k, hit_rate_at_k, ndcg_at_k, average_precision_at_k):
        assert fn([1, 2, 3], [], k=10) == 0.0


def test_fewer_than_k_recommendations():
    # 2 рекомендации, k=10, один релевантный и он найден
    assert recall_at_k([5, 9], {5}, k=10) == 1.0
    assert hit_rate_at_k([5, 9], {5}, k=10) == 1.0
    assert precision_at_k([5, 9], {5}, k=10) == pytest.approx(1 / 10)


def test_duplicate_recommendations_are_collapsed():
    # дубли не должны накручивать recall
    assert recall_at_k([7, 7, 7], {7, 8}, k=10) == pytest.approx(0.5)
    # дедуп идёт до обрезки по k: [7,7,8] -> [7,8], оба релевантны при k=2
    assert recall_at_k([7, 7, 8], {7, 8}, k=2) == pytest.approx(1.0)
    # а дубль не занимает слот, который достался бы нерелевантному соседу
    assert recall_at_k([7, 7, 9], {7, 8}, k=2) == pytest.approx(0.5)


def test_unknown_item_in_recommendations():
    assert recall_at_k([999, 1], {1}, k=10) == 1.0
    assert precision_at_k([999, 1], {1}, k=2) == pytest.approx(0.5)


def test_one_relevant_item_ndcg_position():
    # релевантный на 2-й позиции: DCG = 1/log2(3); IDCG = 1/log2(2) = 1
    got = ndcg_at_k([10, 42, 11], {42}, k=10)
    assert got == pytest.approx(1.0 / math.log2(3))


def test_multiple_relevant_items_recall_and_map():
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 6}
    assert recall_at_k(recommended, relevant, k=5) == pytest.approx(2 / 3)
    # AP@5: попадания на позициях 2 и 4 -> precision 1/2 и 2/4; делим на min(3,5)=3
    assert average_precision_at_k(recommended, relevant, k=5) == pytest.approx((0.5 + 0.5) / 3)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)


def test_hit_rate_binary():
    assert hit_rate_at_k([1, 2, 3], {9}, k=3) == 0.0
    assert hit_rate_at_k([1, 2, 3], {3}, k=3) == 1.0


def test_evaluate_recommendations_skips_users_without_gt():
    recs = {"u1": [1, 2, 3], "u2": [4, 5, 6]}
    gt = {"u1": {2}, "u2": set()}  # u2 без ground truth
    out = evaluate_recommendations(recs, gt, k_values=[3], catalog_size=100)
    assert out["n_users_evaluated"] == 1.0
    assert out["recall@3"] == pytest.approx(1.0)  # только u1
    assert out["coverage@3"] == pytest.approx(3 / 100)


def test_evaluate_recommendations_missing_user_is_zero():
    recs = {}  # модель ничего не выдала
    gt = {"u1": {1}}
    out = evaluate_recommendations(recs, gt, k_values=[5], catalog_size=10)
    assert out["recall@5"] == 0.0
    assert out["n_users_evaluated"] == 1.0


def test_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k([1], {1}, k=0)

"""Модели (ALS/BPR): обучение только по train, форма выдачи, отсутствие неизвестных
itemid, корректный fallback для холодных, детерминизм ALS."""
import numpy as np
import pytest

from retail_recommender.evaluation.protocol import build_eval_task
from retail_recommender.models import GlobalPopularity
from retail_recommender.models.als import ALSRecommender, BPRRecommender

# train-события в июне-июле (train-окно), val-события в конце августа
TRAIN = [
    {"visitorid": 1, "itemid": 10, "date": "2015-06-01", "event": "view"},
    {"visitorid": 1, "itemid": 11, "date": "2015-06-02", "event": "addtocart"},
    {"visitorid": 1, "itemid": 12, "date": "2015-06-03", "event": "view"},
    {"visitorid": 2, "itemid": 10, "date": "2015-06-04", "event": "view"},
    {"visitorid": 2, "itemid": 11, "date": "2015-06-05", "event": "transaction"},
    {"visitorid": 3, "itemid": 12, "date": "2015-06-06", "event": "view"},
    {"visitorid": 3, "itemid": 13, "date": "2015-06-07", "event": "addtocart"},
    {"visitorid": 4, "itemid": 10, "date": "2015-06-08", "event": "view"},
    {"visitorid": 4, "itemid": 13, "date": "2015-06-09", "event": "view"},
]
VAL = [
    {"visitorid": 1, "itemid": 13, "date": "2015-08-25", "event": "transaction"},
    {"visitorid": 2, "itemid": 12, "date": "2015-08-26", "event": "addtocart"},
    {"visitorid": 9, "itemid": 10, "date": "2015-08-27", "event": "addtocart"},
]


@pytest.fixture
def task(cfg, make_interactions):
    return build_eval_task(make_interactions(TRAIN + VAL), cfg, "val", "strong")


@pytest.mark.parametrize("cls", [ALSRecommender, BPRRecommender])
def test_recommend_shape_and_known_items(cfg, task, cls):
    model = cls(factors=4, iterations=3, random_state=0).fit(task, cfg)
    catalog = task.catalog
    n = 3  # синтетический каталог маленький
    for u in task.target_users:
        recs = model.recommend(u, task.history[u], n=n, exclude=set())
        assert isinstance(recs, list)
        assert len(recs) <= n
        assert len(recs) == len(set(recs))  # без дублей
        assert all(isinstance(x, int) for x in recs)
        # неизвестных товаров быть не может: всё из train-каталога либо fallback (тоже train)
        assert all(x in catalog for x in recs)


@pytest.mark.parametrize("cls", [ALSRecommender, BPRRecommender])
def test_cold_user_falls_back_to_popularity(cfg, task, cls):
    model = cls(factors=4, iterations=3, random_state=0).fit(task, cfg)
    pop = GlobalPopularity().fit(task, cfg)
    cold_user = 12345  # нет в train
    hist = task.history.get(cold_user)
    from retail_recommender.evaluation.protocol import UserHistory

    hist = hist or UserHistory([], [], [], None, None, set())
    assert model.recommend(cold_user, hist, n=5, exclude=set()) == pop.recommend(
        cold_user, hist, n=5, exclude=set()
    )


@pytest.mark.parametrize("cls", [ALSRecommender, BPRRecommender])
def test_exclude_is_respected(cfg, task, cls):
    model = cls(factors=4, iterations=3, random_state=0).fit(task, cfg)
    ban = {10, 11}
    for u in task.target_users:
        recs = model.recommend(u, task.history[u], n=5, exclude=ban)
        assert not (set(recs) & ban)


def test_als_is_deterministic(cfg, task):
    a = ALSRecommender(factors=8, iterations=5, random_state=42).fit(task, cfg)
    b = ALSRecommender(factors=8, iterations=5, random_state=42).fit(task, cfg)
    for u in task.target_users:
        assert a.recommend(u, task.history[u], 5, set()) == b.recommend(
            u, task.history[u], 5, set()
        )


def test_fit_uses_only_train_window(cfg, make_interactions):
    """Порча событий val/test не меняет обученную ALS-модель (обучение только на train)."""
    base = make_interactions(TRAIN + VAL)
    corrupt = make_interactions(
        TRAIN
        + [
            {"visitorid": 1, "itemid": 777, "date": "2015-08-25", "event": "transaction"},
            {"visitorid": 5, "itemid": 888, "date": "2015-09-10", "event": "transaction"},
        ]
    )
    t0 = build_eval_task(base, cfg, "val", "strong")
    t1 = build_eval_task(corrupt, cfg, "val", "strong")
    m0 = ALSRecommender(factors=8, iterations=5, random_state=1).fit(t0, cfg)
    m1 = ALSRecommender(factors=8, iterations=5, random_state=1).fit(t1, cfg)
    assert np.array_equal(m0._item_ids, m1._item_ids)
    from retail_recommender.evaluation.protocol import UserHistory

    empty = UserHistory([], [], [], None, None, set())
    for u in m0._u_pos:  # обученные train-пользователи
        h = t0.history.get(u, empty)
        assert m0.recommend(u, h, 3, set()) == m1.recommend(u, h, 3, set())


def test_top20_property_artifact_if_present(cfg):
    """Если артефакт TOP-20 построен — в нём ровно 20 кодов."""
    from retail_recommender.preprocessing.top_properties import top_properties_path

    p = top_properties_path(cfg)
    if not p.exists():
        pytest.skip("top_property_codes.json не построен (нужен scripts/build_top_properties.py)")
    import json

    data = json.loads(p.read_text("utf-8"))
    assert len(data["top_codes"]) == 20

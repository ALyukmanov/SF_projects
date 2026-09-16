"""Инварианты «информация течёт только вперёд»."""
from retail_recommender.evaluation.protocol import build_eval_task, evaluate_model
from retail_recommender.models import GlobalPopularity, ItemItemCooccurrence
from retail_recommender.split.temporal import assert_forward_only, windows

TRAIN = [
    {"visitorid": 1, "itemid": 10, "date": "2015-06-01", "event": "view"},
    {"visitorid": 1, "itemid": 11, "date": "2015-06-02", "event": "addtocart"},
    {"visitorid": 2, "itemid": 10, "date": "2015-06-03", "event": "view"},
    {"visitorid": 2, "itemid": 12, "date": "2015-06-04", "event": "transaction"},
    {"visitorid": 3, "itemid": 11, "date": "2015-06-05", "event": "view"},
    {"visitorid": 3, "itemid": 12, "date": "2015-06-06", "event": "view"},
    {"visitorid": 4, "itemid": 10, "date": "2015-07-01", "event": "view"},
]
VAL = [
    {"visitorid": 1, "itemid": 12, "date": "2015-08-25", "event": "transaction"},
    {"visitorid": 2, "itemid": 11, "date": "2015-08-26", "event": "addtocart"},
]
TEST = [
    {"visitorid": 3, "itemid": 10, "date": "2015-09-10", "event": "transaction"},
]


def _fit_pop_and_cooc(interactions, cfg, window):
    task = build_eval_task(interactions, cfg, window, "strong")
    pop = GlobalPopularity().fit(task, cfg).ranked_items
    cooc = ItemItemCooccurrence().fit(task, cfg).neighbours
    return pop, cooc


def test_boundaries_are_ordered(cfg, make_interactions):
    df = make_interactions(TRAIN + VAL + TEST)
    assert_forward_only(df, cfg)
    w = windows(cfg)
    assert w["train"].end_exclusive <= w["val"].start
    assert w["val"].end_exclusive <= w["test"].start


def test_train_derived_stats_invariant_to_future_corruption(cfg, make_interactions):
    base = make_interactions(TRAIN + VAL + TEST)
    pop0, cooc0 = _fit_pop_and_cooc(base, cfg, "val")

    corrupted = make_interactions(TRAIN + [
        {"visitorid": 1, "itemid": 999, "date": "2015-08-25", "event": "transaction"},
        {"visitorid": 2, "itemid": 888, "date": "2015-08-27", "event": "transaction"},
        {"visitorid": 9, "itemid": 999, "date": "2015-09-11", "event": "transaction"},
        {"visitorid": 9, "itemid": 888, "date": "2015-09-12", "event": "addtocart"},
    ])
    pop1, cooc1 = _fit_pop_and_cooc(corrupted, cfg, "val")

    assert pop0 and cooc0, "фикстура должна давать непустые train-статистики"
    assert pop0 == pop1, "популярность train изменилась при порче val/test"
    assert cooc0 == cooc1, "co-occurrence train изменился при порче val/test"


def test_val_metrics_invariant_to_test_corruption(cfg, make_interactions):
    base = make_interactions(TRAIN + VAL + TEST)
    task = build_eval_task(base, cfg, "val", "strong")
    m0 = evaluate_model(GlobalPopularity().fit(task, cfg), task, cfg)

    corrupted = make_interactions(TRAIN + VAL + [
        {"visitorid": 3, "itemid": 777, "date": "2015-09-10", "event": "transaction"},
        {"visitorid": 5, "itemid": 777, "date": "2015-09-14", "event": "transaction"},
    ])
    task_c = build_eval_task(corrupted, cfg, "val", "strong")
    m1 = evaluate_model(GlobalPopularity().fit(task_c, cfg), task_c, cfg)
    assert m0["ALL"] == m1["ALL"]


def test_fit_events_are_train_only(cfg, make_interactions):
    df = make_interactions(TRAIN + VAL + TEST)
    w = windows(cfg)
    for window in ("val", "test"):
        task = build_eval_task(df, cfg, window, "strong")
        assert task.fit_events["ts"].max() < w["val"].start
        assert task.history_events["ts"].max() < w[window].start

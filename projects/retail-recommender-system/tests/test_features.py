"""Логика недели 2: граница train/target, as-of категория, счётчики событий,
целевая переменная. Проверяем, что признаки не заглядывают в будущее."""
import pandas as pd

from retail_recommender import features as F

# cutoff в конфиге = 2015-07-01. "before" — июнь, "after" — август-сентябрь.
BEFORE = [
    {"visitorid": 1, "itemid": 10, "date": "2015-06-01 10:00", "event": "view"},
    {"visitorid": 1, "itemid": 10, "date": "2015-06-02 10:00", "event": "addtocart"},
    {"visitorid": 1, "itemid": 11, "date": "2015-06-03 10:00", "event": "view"},
    {"visitorid": 2, "itemid": 10, "date": "2015-06-05 10:00", "event": "view"},
    {"visitorid": 2, "itemid": 10, "date": "2015-06-06 10:00", "event": "transaction"},
    {"visitorid": 3, "itemid": 11, "date": "2015-06-10 10:00", "event": "view"},
]
AFTER = [
    {"visitorid": 1, "itemid": 10, "date": "2015-08-01 10:00", "event": "transaction"},
    {"visitorid": 2, "itemid": 10, "date": "2015-08-02 10:00", "event": "view"},
    {"visitorid": 3, "itemid": 11, "date": "2015-08-03 10:00", "event": "addtocart"},
    {"visitorid": 4, "itemid": 12, "date": "2015-08-04 10:00", "event": "view"},
]


def _snapshots():
    rows = [
        (10, "categoryid", "2015-05-15", 100),
        (10, "categoryid", "2015-06-20", 100),
        (11, "categoryid", "2015-06-20", 200),
        # срез уже после cutoff — использоваться не должен
        (11, "categoryid", "2015-08-01", 999),
        (12, "categoryid", "2015-08-10", 300),
    ]
    df = pd.DataFrame(rows, columns=["itemid", "property", "snapshot_ts", "value_int"])
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df["value_int"] = df["value_int"].astype("Int64")
    return df


def test_cutoff_and_window_from_config(cfg):
    assert str(F.cutoff_ts(cfg).date()) == "2015-07-01"
    # правая граница exclusive — начало дня после label_window_end
    assert str(F.label_window_end_ts(cfg).date()) == "2015-09-19"


def test_pre_cutoff_drops_future_and_bots(cfg, make_interactions):
    itx = make_interactions(BEFORE + AFTER)
    itx.loc[itx["visitorid"] == 3, "is_bot"] = True

    pre = F.pre_cutoff(itx, F.cutoff_ts(cfg))
    assert pre["ts"].max() < F.cutoff_ts(cfg)
    assert 3 not in set(pre["visitorid"])          # бот отфильтрован
    assert 4 not in set(pre["visitorid"])          # событие только после cutoff


def test_counts_by_event_aggregates(cfg, make_interactions):
    before = F.pre_cutoff(make_interactions(BEFORE), F.cutoff_ts(cfg))

    by_item = F.counts_by_event(before, "itemid")
    assert by_item.loc[10].tolist() == [2, 1, 1]   # view, addtocart, transaction
    assert by_item.loc[11].tolist() == [2, 0, 0]

    by_user = F.counts_by_event(before, "visitorid")
    assert by_user.loc[1].tolist() == [2, 1, 0]

    by_pair = F.counts_by_event(before, ["visitorid", "itemid"])
    assert by_pair.loc[(2, 10)].tolist() == [1, 0, 1]


def test_category_as_of_uses_last_snapshot_before_cutoff():
    cat = F.category_as_of(_snapshots(), pd.Timestamp("2015-07-01", tz="UTC"))
    assert cat.loc[10] == 100
    assert cat.loc[11] == 200        # не 999 из среза 2015-08-01
    assert 12 not in cat.index        # первый срез товара 12 — после cutoff


def test_strong_events_after_only_strong_types(cfg, make_interactions):
    itx = make_interactions(BEFORE + AFTER)
    sp = F.strong_events_after(itx, F.cutoff_ts(cfg), F.label_window_end_ts(cfg),
                               cfg.strong_events)
    pairs = set(map(tuple, sp.to_numpy()))
    assert (1, 10) in pairs           # transaction
    assert (3, 11) in pairs           # addtocart
    assert (2, 10) not in pairs       # только view после cutoff
    assert (4, 12) not in pairs


def test_build_target_positive_only_for_historical_strong_pairs(cfg, make_interactions):
    itx = make_interactions(BEFORE + AFTER)
    cutoff = F.cutoff_ts(cfg)
    before = F.pre_cutoff(itx, cutoff)
    pairs = before[["visitorid", "itemid"]].drop_duplicates()

    strong = F.strong_events_after(itx, cutoff, F.label_window_end_ts(cfg), cfg.strong_events)
    target = F.build_target(pairs, strong).set_index(["visitorid", "itemid"])

    assert target.loc[(1, 10), "target"] == 1
    assert target.loc[(3, 11), "target"] == 1
    assert target.loc[(2, 10), "target"] == 0     # после cutoff только view
    assert target.loc[(1, 11), "target"] == 0
    # пары (4, 12) не было до cutoff — её нет в обучающей таблице
    assert (4, 12) not in target.index
    assert set(target["target"].unique()) <= {0, 1}


def test_features_do_not_use_post_cutoff_events(cfg, make_interactions):
    cutoff = F.cutoff_ts(cfg)
    base = make_interactions(BEFORE + AFTER)
    corrupt = make_interactions(BEFORE + [
        {"visitorid": 1, "itemid": 10, "date": "2015-08-01 10:00", "event": "transaction"},
        {"visitorid": 9, "itemid": 10, "date": "2015-09-01 10:00", "event": "transaction"},
    ])
    a = F.counts_by_event(F.pre_cutoff(base, cutoff), "itemid")
    b = F.counts_by_event(F.pre_cutoff(corrupt, cutoff), "itemid")
    pd.testing.assert_frame_equal(a, b)


def test_days_before_nonnegative_for_pre_cutoff(cfg, make_interactions):
    cutoff = F.cutoff_ts(cfg)
    before = F.pre_cutoff(make_interactions(BEFORE), cutoff)
    assert (F.days_before(cutoff, before["ts"]) >= 0).all()

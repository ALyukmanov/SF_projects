"""as_of lookup: никогда не берёт срез из будущего; до первого среза — UNKNOWN."""
import pandas as pd

from second_year.preprocessing.properties import as_of_lookup


def _snapshots():
    rows = [
        # itemid, property, snapshot_ts, value_int
        (1, "categoryid", "2015-05-10", 100),
        (1, "categoryid", "2015-05-24", 200),  # категория поменялась
        (2, "categoryid", "2015-05-17", 300),
    ]
    df = pd.DataFrame(rows, columns=["itemid", "property", "snapshot_ts", "value_int"])
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df["value_int"] = df["value_int"].astype("Int64")
    return df


def test_before_first_snapshot_is_unknown():
    snap = _snapshots()
    q = pd.DataFrame({"itemid": [1], "ts": pd.to_datetime(["2015-05-05"], utc=True)})
    res = as_of_lookup(snap, q, "categoryid")
    assert res.isna().all()


def test_picks_latest_snapshot_not_in_future():
    snap = _snapshots()
    q = pd.DataFrame({
        "itemid": [1, 1, 1, 2],
        "ts": pd.to_datetime(
            ["2015-05-10", "2015-05-20", "2015-06-01", "2015-05-16"], utc=True),
    })
    res = as_of_lookup(snap, q, "categoryid").tolist()
    assert res[0] == 100          # ровно на дате среза
    assert res[1] == 100          # между 10 и 24 -> ещё старая категория
    assert res[2] == 200          # после 24 -> новая
    assert pd.isna(res[3])        # для item 2 срез 17-го ещё в будущем относительно 16-го


def test_never_returns_future_snapshot_property():
    snap = _snapshots()
    q = pd.DataFrame({
        "itemid": [1, 1, 2],
        "ts": pd.to_datetime(["2015-05-09", "2015-05-23", "2015-09-01"], utc=True),
    })
    merged_ts = []
    # проверяем инвариант через публичный результат: значение соответствует
    # только «прошлым» срезам
    res = as_of_lookup(snap, q, "categoryid").tolist()
    assert pd.isna(res[0])
    assert res[1] == 100
    assert res[2] == 300


def test_result_index_matches_input():
    snap = _snapshots()
    q = pd.DataFrame(
        {"itemid": [2, 1], "ts": pd.to_datetime(["2015-06-01", "2015-06-01"], utc=True)},
        index=[7, 3],
    )
    res = as_of_lookup(snap, q, "categoryid")
    assert list(res.index) == [7, 3]
    assert res.loc[7] == 300
    assert res.loc[3] == 200

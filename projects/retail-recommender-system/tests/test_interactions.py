"""Дедуп, веса событий, пометка ботов, разметка split."""
import pandas as pd

from retail_recommender.preprocessing import interactions as itx


def _fake_events():
    rows = [
        # полный дубль (visitor 1, item 5, view, ts X) — три раза
        {"timestamp": 1_430_700_000_000, "visitorid": 1, "event": "view", "itemid": 5, "transactionid": pd.NA},
        {"timestamp": 1_430_700_000_000, "visitorid": 1, "event": "view", "itemid": 5, "transactionid": pd.NA},
        {"timestamp": 1_430_700_000_000, "visitorid": 1, "event": "view", "itemid": 5, "transactionid": pd.NA},
        {"timestamp": 1_432_000_000_000, "visitorid": 1, "event": "addtocart", "itemid": 6, "transactionid": pd.NA},
        {"timestamp": 1_440_400_000_000, "visitorid": 2, "event": "transaction", "itemid": 7, "transactionid": 111.0},
        {"timestamp": 1_441_500_000_000, "visitorid": 2, "event": "view", "itemid": 8, "transactionid": pd.NA},
    ]
    df = pd.DataFrame(rows)
    df["event"] = df["event"].astype("string")
    return df


def test_dedup_weights_bots_split(cfg, monkeypatch):
    monkeypatch.setattr(itx, "load_events", lambda _cfg: _fake_events())
    # порог ботов = 2 события, чтобы visitor 1 (3 события после дедупа: 2) не стал ботом,
    # а сделаем отдельный сценарий ниже
    df, stats = itx.build_interactions(cfg)

    assert stats.raw_rows == 6
    assert stats.full_duplicates_removed == 2
    assert stats.processed_rows == 4

    # веса
    wmap = dict(zip(df["event"], df["weight"]))
    assert wmap["view"] == cfg.event_weights["view"]
    assert wmap["addtocart"] == cfg.event_weights["addtocart"]
    assert wmap["transaction"] == cfg.event_weights["transaction"]

    # split размечен по датам конфигурации
    assert set(df["split"].dropna().unique()) <= {"train", "val", "test"}
    # ts 1_430_700_000_000 = 2015-05-03 -> train
    row = df[df["itemid"] == 5].iloc[0]
    assert row["split"] == "train"


def test_bot_flag_threshold(cfg, monkeypatch):
    many = [
        {"timestamp": 1_432_000_000_000 + i * 1000, "visitorid": 42, "event": "view",
         "itemid": i, "transactionid": pd.NA}
        for i in range(cfg.bot_event_threshold + 5)
    ]
    normal = [{"timestamp": 1_432_000_000_000, "visitorid": 7, "event": "view",
               "itemid": 1, "transactionid": pd.NA}]
    df_events = pd.DataFrame(many + normal)
    df_events["event"] = df_events["event"].astype("string")
    monkeypatch.setattr(itx, "load_events", lambda _cfg: df_events)

    df, stats = itx.build_interactions(cfg)
    assert stats.n_bots == 1
    assert df.loc[df["visitorid"] == 42, "is_bot"].all()
    assert not df.loc[df["visitorid"] == 7, "is_bot"].any()

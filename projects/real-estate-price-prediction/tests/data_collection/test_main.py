"""
Tests for src/data_collection/__main__.py -- CLI argument resolution and the
--track-history persistence wiring.

Only the pure/side-effect-isolated helpers are covered here (no network,
no argparse end-to-end invocation) -- see build_parser()'s own --help output
for the full flag surface, which is exercised manually rather than
duplicated as brittle string-matching tests here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_collection.__main__ import (
    _resolve_categories,
    _resolve_sort_orders,
)
from src.data_collection.sources.restate import RESTATE_SALE_CATEGORIES, RESTATE_SORT_ORDERS


class TestResolveCategories:
    def test_none_returns_none(self):
        assert _resolve_categories(None) is None

    def test_empty_string_returns_none(self):
        assert _resolve_categories("") is None

    def test_single_known_slug_resolves(self):
        cats = _resolve_categories("cottages_sale")
        assert len(cats) == 1
        assert cats[0].slug == "cottages_sale"
        assert cats[0].property_type == "house"

    def test_multiple_slugs_resolve_in_order(self):
        cats = _resolve_categories("room_sale,cottages_sale")
        assert [c.slug for c in cats] == ["room_sale", "cottages_sale"]

    def test_whitespace_around_slugs_is_stripped(self):
        cats = _resolve_categories(" cottages_sale , room_sale ")
        assert [c.slug for c in cats] == ["cottages_sale", "room_sale"]

    def test_unknown_slug_raises_system_exit_with_known_list(self):
        with pytest.raises(SystemExit) as exc_info:
            _resolve_categories("not_a_real_category")
        assert "not_a_real_category" in str(exc_info.value)
        assert "1_rooms_flats_sale" in str(exc_info.value)

    def test_all_default_categories_resolvable_individually(self):
        slugs = ",".join(c.slug for c in RESTATE_SALE_CATEGORIES)
        cats = _resolve_categories(slugs)
        assert len(cats) == len(RESTATE_SALE_CATEGORIES)


class TestResolveSortOrders:
    def test_none_returns_none(self):
        assert _resolve_sort_orders(None) is None

    def test_single_known_order_resolves(self):
        assert _resolve_sort_orders("price_asc") == ["price_asc"]

    def test_multiple_orders_resolve_in_order(self):
        assert _resolve_sort_orders("default,price_asc,price_desc") == [
            "default",
            "price_asc",
            "price_desc",
        ]

    def test_unknown_order_raises_system_exit_with_known_list(self):
        with pytest.raises(SystemExit) as exc_info:
            _resolve_sort_orders("not_a_real_sort")
        assert "not_a_real_sort" in str(exc_info.value)
        for known in RESTATE_SORT_ORDERS:
            assert known in str(exc_info.value)


class TestUpdateHistoryTable:
    def test_first_call_creates_current_table_and_events(self, tmp_path, monkeypatch):
        import src.data_collection.__main__ as main_mod

        monkeypatch.setattr(main_mod, "DATA_HISTORY", tmp_path)

        records = [
            {"url": "https://x/1", "price": 10_000_000},
            {"url": "https://x/2", "price": 20_000_000},
        ]
        main_mod._update_history_table("restate", "testcity", records, run_dir_name="run1")

        current_path = tmp_path / "restate" / "testcity" / "listings_current.parquet"
        events_path = tmp_path / "restate" / "testcity" / "listing_events.jsonl"
        assert current_path.exists()
        assert events_path.exists()

        current_df = pd.read_parquet(current_path)
        assert len(current_df) == 2
        assert set(current_df["status"]) == {"NEW"}

    def test_second_call_accumulates_not_overwrites(self, tmp_path, monkeypatch):
        import src.data_collection.__main__ as main_mod

        monkeypatch.setattr(main_mod, "DATA_HISTORY", tmp_path)

        main_mod._update_history_table(
            "restate",
            "testcity",
            [{"url": "https://x/1", "price": 10_000_000}],
            run_dir_name="run1",
        )
        main_mod._update_history_table(
            "restate",
            "testcity",
            [
                {"url": "https://x/1", "price": 10_500_000},
                {"url": "https://x/2", "price": 5_000_000},
            ],
            run_dir_name="run2",
        )

        current_path = tmp_path / "restate" / "testcity" / "listings_current.parquet"
        current_df = pd.read_parquet(current_path).set_index("url")
        assert len(current_df) == 2
        assert current_df.loc["https://x/1", "status"] == "UPDATED"
        assert current_df.loc["https://x/2", "status"] == "NEW"

        events_path = tmp_path / "restate" / "testcity" / "listing_events.jsonl"
        # Both runs' events should be present (append-only, not overwritten).
        lines = events_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3  # 1 NEW (run1) + 1 NEW + 1 UPDATED (run2)

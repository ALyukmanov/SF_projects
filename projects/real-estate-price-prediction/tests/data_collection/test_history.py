"""Tests for src/data_collection/history.py -- cross-run listing state tracking."""

from __future__ import annotations

from src.data_collection.history import (
    STATUS_CONFIRMED_REMOVED,
    STATUS_NEW,
    STATUS_NOT_SEEN_THIS_RUN,
    STATUS_SEEN_UNCHANGED,
    STATUS_UPDATED,
    new_current_table,
    update_history,
)


def _rec(url: str, price: float, **extra) -> dict:
    return {"url": url, "price": price, **extra}


class TestFirstRun:
    def test_all_records_are_new_on_first_run(self):
        current = new_current_table()
        run = [_rec("u1", 1_000_000), _rec("u2", 2_000_000)]
        updated, events = update_history(current, run, run_id="run1", run_at="2026-08-25T00:00:00Z")

        assert set(updated["status"]) == {STATUS_NEW}
        assert len(updated) == 2
        assert set(events["event_type"]) == {STATUS_NEW}
        assert len(events) == 2

    def test_first_seen_and_last_seen_set_to_run_at(self):
        current = new_current_table()
        run = [_rec("u1", 1_000_000)]
        updated, _ = update_history(current, run, run_id="run1", run_at="2026-08-25T00:00:00Z")
        row = updated.iloc[0]
        assert row["first_seen_at"] == "2026-08-25T00:00:00Z"
        assert row["last_seen_at"] == "2026-08-25T00:00:00Z"
        assert row["consecutive_misses"] == 0


class TestSecondRunTransitions:
    def _after_first_run(self):
        current = new_current_table()
        run1 = [_rec("u1", 1_000_000), _rec("u2", 2_000_000), _rec("u3", 3_000_000)]
        updated, _ = update_history(current, run1, run_id="run1", run_at="2026-08-25T00:00:00Z")
        return updated

    def test_unchanged_price_is_seen_unchanged(self):
        current = self._after_first_run()
        run2 = [_rec("u1", 1_000_000), _rec("u2", 2_000_000), _rec("u3", 3_000_000)]
        updated, events = update_history(
            current, run2, run_id="run2", run_at="2026-08-26T00:00:00Z"
        )

        assert set(updated["status"]) == {STATUS_SEEN_UNCHANGED}
        # No events emitted for unchanged listings -- keeps the log from growing unboundedly.
        assert events.empty

    def test_changed_price_is_updated_with_event(self):
        current = self._after_first_run()
        run2 = [_rec("u1", 1_500_000), _rec("u2", 2_000_000), _rec("u3", 3_000_000)]
        updated, events = update_history(
            current, run2, run_id="run2", run_at="2026-08-26T00:00:00Z"
        )

        u1 = updated.set_index("url").loc["u1"]
        assert u1["status"] == STATUS_UPDATED
        assert u1["previous_price"] == 1_000_000
        assert u1["price"] == 1_500_000

        assert len(events) == 1
        ev = events.iloc[0]
        assert ev["event_type"] == STATUS_UPDATED
        assert ev["old_price"] == 1_000_000
        assert ev["new_price"] == 1_500_000

    def test_first_seen_at_preserved_across_updates(self):
        current = self._after_first_run()
        run2 = [_rec("u1", 1_500_000), _rec("u2", 2_000_000), _rec("u3", 3_000_000)]
        updated, _ = update_history(current, run2, run_id="run2", run_at="2026-08-26T00:00:00Z")
        u1 = updated.set_index("url").loc["u1"]
        assert u1["first_seen_at"] == "2026-08-25T00:00:00Z"
        assert u1["last_seen_at"] == "2026-08-26T00:00:00Z"

    def test_new_listing_in_second_run_is_new(self):
        current = self._after_first_run()
        run2 = [
            _rec("u1", 1_000_000),
            _rec("u2", 2_000_000),
            _rec("u3", 3_000_000),
            _rec("u4", 4_000_000),
        ]
        updated, events = update_history(
            current, run2, run_id="run2", run_at="2026-08-26T00:00:00Z"
        )
        u4 = updated.set_index("url").loc["u4"]
        assert u4["status"] == STATUS_NEW
        assert any((events["url"] == "u4") & (events["event_type"] == STATUS_NEW))

    def test_missing_listing_is_not_seen_this_run_not_confirmed_removed(self):
        current = self._after_first_run()
        run2 = [_rec("u1", 1_000_000), _rec("u2", 2_000_000)]  # u3 dropped
        updated, events = update_history(
            current, run2, run_id="run2", run_at="2026-08-26T00:00:00Z"
        )

        u3 = updated.set_index("url").loc["u3"]
        assert (
            u3["status"] == STATUS_NOT_SEEN_THIS_RUN
        ), "A single missed run must NOT be treated as proof of removal."
        assert u3["consecutive_misses"] == 1
        ev = events[events["url"] == "u3"].iloc[0]
        assert ev["event_type"] == STATUS_NOT_SEEN_THIS_RUN


class TestConfirmedRemoved:
    def test_two_consecutive_misses_confirms_removed_with_default_threshold(self):
        current = new_current_table()
        run1 = [_rec("u1", 1_000_000)]
        current, _ = update_history(current, run1, run_id="run1", run_at="2026-08-25T00:00:00Z")

        # Run 2: u1 absent -> NOT_SEEN_THIS_RUN
        current, events2 = update_history(current, [], run_id="run2", run_at="2026-08-26T00:00:00Z")
        assert current.set_index("url").loc["u1", "status"] == STATUS_NOT_SEEN_THIS_RUN

        # Run 3: u1 still absent -> CONFIRMED_REMOVED (2 consecutive misses)
        current, events3 = update_history(current, [], run_id="run3", run_at="2026-08-27T00:00:00Z")
        row = current.set_index("url").loc["u1"]
        assert row["status"] == STATUS_CONFIRMED_REMOVED
        assert row["consecutive_misses"] == 2
        ev = events3[events3["url"] == "u1"].iloc[0]
        assert ev["event_type"] == STATUS_CONFIRMED_REMOVED

    def test_reappearing_listing_resets_miss_counter(self):
        current = new_current_table()
        run1 = [_rec("u1", 1_000_000)]
        current, _ = update_history(current, run1, run_id="run1", run_at="2026-08-25T00:00:00Z")
        current, _ = update_history(current, [], run_id="run2", run_at="2026-08-26T00:00:00Z")
        assert current.set_index("url").loc["u1", "consecutive_misses"] == 1

        # u1 reappears before hitting the confirm-removed threshold.
        current, events = update_history(
            current, [_rec("u1", 1_000_000)], run_id="run3", run_at="2026-08-27T00:00:00Z"
        )
        row = current.set_index("url").loc["u1"]
        assert row["status"] == STATUS_SEEN_UNCHANGED
        assert row["consecutive_misses"] == 0

    def test_confirmed_removed_does_not_re_emit_event_every_subsequent_run(self):
        current = new_current_table()
        current, _ = update_history(current, [_rec("u1", 1_000_000)], run_id="r1", run_at="t1")
        current, _ = update_history(current, [], run_id="r2", run_at="t2")
        current, events3 = update_history(current, [], run_id="r3", run_at="t3")
        assert len(events3) == 1  # the CONFIRMED_REMOVED transition itself

        current, events4 = update_history(current, [], run_id="r4", run_at="t4")
        assert events4.empty, "Should not re-flag an already-confirmed-removed listing every run."
        assert current.set_index("url").loc["u1", "status"] == STATUS_CONFIRMED_REMOVED

    def test_custom_threshold_respected(self):
        current = new_current_table()
        current, _ = update_history(current, [_rec("u1", 1_000_000)], run_id="r1", run_at="t1")
        current, events = update_history(
            current, [], run_id="r2", run_at="t2", confirm_removed_after_n_misses=1
        )
        assert current.set_index("url").loc["u1", "status"] == STATUS_CONFIRMED_REMOVED
        assert events.iloc[0]["event_type"] == STATUS_CONFIRMED_REMOVED


class TestNoAutoDeleteOrOverwrite:
    def test_non_tracking_fields_are_preserved_and_updated(self):
        current = new_current_table()
        run1 = [_rec("u1", 1_000_000, city="Москва", rooms=2)]
        current, _ = update_history(current, run1, run_id="r1", run_at="t1")

        run2 = [_rec("u1", 1_500_000, city="Москва", rooms=2)]
        current, _ = update_history(current, run2, run_id="r2", run_at="t2")
        row = current.set_index("url").loc["u1"]
        assert row["city"] == "Москва"
        assert row["rooms"] == 2
        assert row["price"] == 1_500_000

    def test_empty_run_records_does_not_error_on_empty_current(self):
        current = new_current_table()
        updated, events = update_history(current, [], run_id="r1", run_at="t1")
        assert updated.empty
        assert events.empty

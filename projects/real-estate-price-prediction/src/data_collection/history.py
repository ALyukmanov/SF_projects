"""
Cross-run listing history: NEW / SEEN_UNCHANGED / UPDATED / NOT_SEEN_THIS_RUN /
CONFIRMED_REMOVED state tracking.

Why this exists
----------------
The CLI's ``--incremental`` flag (see ``__main__.py``) compares exactly two
run directories -- "this run" vs. "the most recent previous run" -- and
computes a one-off NEW/UPDATED/UNCHANGED/MISSING diff. That is useful for a
quick two-run comparison, but it does not accumulate: a listing's real
history (when it first appeared, how many times its price changed, whether
it has been consistently absent across several runs rather than just one)
is lost as soon as a third run happens, because nothing carries state
forward between runs.

This module adds that carry-forward layer on top of normalized listing
dicts (i.e. call this AFTER ``normalize.py``, not on raw JSON-LD). It is
intentionally NOT a database: the "current" table is a plain DataFrame
(persisted as Parquet/CSV by the caller) and the "events" table is a plain
append-only list of dicts (persisted as JSONL by the caller) -- matching the
project's existing "don't over-engineer with a full DB" persistence style
(see persistence.py's JsonlCheckpoint).

Explicit "no fake completeness" rule this module exists to satisfy: a
listing missing from a SINGLE run is not proof it was removed from the
market (the run could have been partial, rate-limited, or simply not swept
that category/sort combination this time) -- see ``NOT_SEEN_THIS_RUN`` vs.
``CONFIRMED_REMOVED`` below.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------
STATUS_NEW = "NEW"
STATUS_SEEN_UNCHANGED = "SEEN_UNCHANGED"
STATUS_UPDATED = "UPDATED"
STATUS_NOT_SEEN_THIS_RUN = "NOT_SEEN_THIS_RUN"
STATUS_CONFIRMED_REMOVED = "CONFIRMED_REMOVED"

ALL_STATUSES = (
    STATUS_NEW,
    STATUS_SEEN_UNCHANGED,
    STATUS_UPDATED,
    STATUS_NOT_SEEN_THIS_RUN,
    STATUS_CONFIRMED_REMOVED,
)

DEFAULT_CONFIRM_REMOVED_AFTER_N_MISSES = 2

_TRACKING_COLUMNS = (
    "status",
    "first_seen_at",
    "last_seen_at",
    "last_price_change_at",
    "consecutive_misses",
    "previous_price",
)


def new_current_table() -> pd.DataFrame:
    """Return an empty ``listings_current`` table with the right columns,
    for the very first run of a source/city (no prior history to load)."""
    return pd.DataFrame(columns=["url", *_TRACKING_COLUMNS])


def update_history(
    current_df: pd.DataFrame,
    run_records: List[Dict[str, Any]],
    run_id: str,
    run_at: str,
    confirm_removed_after_n_misses: int = DEFAULT_CONFIRM_REMOVED_AFTER_N_MISSES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fold one collection run's normalized records into the running
    ``listings_current`` state, returning the updated current table and this
    run's events.

    Args:
        current_df: The ``listings_current`` table from the END of the
            previous run (use :func:`new_current_table` for the first run).
            Must have a ``url`` column plus whatever normalized listing
            fields were tracked previously; any additional columns from
            ``run_records`` are preserved going forward.
        run_records: Normalized listing dicts for the CURRENT run (i.e.
            ``normalize_restate_records()`` output) -- must each have a
            ``url`` and (usually) a ``price`` key.
        run_id: Identifier for this run (e.g. the run directory name), used
            only to tag events, not for any dedup logic.
        run_at: ISO timestamp for this run, used to stamp
            ``first_seen_at``/``last_seen_at``/event timestamps.
        confirm_removed_after_n_misses: A listing must be absent for this
            many CONSECUTIVE runs before its status flips to
            ``CONFIRMED_REMOVED``. Before that threshold it is
            ``NOT_SEEN_THIS_RUN`` -- a single missed run is not treated as
            proof of removal (the run could have been partial/rate-limited/
              not swept that category this time), only a sustained absence is.

    Returns:
        ``(updated_current_df, events_df)`` -- ``events_df`` has one row per
        NEW/UPDATED/NOT_SEEN_THIS_RUN/CONFIRMED_REMOVED event detected THIS
        run (no row for SEEN_UNCHANGED listings, to keep the events log from
        growing unboundedly with "nothing happened" entries).
    """
    current_by_url: Dict[str, Dict[str, Any]] = (
        {row["url"]: row for row in current_df.to_dict("records")} if not current_df.empty else {}
    )
    run_by_url: Dict[str, Dict[str, Any]] = {r["url"]: r for r in run_records if r.get("url")}

    events: List[Dict[str, Any]] = []
    updated_rows: Dict[str, Dict[str, Any]] = {}

    for url, rec in run_by_url.items():
        prev = current_by_url.get(url)
        new_price = rec.get("price")

        if prev is None:
            updated_rows[url] = {
                **rec,
                "status": STATUS_NEW,
                "first_seen_at": run_at,
                "last_seen_at": run_at,
                "last_price_change_at": run_at,
                "consecutive_misses": 0,
                "previous_price": None,
            }
            events.append(
                {
                    "run_id": run_id,
                    "run_at": run_at,
                    "url": url,
                    "event_type": STATUS_NEW,
                    "old_price": None,
                    "new_price": new_price,
                }
            )
            continue

        prev_price = prev.get("price")
        price_changed = pd.notna(prev_price) and pd.notna(new_price) and prev_price != new_price
        status = STATUS_UPDATED if price_changed else STATUS_SEEN_UNCHANGED
        updated_rows[url] = {
            **rec,
            "status": status,
            "first_seen_at": prev.get("first_seen_at", run_at),
            "last_seen_at": run_at,
            "last_price_change_at": run_at
            if price_changed
            else prev.get("last_price_change_at", run_at),
            "consecutive_misses": 0,
            "previous_price": prev_price if price_changed else prev.get("previous_price"),
        }
        if price_changed:
            events.append(
                {
                    "run_id": run_id,
                    "run_at": run_at,
                    "url": url,
                    "event_type": STATUS_UPDATED,
                    "old_price": prev_price,
                    "new_price": new_price,
                }
            )

    # Listings that were tracked before but did not appear in this run.
    for url, prev in current_by_url.items():
        if url in run_by_url:
            continue
        if prev.get("status") == STATUS_CONFIRMED_REMOVED:
            # Already confirmed removed in an earlier run -- carry forward
            # as-is, don't re-emit a CONFIRMED_REMOVED event every run.
            updated_rows[url] = prev
            continue

        misses = int(prev.get("consecutive_misses") or 0) + 1
        if misses >= confirm_removed_after_n_misses:
            status = STATUS_CONFIRMED_REMOVED
            event_type = STATUS_CONFIRMED_REMOVED
        else:
            status = STATUS_NOT_SEEN_THIS_RUN
            event_type = STATUS_NOT_SEEN_THIS_RUN

        updated_rows[url] = {**prev, "status": status, "consecutive_misses": misses}
        events.append(
            {
                "run_id": run_id,
                "run_at": run_at,
                "url": url,
                "event_type": event_type,
                "old_price": prev.get("price"),
                "new_price": None,
            }
        )

    updated_df = pd.DataFrame(list(updated_rows.values()))
    events_df = pd.DataFrame(events)

    logger.info(
        "History update (run_id=%s): %d new, %d updated, %d unchanged, %d not-seen, %d confirmed-removed "
        "(current table now has %d row(s)).",
        run_id,
        sum(1 for r in updated_rows.values() if r["status"] == STATUS_NEW),
        sum(1 for r in updated_rows.values() if r["status"] == STATUS_UPDATED),
        sum(1 for r in updated_rows.values() if r["status"] == STATUS_SEEN_UNCHANGED),
        sum(1 for r in updated_rows.values() if r["status"] == STATUS_NOT_SEEN_THIS_RUN),
        sum(1 for r in updated_rows.values() if r["status"] == STATUS_CONFIRMED_REMOVED),
        len(updated_df),
    )
    return updated_df, events_df

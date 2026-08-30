"""
Persistence / checkpointing component — isolated from parsing and network.

Addresses a gap explicitly disclosed (not silently fixed-and-forgotten) in
``src/data_collection/README.md`` "Known gap — checkpointing/resume": prior
to this module, neither scraper persisted partial progress mid-run, so a
crash/interrupt lost the whole in-progress batch. This is a small,
dependency-free JSONL-based checkpoint writer/reader:

- Append-only JSONL (one listing dict per line) so a crash mid-write loses
  at most the last partial line, not the whole file.
- Resume support: read back the set of already-checkpointed URLs so a
  restarted collection run can skip pages/listings it already has, instead
  of re-fetching (and re-risking a rate-limit block on) the same URLs.

Deliberately NOT a database/queue — for the data volumes this project
actually deals with (a handful of pages per run, per the "no mass scraping"
policy — see ``src/data_collection/README.md``), a flat JSONL file is
simpler, has zero new dependencies, and is trivially inspectable by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Set

from src.utils.logger import get_logger

logger = get_logger(__name__)


class JsonlCheckpoint:
    """Append-only JSONL checkpoint file with URL-based resume support.

    Args:
        path: Path to the checkpoint file (created on first write if it
            doesn't exist; existing content is preserved and appended to,
            never truncated, so re-running with the same path resumes
            rather than silently discarding prior progress).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_seen_urls(self) -> Set[str]:
        """Return the set of ``url`` values already present in the checkpoint.

        Malformed lines (partial writes from a crash mid-append, for
        instance) are skipped with a warning rather than raising — a
        checkpoint file existing for resume purposes should degrade
        gracefully, not block a restart entirely.
        """
        seen: Set[str] = set()
        if not self.path.exists():
            return seen
        with self.path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed checkpoint line %d in %s (partial write?)",
                        line_num,
                        self.path,
                    )
                    continue
                url = row.get("url") or row.get("source_url")
                if url:
                    seen.add(url)
        return seen

    def append(self, listings: Iterable[Dict[str, Any]]) -> int:
        """Append *listings* to the checkpoint file, one JSON object per line.

        Returns:
            Number of rows written.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with self.path.open("a", encoding="utf-8") as f:
            for listing in listings:
                f.write(json.dumps(listing, ensure_ascii=False, default=str))
                f.write("\n")
                count += 1
        if count:
            logger.info("Checkpointed %d row(s) -> %s", count, self.path)
        return count

    def read_all(self) -> Iterator[Dict[str, Any]]:
        """Yield every valid row in the checkpoint file, skipping malformed lines."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

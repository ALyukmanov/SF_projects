"""Tests for src/data_collection/persistence.py — JSONL checkpoint/resume."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_collection.persistence import JsonlCheckpoint  # noqa: E402


class TestJsonlCheckpoint:
    def test_load_seen_urls_on_missing_file_returns_empty_set(self, tmp_path):
        cp = JsonlCheckpoint(tmp_path / "does_not_exist.jsonl")
        assert cp.load_seen_urls() == set()

    def test_append_then_load_seen_urls_roundtrip(self, tmp_path):
        cp = JsonlCheckpoint(tmp_path / "checkpoint.jsonl")
        rows = [
            {"url": "https://example.com/1/", "price": 1_000_000},
            {"url": "https://example.com/2/", "price": 2_000_000},
        ]
        written = cp.append(rows)
        assert written == 2

        seen = cp.load_seen_urls()
        assert seen == {"https://example.com/1/", "https://example.com/2/"}

    def test_append_is_additive_across_multiple_calls_resume_semantics(self, tmp_path):
        path = tmp_path / "checkpoint.jsonl"
        cp1 = JsonlCheckpoint(path)
        cp1.append([{"url": "https://example.com/1/"}])

        # Simulate a fresh process restarting and resuming from the same file.
        cp2 = JsonlCheckpoint(path)
        assert cp2.load_seen_urls() == {"https://example.com/1/"}
        cp2.append([{"url": "https://example.com/2/"}])

        cp3 = JsonlCheckpoint(path)
        assert cp3.load_seen_urls() == {
            "https://example.com/1/",
            "https://example.com/2/",
        }, "Second append must not have truncated the first run's data."

    def test_falls_back_to_source_url_key(self, tmp_path):
        cp = JsonlCheckpoint(tmp_path / "checkpoint.jsonl")
        cp.append([{"source_url": "https://example.com/only-source-url/"}])
        assert cp.load_seen_urls() == {"https://example.com/only-source-url/"}

    def test_malformed_line_is_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "checkpoint.jsonl"
        path.write_text(
            '{"url": "https://example.com/good/"}\n'
            "not valid json at all\n"
            '{"url": "https://example.com/also-good/"}\n',
            encoding="utf-8",
        )
        cp = JsonlCheckpoint(path)
        seen = cp.load_seen_urls()
        assert seen == {"https://example.com/good/", "https://example.com/also-good/"}

    def test_read_all_yields_full_rows_not_just_urls(self, tmp_path):
        cp = JsonlCheckpoint(tmp_path / "checkpoint.jsonl")
        cp.append([{"url": "https://example.com/1/", "price": 5_000_000, "rooms": 2}])
        rows = list(cp.read_all())
        assert rows == [{"url": "https://example.com/1/", "price": 5_000_000, "rooms": 2}]

    def test_non_ascii_content_roundtrips(self, tmp_path):
        cp = JsonlCheckpoint(tmp_path / "checkpoint.jsonl")
        cp.append([{"url": "https://example.com/1/", "city": "москва", "address": "ул. Ленина"}])
        rows = list(cp.read_all())
        assert rows[0]["city"] == "москва"
        assert rows[0]["address"] == "ул. Ленина"

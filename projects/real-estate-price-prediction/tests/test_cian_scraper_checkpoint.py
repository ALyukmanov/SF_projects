"""
Tests for CianScraper's optional checkpoint/resume integration
(src/data_collection/persistence.py wired into scrape_pages()).

No network access — `_get_with_retry` is monkeypatched exactly like the
existing offline parsing tests in tests/test_cian_scraper_parsing.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_collection.cian_scraper import CianScraper  # noqa: E402
from src.data_collection.persistence import JsonlCheckpoint  # noqa: E402

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "cian_html"


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code


def _load(name: str) -> str:
    return (_FIXTURES_DIR / name).read_text(encoding="utf-8")


class TestCheckpointDisabledByDefault:
    def test_no_checkpoint_path_means_no_file_written(self, tmp_path, monkeypatch):
        scraper = CianScraper(city="moskva", checkpoint_path=None)
        assert scraper._checkpoint is None
        html = _load("normal_card.html")
        monkeypatch.setattr(scraper, "_get_with_retry", lambda url: _FakeResponse(html))
        scraper.scrape_pages(base_url="https://example.com/cat.php?p=1", max_pages=1)
        assert not (tmp_path / "checkpoint.jsonl").exists()


class TestCheckpointWiring:
    def test_scrape_pages_writes_checkpoint(self, tmp_path, monkeypatch):
        checkpoint_path = tmp_path / "checkpoint.jsonl"
        scraper = CianScraper(city="moskva", checkpoint_path=str(checkpoint_path))
        html = _load("normal_card.html")
        monkeypatch.setattr(scraper, "_get_with_retry", lambda url: _FakeResponse(html))

        listings = scraper.scrape_pages(base_url="https://example.com/cat.php?p=1", max_pages=1)

        assert len(listings) == 1
        assert checkpoint_path.exists()
        rows = list(JsonlCheckpoint(checkpoint_path).read_all())
        assert len(rows) == 1
        assert rows[0]["url"] == listings[0]["url"]

    def test_resume_skips_already_checkpointed_urls(self, tmp_path, monkeypatch):
        checkpoint_path = tmp_path / "checkpoint.jsonl"
        html = _load("normal_card.html")

        # First "run": collect and checkpoint one listing.
        scraper1 = CianScraper(city="moskva", checkpoint_path=str(checkpoint_path))
        monkeypatch.setattr(scraper1, "_get_with_retry", lambda url: _FakeResponse(html))
        first_listings = scraper1.scrape_pages(
            base_url="https://example.com/cat.php?p=1", max_pages=1
        )
        assert len(first_listings) == 1
        collected_url = first_listings[0]["url"]

        # Second "run" (simulating a restart): same checkpoint path, same
        # fixture (same URL would be produced again) — resume must dedupe it.
        scraper2 = CianScraper(city="moskva", checkpoint_path=str(checkpoint_path))
        assert (
            collected_url in scraper2._seen_urls
        ), "Resuming from an existing checkpoint must pre-populate seen URLs."
        monkeypatch.setattr(scraper2, "_get_with_retry", lambda url: _FakeResponse(html))
        second_listings = scraper2.scrape_pages(
            base_url="https://example.com/cat.php?p=1", max_pages=1
        )
        assert (
            second_listings == []
        ), "The same listing URL should be deduplicated against the resumed checkpoint."

    @pytest.mark.parametrize("page_count", [1, 2])
    def test_checkpoint_row_count_matches_pages_with_listings(
        self, tmp_path, monkeypatch, page_count
    ):
        checkpoint_path = tmp_path / "checkpoint.jsonl"
        html = _load("normal_card.html")
        scraper = CianScraper(city="moskva", checkpoint_path=str(checkpoint_path))
        # Each fake page returns the same single-listing fixture; distinct
        # page URLs still produce the same listing URL from this fixture, so
        # only the first page's listing survives dedup — this test asserts
        # that behavior explicitly rather than assuming it.
        monkeypatch.setattr(scraper, "_get_with_retry", lambda url: _FakeResponse(html))
        scraper.scrape_pages(base_url="https://example.com/cat.php?p=1", max_pages=page_count)
        rows = list(JsonlCheckpoint(checkpoint_path).read_all())
        assert len(rows) == 1, "Fixture always yields the same URL, so dedup caps this at 1 row."

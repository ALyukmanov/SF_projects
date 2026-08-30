"""Offline parsing tests for CianScraper (requests+BeautifulSoup variant).

These exercise the three parsing strategies (JSON-LD, embedded JS JSON,
<article>-tag fallback) directly against de-identified local HTML fixtures
in tests/data_collection/fixtures/ — no network access, no real CIAN pages.
Fixture markup is hand-written to exercise specific edge cases; it does not
reproduce any real listing content.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_collection.cian_scraper import CianScraper  # noqa: E402
from src.data_collection.parsing_utils import PARSER_VERSION  # noqa: E402

_FIXTURES = Path(__file__).parent / "fixtures"


def _soup(name: str) -> BeautifulSoup:
    html = (_FIXTURES / name).read_text(encoding="utf-8")
    return BeautifulSoup(html, "html.parser")


@pytest.fixture
def scraper() -> CianScraper:
    return CianScraper(city="moskva")


class TestJsonLdStrategy:
    def test_normal_card_json_ld(self, scraper):
        listings = scraper._parse_json_ld(_soup("json_ld_normal.html"), "https://cian.ru/fixture")
        assert len(listings) == 2
        first = listings[0]
        assert first["url"] == "https://cian.ru/sale/flat/900001/"
        assert first["price"] == 12_500_000.0
        assert first["rooms"] == 2

    def test_studio_detected(self, scraper):
        listings = scraper._parse_json_ld(_soup("json_ld_normal.html"), "https://cian.ru/fixture")
        studio = [x for x in listings if x["url"].endswith("900002/")][0]
        assert studio["rooms"] == 0

    def test_missing_price_field_does_not_raise(self, scraper):
        listings = scraper._parse_json_ld(
            _soup("json_ld_missing_price.html"), "https://cian.ru/fixture"
        )
        assert len(listings) == 1
        assert listings[0]["price"] is None

    def test_provenance_fields_present(self, scraper):
        listings = scraper._parse_json_ld(_soup("json_ld_normal.html"), "https://cian.ru/fixture")
        for listing in listings:
            assert listing["parser_version"] == PARSER_VERSION
            assert listing["source_city"] == "moskva"
            assert listing["source_url"] == listing["url"]


class TestEmbeddedJsonStrategy:
    def test_normal_embedded_json(self, scraper):
        listings = scraper._parse_embedded_json(
            _soup("embedded_json_normal.html"), "https://cian.ru/fixture"
        )
        assert len(listings) == 1
        listing = listings[0]
        assert listing["url"] == "https://cian.ru/sale/flat/900010/"
        assert listing["price"] == 9_800_000.0
        assert listing["rooms"] == 1
        assert listing["total_area"] == 33.5
        assert listing["floor"] == 4
        assert listing["floors_total"] == 9


class TestArticleTagFallback:
    def test_used_only_when_others_fail(self, scraper):
        """A page with neither JSON-LD nor embedded JSON should still yield
        listings via the <article>/CardComponent fallback."""
        soup = _soup("article_tags_fallback.html")
        assert scraper._parse_json_ld(soup, "u") == []
        assert scraper._parse_embedded_json(soup, "u") == []
        listings = scraper._parse_article_tags(soup, "https://cian.ru/fixture")
        assert len(listings) == 2

    def test_price_with_spaces_and_currency_symbol(self, scraper):
        soup = _soup("article_tags_fallback.html")
        listings = scraper._parse_article_tags(soup, "u")
        prices = {x["price"] for x in listings}
        assert 13_200_000.0 in prices
        assert 5_100_000.0 in prices

    def test_studio_and_floor_parsed(self, scraper):
        soup = _soup("article_tags_fallback.html")
        listings = scraper._parse_article_tags(soup, "u")
        studio = [x for x in listings if x["rooms"] == 0][0]
        assert studio["floor"] == 1
        assert studio["floors_total"] == 5


class TestRobustness:
    def test_empty_page_returns_empty_list_all_strategies(self, scraper):
        soup = _soup("empty_page.html")
        assert scraper._parse_json_ld(soup, "u") == []
        assert scraper._parse_embedded_json(soup, "u") == []
        assert scraper._parse_article_tags(soup, "u") == []

    def test_malformed_html_does_not_raise(self, scraper):
        """BeautifulSoup itself tolerates broken markup; the malformed
        JSON-LD block inside must be skipped (json.JSONDecodeError caught),
        not propagate as an unhandled exception."""
        soup = _soup("malformed.html")
        listings = scraper._parse_json_ld(soup, "u")
        assert listings == []
        # article-tag fallback should not raise even on truncated tags
        scraper._parse_article_tags(soup, "u")


class TestDeduplication:
    def test_duplicate_urls_removed(self, scraper):
        listings = [
            {"url": "https://cian.ru/sale/flat/1/", "price": 1.0},
            {"url": "https://cian.ru/sale/flat/1/", "price": 1.0},
            {"url": "https://cian.ru/sale/flat/2/", "price": 2.0},
        ]
        deduped = scraper._deduplicate(listings)
        assert len(deduped) == 2
        assert {x["url"] for x in deduped} == {
            "https://cian.ru/sale/flat/1/",
            "https://cian.ru/sale/flat/2/",
        }

    def test_dedup_persists_across_calls_within_a_session(self, scraper):
        """_seen_urls accumulates within a single scraper instance so the
        same listing seen on two different result pages is only kept once."""
        first = scraper._deduplicate([{"url": "https://cian.ru/sale/flat/1/"}])
        second = scraper._deduplicate([{"url": "https://cian.ru/sale/flat/1/"}])
        assert len(first) == 1
        assert len(second) == 0

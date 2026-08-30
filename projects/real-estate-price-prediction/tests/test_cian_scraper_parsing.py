"""
Pure-parsing tests for ``src/data_collection/cian_scraper.py`` (requests +
BeautifulSoup).

Scope and seam used
--------------------
``CianScraper`` already separates network I/O (``_get_with_retry``) from
HTML/JSON parsing (``_parse_json_ld``, ``_parse_embedded_json``,
``_parse_article_tags``/``_parse_single_card``, and the module-level regex
helpers ``_parse_price``/``_parse_rooms``/``_parse_area``/``_parse_floor_info``).
No refactor was needed to make parsing testable: these tests call the
parsing methods directly on ``BeautifulSoup`` objects built from hand-written
fixture files under ``tests/fixtures/cian_html/`` (no ``requests.get`` is
ever invoked). A handful of tests also exercise ``scrape_page()`` end-to-end
with ``_get_with_retry`` monkeypatched to return an in-memory fake response,
to confirm the strategy cascade (JSON-LD -> embedded JS JSON -> <article>
fallback) and error handling behave correctly without touching the network.

Constructing ``CianScraper()`` itself does not touch the network either:
``_build_session()`` calls ``_random_ua()``, which tries to import
``fake_useragent`` and falls back to a hardcoded User-Agent list via a caught
``ImportError`` — confirmed safe in this environment where ``fake_useragent``
is not installed.

All fixture HTML uses ``example.com``-style fake URLs/addresses — nothing
was scraped live from cian.ru.

See ``tests/test_cian_selenium_scraper_parsing.py`` for the Selenium-based
scraper's parsing helpers; those tests are skipped in this environment
because the ``selenium``/``webdriver-manager`` packages (declared in
requirements.txt) are not installed in the ambient interpreter and could not
be installed here (see that file's docstring for the exact blocker).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_collection.cian_scraper import (  # noqa: E402
    CianScraper,
    _parse_area,
    _parse_floor_info,
    _parse_price,
    _parse_rooms,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "cian_html"


def _load(name: str) -> str:
    return (_FIXTURES_DIR / name).read_text(encoding="utf-8")


def _soup(name: str) -> BeautifulSoup:
    return BeautifulSoup(_load(name), "html.parser")


@pytest.fixture
def scraper() -> CianScraper:
    """A ``CianScraper`` built entirely offline (see module docstring)."""
    return CianScraper(city="moskva", deal_type="sale")


# ===========================================================================
# Module-level regex helpers
# ===========================================================================


class TestParsePrice:
    def test_plain_digits(self):
        assert _parse_price("8500000") == 8500000.0

    def test_spaces_and_ruble_symbol(self):
        assert _parse_price("12 500 000 ₽") == 12500000.0

    def test_nbsp_and_symbol(self):
        assert _parse_price("12 500 000 ₽") == 12500000.0

    def test_rub_word_suffix(self):
        assert _parse_price("9 300 000 руб.") == 9300000.0

    def test_empty_string_returns_none(self):
        assert _parse_price("") is None

    def test_none_input_returns_none(self):
        assert _parse_price(None) is None

    def test_no_digits_returns_none(self):
        assert _parse_price("Цена по запросу") is None


class TestParseRooms:
    def test_studio_russian(self):
        assert _parse_rooms("Студия, 24 м²") == 0

    def test_studio_english(self):
        assert _parse_rooms("Studio apartment") == 0

    def test_standard_rooms(self):
        assert _parse_rooms("2-комн. квартира") == 2

    def test_short_form(self):
        assert _parse_rooms("3-к квартира") == 3

    def test_empty_string_returns_none(self):
        assert _parse_rooms("") is None

    def test_unparseable_returns_none(self):
        assert _parse_rooms("хорошая квартира") is None


class TestParseArea:
    def test_comma_decimal(self):
        assert _parse_area("45,5 м²") == 45.5

    def test_dot_decimal_kvm(self):
        assert _parse_area("45.5 кв.м") == 45.5

    def test_empty_returns_none(self):
        assert _parse_area("") is None

    def test_no_match_returns_none(self):
        assert _parse_area("нет данных") is None


class TestParseFloorInfo:
    def test_floor_and_total(self):
        assert _parse_floor_info("5/12 эт.") == (5, 12)

    def test_floor_only(self):
        assert _parse_floor_info("3 эт.") == (3, None)

    def test_empty_returns_none_none(self):
        assert _parse_floor_info("") == (None, None)

    def test_no_match_returns_none_none(self):
        assert _parse_floor_info("не указано") == (None, None)


# ===========================================================================
# Strategy 3 — <article> / card HTML fallback parsing (soup-level, no network)
# ===========================================================================


class TestArticleTagParsing:
    def test_normal_card_parses_all_fields(self, scraper: CianScraper):
        soup = _soup("normal_card.html")
        listings = scraper._parse_article_tags(soup, "https://example.com/cat.php")
        assert len(listings) == 1
        listing = listings[0]
        assert listing["url"] == "https://example.com/sale/flat/900001/"
        assert listing["price"] == 12500000.0
        assert listing["rooms"] == 2
        assert listing["total_area"] == 54.3
        assert listing["floor"] == 5
        assert listing["floors_total"] == 12
        assert "Примерная" in listing["address"]

    def test_missing_floor_does_not_crash(self, scraper: CianScraper):
        """A card with no floor info must yield floor=None, not raise."""
        soup = _soup("missing_floor.html")
        listings = scraper._parse_article_tags(soup, "https://example.com/cat.php")
        assert len(listings) == 1
        assert listings[0]["floor"] is None
        assert listings[0]["floors_total"] is None
        assert listings[0]["price"] == 7200000.0
        assert listings[0]["rooms"] == 3

    def test_studio_apartment_rooms_zero(self, scraper: CianScraper):
        soup = _soup("studio.html")
        listings = scraper._parse_article_tags(soup, "https://example.com/cat.php")
        assert len(listings) == 1
        assert listings[0]["rooms"] == 0
        assert listings[0]["total_area"] == 24.5

    def test_price_with_nbsp_and_symbol_parses_to_plain_number(self, scraper: CianScraper):
        soup = _soup("price_format_nbsp_symbol.html")
        listings = scraper._parse_article_tags(soup, "https://example.com/cat.php")
        assert len(listings) == 1
        assert listings[0]["price"] == 12500000.0
        assert isinstance(listings[0]["price"], float)

    def test_changed_css_classes_falls_back_to_generic_selector(self, scraper: CianScraper):
        """No <article>/CardComponent markup — must still recover via the
        third-level `[class*='card']` fallback selector rather than
        returning nothing."""
        soup = _soup("changed_css_classes.html")
        listings = scraper._parse_article_tags(soup, "https://example.com/cat.php")
        assert len(listings) == 1
        assert listings[0]["url"] == "https://example.com/sale/flat/900020/"
        assert listings[0]["price"] == 6100000.0
        assert listings[0]["rooms"] == 1

    def test_unrecognized_markup_returns_empty_not_crash(self, scraper: CianScraper):
        """No selector in any strategy matches — must degrade to an empty
        list, not raise."""
        soup = _soup("unrecognized_markup.html")
        listings = scraper._parse_article_tags(soup, "https://example.com/cat.php")
        assert listings == []

    def test_empty_results_page_returns_empty_list(self, scraper: CianScraper):
        soup = _soup("empty_results.html")
        listings = scraper._parse_article_tags(soup, "https://example.com/cat.php")
        assert listings == []

    def test_malformed_html_does_not_raise(self, scraper: CianScraper):
        """BeautifulSoup construction and parsing must not raise on
        unclosed/malformed markup."""
        soup = _soup("malformed.html")  # construction itself must not raise
        listings = scraper._parse_article_tags(soup, "https://example.com/cat.php")
        assert isinstance(listings, list)


# ===========================================================================
# Strategy 1 — JSON-LD
# ===========================================================================


class TestJsonLdParsing:
    def test_json_ld_offer_parses(self, scraper: CianScraper):
        soup = _soup("json_ld_listing.html")
        listings = scraper._parse_json_ld(soup, "https://example.com/cat.php")
        assert len(listings) == 1
        listing = listings[0]
        assert listing["url"] == "https://example.com/sale/flat/900010/"
        assert listing["price"] == 8500000.0
        assert listing["rooms"] == 2


# ===========================================================================
# Strategy 2 — embedded JS JSON (window.__initialData__)
# ===========================================================================


class TestEmbeddedJsonParsing:
    def test_embedded_initial_data_parses(self, scraper: CianScraper):
        soup = _soup("embedded_json_listing.html")
        listings = scraper._parse_embedded_json(soup, "https://example.com/cat.php")
        assert len(listings) == 1
        listing = listings[0]
        assert listing["price"] == 9500000.0
        assert listing["rooms"] == 3
        assert listing["total_area"] == 75.5
        assert listing["floor"] == 4
        assert listing["floors_total"] == 9
        assert listing["url"] == "https://www.cian.ru/sale/flat/900011/"


# ===========================================================================
# Deduplication
# ===========================================================================


class TestDeduplicate:
    def test_duplicate_urls_are_removed(self, scraper: CianScraper):
        listings = [
            {"url": "https://example.com/1/", "price": 1},
            {"url": "https://example.com/2/", "price": 2},
            {"url": "https://example.com/1/", "price": 1},  # duplicate
        ]
        unique = scraper._deduplicate(listings)
        assert len(unique) == 2
        assert {item["url"] for item in unique} == {
            "https://example.com/1/",
            "https://example.com/2/",
        }

    def test_listings_without_url_are_dropped(self, scraper: CianScraper):
        listings = [{"url": "", "price": 1}, {"price": 2}]
        unique = scraper._deduplicate(listings)
        assert unique == []


# ===========================================================================
# End-to-end scrape_page() with the network layer monkeypatched (no real HTTP)
# ===========================================================================


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code


class TestScrapePageNoNetwork:
    def test_scrape_page_uses_article_fallback(self, scraper: CianScraper, monkeypatch):
        html = _load("normal_card.html")
        monkeypatch.setattr(scraper, "_get_with_retry", lambda url: _FakeResponse(html))
        listings = scraper.scrape_page("https://example.com/cat.php?p=1")
        assert len(listings) == 1
        assert listings[0]["price"] == 12500000.0

    def test_scrape_page_empty_results_returns_empty_list(self, scraper: CianScraper, monkeypatch):
        html = _load("empty_results.html")
        monkeypatch.setattr(scraper, "_get_with_retry", lambda url: _FakeResponse(html))
        listings = scraper.scrape_page("https://example.com/cat.php?p=1")
        assert listings == []

    def test_scrape_page_prefers_json_ld_over_article(self, scraper: CianScraper, monkeypatch):
        html = _load("json_ld_listing.html")
        monkeypatch.setattr(scraper, "_get_with_retry", lambda url: _FakeResponse(html))
        listings = scraper.scrape_page("https://example.com/cat.php?p=1")
        assert len(listings) == 1
        assert listings[0]["price"] == 8500000.0

    def test_scrape_page_returns_empty_list_when_fetch_fails(
        self, scraper: CianScraper, monkeypatch
    ):
        """_get_with_retry returning None (exhausted retries) must not raise."""
        monkeypatch.setattr(scraper, "_get_with_retry", lambda url: None)
        listings = scraper.scrape_page("https://example.com/cat.php?p=1")
        assert listings == []

    def test_scrape_page_malformed_html_does_not_raise(self, scraper: CianScraper, monkeypatch):
        html = _load("malformed.html")
        monkeypatch.setattr(scraper, "_get_with_retry", lambda url: _FakeResponse(html))
        listings = scraper.scrape_page("https://example.com/cat.php?p=1")
        assert isinstance(listings, list)

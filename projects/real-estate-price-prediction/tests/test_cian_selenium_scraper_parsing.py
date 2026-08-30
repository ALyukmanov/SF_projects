"""
Pure-parsing tests for ``src/data_collection/cian_selenium_scraper.py``.

ENVIRONMENT BLOCKER (documented, not a code defect)
----------------------------------------------------
``cian_selenium_scraper.py`` imports ``selenium`` and ``webdriver_manager``
unconditionally at module level (``from selenium import webdriver``, ``from
webdriver_manager.chrome import ChromeDriverManager``, ...). Both packages
are pinned in ``requirements.txt`` (``selenium==4.16.0``,
``webdriver-manager==4.0.1``) but are not installed in every environment
this suite runs in. As a result this entire module is skipped via
``pytest.importorskip`` rather than failing — this is an environment/
tooling gap, not a bug in the scraper. To make these tests actually execute:
``pip install -r requirements.txt`` (or at minimum ``selenium`` and
``webdriver-manager``) in the environment running the suite, then re-run.

What IS safely testable without a real browser/webdriver
----------------------------------------------------------
The module-level regex helpers (``_parse_price``, ``_parse_rooms``,
``_parse_area``, ``_parse_floor_info``, ``_safe_text``) are pure string
functions independent of Selenium's WebElement/WebDriver types — these are
covered below once the module import succeeds.

What is explicitly OUT of scope for this pass
------------------------------------------------
``CianSeleniumScraper._parse_card`` / ``_extract_card_data`` / the
``_wait_for_cards`` / ``_scroll_to_bottom`` DOM-interaction methods operate
on live Selenium ``WebElement`` objects (``card.find_element(...)``,
``.text``, ``.get_attribute(...)``). Testing these without a real (or heavily
mocked) browser would require either a running headless Chrome + webdriver
in CI, or hand-rolling fake WebElement objects that faithfully reproduce
Selenium's API surface (``find_element`` raising ``NoSuchElementException``,
etc.) — both are out of scope for this bounded pass per the task brief
(no architectural rewrite of the Selenium scraper). This is a real, tracked
coverage gap for the Selenium scraper's card-extraction logic, not a
false-negative: recorded as a roadmap item in the final test report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "selenium", reason="selenium not installed in this environment (see module docstring)"
)
pytest.importorskip(
    "webdriver_manager",
    reason="webdriver-manager not installed in this environment (see module docstring)",
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_collection.cian_selenium_scraper import (  # noqa: E402
    _parse_area,
    _parse_floor_info,
    _parse_price,
    _parse_rooms,
    _safe_text,
)


class TestParsePrice:
    def test_plain_digits(self):
        assert _parse_price("8500000") == 8500000.0

    def test_spaces_and_ruble_symbol(self):
        assert _parse_price("12 500 000 ₽") == 12500000.0

    def test_empty_returns_none(self):
        assert _parse_price("") is None


class TestParseRooms:
    def test_studio(self):
        assert _parse_rooms("Студия") == 0

    def test_standard_rooms(self):
        assert _parse_rooms("2-комн. квартира") == 2

    def test_dash_prefixed_form(self):
        assert _parse_rooms("3-комнатная квартира") == 3

    def test_unparseable_returns_none(self):
        assert _parse_rooms("хорошая квартира") is None


class TestParseArea:
    def test_comma_decimal(self):
        assert _parse_area("45,5 м²") == 45.5

    def test_empty_returns_none(self):
        assert _parse_area("") is None


class TestParseFloorInfo:
    def test_floor_and_total(self):
        assert _parse_floor_info("5/12 эт.") == (5, 12)

    def test_empty_returns_none_none(self):
        assert _parse_floor_info("") == (None, None)


class TestSafeText:
    def test_returns_stripped_text_from_object_with_text_attr(self):
        class _FakeElement:
            text = "  Кирпичный дом  "

        assert _safe_text(_FakeElement()) == "Кирпичный дом"

    def test_returns_empty_string_on_exception(self):
        class _RaisesOnAccess:
            @property
            def text(self):
                raise RuntimeError("stale element")

        assert _safe_text(_RaisesOnAccess()) == ""

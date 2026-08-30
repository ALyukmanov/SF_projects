"""
Pure, network-free parsing helpers shared by both CIAN scrapers.

These functions used to be duplicated almost verbatim in ``cian_scraper.py``
(requests+BeautifulSoup) and ``cian_selenium_scraper.py`` (Selenium) — this
module removes that duplication. They took only strings in and out (no network, no WebDriver, no
BeautifulSoup objects), so they are trivially unit-testable without internet
access — see ``tests/data_collection/test_parsing_utils.py``.

``PARSER_VERSION`` should be bumped whenever the *parsing behaviour* changes
(a regex gets stricter/looser, a new fallback strategy is added, etc.) so
that rows collected with different parser versions can be told apart later
(e.g. via a ``parser_version`` column on each collected listing).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

PARSER_VERSION = "1.0"

_FALLBACK_USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.76",
]

CITY_REGION_MAP: dict[str, int] = {
    "moskva": 1,
    "spb": 2,
    "ekaterinburg": 4743,
    "novosibirsk": 4897,
    "kazan": 4777,
    "nizhniy-novgorod": 4885,
    "samara": 4966,
    "krasnodar": 4820,
}

CITY_NAMES_RU: dict[str, str] = {
    "moskva": "Москва",
    "spb": "Санкт-Петербург",
    "ekaterinburg": "Екатеринбург",
    "novosibirsk": "Новосибирск",
    "kazan": "Казань",
    "nizhniy-novgorod": "Нижний Новгород",
    "samara": "Самара",
    "krasnodar": "Краснодар",
}

CIAN_BASE = "https://www.cian.ru"


def random_user_agent() -> str:
    """Return a random desktop-browser User-Agent string.

    Tries ``fake_useragent`` first (a larger, regularly-updated pool); falls
    back to a small hardcoded list if that package/network call is
    unavailable, so this never raises.
    """
    try:
        from fake_useragent import UserAgent  # type: ignore

        return UserAgent().random
    except Exception:
        import random

        return random.choice(_FALLBACK_USER_AGENTS)


def parse_price(raw: str) -> Optional[float]:
    """Extract a numeric price (RUB) from a Russian-formatted price string.

    Handles thousands separators (spaces, non-breaking spaces) and a
    trailing ``₽``/``руб`` currency marker by simply stripping every
    non-digit character. Returns ``None`` for empty/unparseable input.

    Examples:
        >>> parse_price("15 400 000 ₽")
        15400000.0
        >>> parse_price("")
        None
    """
    if not raw:
        return None
    cleaned = re.sub(r"[^\d]", "", raw)
    return float(cleaned) if cleaned else None


def parse_rooms(raw: str) -> Optional[int]:
    """Parse room count from a CIAN listing title/label.

    Examples: ``'2-комн.'``, ``'Студия'``, ``'3-к квартира'``. Studios map
    to ``0`` rooms (matches the convention used throughout this project's
    data pipeline — see ``src/preprocessing/cleaner.py``).
    """
    if not raw:
        return None
    raw_lower = raw.lower()
    if "студ" in raw_lower or "studio" in raw_lower:
        return 0
    match = re.search(r"(\d+)[-\s]*комн", raw_lower)
    if match:
        return int(match.group(1))
    match = re.search(r"^(\d+)[-\s]", raw.strip())
    if match:
        val = int(match.group(1))
        if 0 <= val <= 10:
            return val
    return None


def parse_area(raw: str) -> Optional[float]:
    """Extract total area in m² from strings like ``'45,5 м²'`` or ``'45.5 кв.м'``.

    Regression note: the original regex (``r"([\\d.]+)\\s*м"``) only matched a
    bare ``"м"`` directly after the number and whitespace, so it silently
    failed on the ``"кв.м"`` form its own docstring claimed to support (e.g.
    ``"45.5 кв.м"`` -> ``None``) — caught by
    ``tests/data_collection/test_parsing_utils.py::TestParseArea::test_dot_decimal_kvm``.
    """
    if not raw:
        return None
    cleaned = raw.replace(",", ".").replace("\xa0", " ")
    match = re.search(r"([\d.]+)\s*(?:кв\.?\s*)?м", cleaned)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def parse_floor_info(raw: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse ``'3/9 эт.'`` -> ``(3, 9)``, or ``'3 эт.'`` -> ``(3, None)``."""
    if not raw:
        return None, None
    match = re.search(r"(\d+)\s*/\s*(\d+)", raw)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"(\d+)\s*эт", raw)
    if match:
        return int(match.group(1)), None
    return None, None

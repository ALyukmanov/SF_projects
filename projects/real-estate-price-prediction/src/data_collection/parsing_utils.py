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

import json
import re
from typing import Optional, Tuple

PARSER_VERSION = "1.1"  # 1.1: added parse_listing_coordinates()

# Rough bounding box of Russia — used to reject obviously-wrong coordinates
# (0/0, a swapped lat/lon pair, a value picked up from an unrelated script).
_RU_LAT_RANGE = (41.0, 82.0)
_RU_LON_RANGE = (19.0, 180.0)

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


def _plausible_ru_point(lat: float, lon: float) -> bool:
    return (
        _RU_LAT_RANGE[0] <= lat <= _RU_LAT_RANGE[1] and _RU_LON_RANGE[0] <= lon <= _RU_LON_RANGE[1]
    )


def parse_listing_coordinates(html: str) -> Optional[Tuple[float, float]]:
    """Extract ``(latitude, longitude)`` from a restate.ru listing page.

    restate.ru puts the point on a lazy-loaded map ``<div>``:

        <div class="base__map-wrp" data-lat="55.7737" data-lng="37.5079" ...>

    That ``data-lat`` / ``data-lng`` pair is the primary and, in practice, the
    only source — detail pages carry no JSON-LD. A couple of generic fallbacks
    (``GeoCoordinates`` / an ``ll=lon,lat`` Yandex-style map parameter) are
    tried too so the function still works if the markup changes for some
    listings. Returns ``None`` if nothing plausible is found; the coordinates
    are sanity-checked against a rough bounding box of Russia and a swapped
    pair is corrected.

    Args:
        html: Raw HTML of a ``https://www.restate.ru/base/<id>.html`` page.

    Returns:
        ``(lat, lon)`` rounded to 6 decimals, or ``None``.
    """
    if not html:
        return None

    candidates: list[Tuple[float, float]] = []

    # 1. data-lat / data-lng (also data-latitude / data-longitude) attributes.
    lat_m = re.search(r'data-lat(?:itude)?\s*=\s*"(-?\d{1,3}\.\d+)"', html)
    lon_m = re.search(r'data-l(?:ng|on|ongitude)\s*=\s*"(-?\d{1,3}\.\d+)"', html)
    if lat_m and lon_m:
        candidates.append((float(lat_m.group(1)), float(lon_m.group(1))))

    # 2. schema.org GeoCoordinates / any embedded {"latitude": .., "longitude": ..}.
    geo_m = re.search(
        r'"latitude"\s*:\s*"?(-?\d{1,3}\.\d+)"?\s*,\s*"longitude"\s*:\s*"?(-?\d{1,3}\.\d+)"?',
        html,
    )
    if geo_m:
        candidates.append((float(geo_m.group(1)), float(geo_m.group(2))))

    # 3. Yandex-style map parameter ll=<lon>,<lat> (note: lon first).
    ll_m = re.search(r"[?&]ll=(-?\d{1,3}\.\d+)(?:,|%2C)(-?\d{1,3}\.\d+)", html)
    if ll_m:
        candidates.append((float(ll_m.group(2)), float(ll_m.group(1))))

    for lat, lon in candidates:
        if _plausible_ru_point(lat, lon):
            return round(lat, 6), round(lon, 6)
        if _plausible_ru_point(lon, lat):  # pair was swapped
            return round(lon, 6), round(lat, 6)
    return None


def coordinates_from_json_ld(raw_json: str) -> Optional[Tuple[float, float]]:
    """Pull a ``geo`` point out of a JSON-LD string, if it has one.

    Kept separate from :func:`parse_listing_coordinates` (which works on raw
    HTML) so it can be reused on the JSON-LD already stored for each listing
    by the search-results scraper without re-parsing HTML.
    """
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return None

    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            geo = node.get("geo")
            if isinstance(geo, dict) and "latitude" in geo and "longitude" in geo:
                try:
                    lat, lon = float(geo["latitude"]), float(geo["longitude"])
                except (TypeError, ValueError):
                    pass
                else:
                    if _plausible_ru_point(lat, lon):
                        return round(lat, 6), round(lon, 6)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None

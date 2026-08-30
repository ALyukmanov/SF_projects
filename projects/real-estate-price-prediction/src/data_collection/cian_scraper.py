"""
Requests-based CIAN scraper for Russian real estate listings.

Uses rotating User-Agent headers, random delays, retry logic with exponential
backoff and multiple HTML-parsing strategies (JSON-LD → embedded JS JSON →
article-tag fallback).
"""

import json
import random
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.data_collection.network import (
    build_session as _build_session_impl,
    fetch_with_retry as _fetch_with_retry,
    rotate_user_agent as _rotate_user_agent,
)
from src.data_collection.parsing_utils import (
    CIAN_BASE as _CIAN_BASE,
    CITY_NAMES_RU as _CITY_NAMES_RU,
    CITY_REGION_MAP as _CITY_REGION_MAP,
    PARSER_VERSION,
    parse_area as _parse_area,
    parse_floor_info as _parse_floor_info,
    parse_price as _parse_price,
    parse_rooms as _parse_rooms,
)
from src.data_collection.persistence import JsonlCheckpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)


class _ResponseLike:
    """Minimal ``requests.Response``-shaped wrapper around a ``FetchResult``.

    Exists so :meth:`CianScraper._get_with_retry`'s external contract
    (``response.text`` / ``response.status_code``) is unchanged for existing
    callers and tests after the retry logic itself moved into
    ``src/data_collection/network.py`` (Phase 1 parser-architecture
    refactor) — see ``tests/test_cian_scraper_parsing.py``'s module
    docstring, which documents monkeypatching ``_get_with_retry`` to return
    exactly this shape.
    """

    __slots__ = ("text", "status_code")

    def __init__(self, text: str, status_code: int) -> None:
        self.text = text
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Main scraper class
# ---------------------------------------------------------------------------


class CianScraper:
    """Requests-based scraper for cian.ru apartment listings.

    Args:
        city:      City key from :data:`_CITY_REGION_MAP` (default: ``'moskva'``).
        deal_type: ``'sale'`` or ``'rent'`` (default: ``'sale'``).
        delay_min: Minimum delay in seconds between requests (default: 2.0).
        delay_max: Maximum delay in seconds between requests (default: 5.0).
    """

    def __init__(
        self,
        city: str = "moskva",
        deal_type: str = "sale",
        delay_min: float = 2.0,
        delay_max: float = 5.0,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.city = city.lower()
        self.deal_type = deal_type
        self._delay_min = delay_min
        self._delay_max = delay_max

        self._region_id = _CITY_REGION_MAP.get(self.city, 1)
        self._city_name_ru = _CITY_NAMES_RU.get(self.city, self.city)

        self._session = self._build_session()
        self._seen_urls: set = set()

        # Optional checkpointing/resume — see src/data_collection/persistence.py.
        # Off by default (checkpoint_path=None) so existing callers/tests that
        # don't pass it see identical in-memory-only behaviour as before.
        self._checkpoint = JsonlCheckpoint(checkpoint_path) if checkpoint_path else None
        if self._checkpoint is not None:
            self._seen_urls |= self._checkpoint.load_seen_urls()
            if self._seen_urls:
                logger.info(
                    "Resuming from checkpoint %s: %d URL(s) already collected, will be skipped.",
                    checkpoint_path,
                    len(self._seen_urls),
                )

        logger.info(
            "CianScraper initialised | city=%s | region_id=%d | deal=%s",
            self.city,
            self._region_id,
            self.deal_type,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """Return page-1 URL for the configured city and deal type."""
        return (
            f"{_CIAN_BASE}/cat.php"
            f"?deal_type={self.deal_type}"
            f"&engine_version=2"
            f"&offer_type=flat"
            f"&region={self._region_id}"
            f"&p=1"
        )

    @property
    def delay_range(self) -> Tuple[float, float]:
        """Return ``(delay_min, delay_max)`` tuple."""
        return self._delay_min, self._delay_max

    # ------------------------------------------------------------------
    # Session setup
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        """Build the HTTP session. Delegates to ``network.build_session()``
        (Phase 1 refactor) — kept as a method (not called directly at import
        time) so subclasses/tests can still override it the same way as
        before this refactor.
        """
        return _build_session_impl()

    def _rotate_ua(self) -> None:
        """Rotate the User-Agent header. Delegates to ``network.rotate_user_agent()``."""
        _rotate_user_agent(self._session)

    # ------------------------------------------------------------------
    # Core scraping methods
    # ------------------------------------------------------------------

    def scrape_pages(
        self,
        base_url: Optional[str] = None,
        max_pages: int = 2,
    ) -> List[Dict[str, Any]]:
        """Scrape multiple pages and return aggregated listings.

        Args:
            base_url:  Optional override for the base pagination URL.
                       Must contain a ``{page}`` or ``&p=N`` placeholder.
                       If *None*, the scraper builds the URL from :attr:`city`
                       and :attr:`deal_type`.
            max_pages: Maximum number of pages to scrape (default: 2).

        Returns:
            A list of listing dictionaries with deduplicated entries.
        """
        all_listings: List[Dict[str, Any]] = []
        # NOTE: does NOT clear self._seen_urls when checkpoint-resume loaded
        # some at __init__ time — clearing here would defeat resume by
        # forgetting every URL from a prior run and re-fetching them.
        if self._checkpoint is None:
            self._seen_urls.clear()

        for page_num in range(1, max_pages + 1):
            page_url = self._build_page_url(base_url, page_num)
            logger.info("Scraping page %d/%d → %s", page_num, max_pages, page_url)

            listings = self._scrape_page_with_retry(page_url)
            if not listings:
                logger.warning("No listings found on page %d, stopping early.", page_num)
                break

            all_listings.extend(listings)
            if self._checkpoint is not None:
                self._checkpoint.append(listings)
            logger.info(
                "Page %d: +%d listings (total so far: %d)",
                page_num,
                len(listings),
                len(all_listings),
            )

            if page_num < max_pages:
                delay = random.uniform(self._delay_min, self._delay_max)
                logger.debug("Sleeping %.1f s before next page…", delay)
                time.sleep(delay)
                self._rotate_ua()

        logger.info("Scraping complete. Total listings collected: %d", len(all_listings))
        return all_listings

    def scrape_page(self, url: str) -> List[Dict[str, Any]]:
        """Parse a single CIAN search-results page.

        Tries multiple parsing strategies in order:
        1. JSON-LD ``<script>`` blocks
        2. Embedded ``window.__initialData__`` / ``window._cianConfig`` JS object
        3. ``<article>`` tag HTML fallback

        Args:
            url: Full URL of the search-results page.

        Returns:
            List of listing dicts (may be empty if parsing fails).
        """
        response = self._get_with_retry(url)
        if response is None:
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        listings: List[Dict[str, Any]] = []

        # Strategy 1 — JSON-LD
        listings = self._parse_json_ld(soup, url)
        if listings:
            logger.debug("Strategy 1 (JSON-LD) found %d listings.", len(listings))
            return self._deduplicate(listings)

        # Strategy 2 — embedded JSON in <script>
        listings = self._parse_embedded_json(soup, url)
        if listings:
            logger.debug("Strategy 2 (embedded JS JSON) found %d listings.", len(listings))
            return self._deduplicate(listings)

        # Strategy 3 — HTML article tags
        listings = self._parse_article_tags(soup, url)
        if listings:
            logger.debug("Strategy 3 (article tags) found %d listings.", len(listings))
            return self._deduplicate(listings)

        logger.warning("All parsing strategies returned 0 listings for %s", url)
        return []

    # ------------------------------------------------------------------
    # Parsing strategies
    # ------------------------------------------------------------------

    def _parse_json_ld(self, soup: BeautifulSoup, page_url: str) -> List[Dict[str, Any]]:
        listings: List[Dict[str, Any]] = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
            except (json.JSONDecodeError, TypeError):
                continue

            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                if data.get("@type") == "ItemList":
                    items = data.get("itemListElement", [])
                else:
                    items = [data]

            for item in items:
                listing = self._extract_from_ld_item(item, page_url)
                if listing:
                    listings.append(listing)

        return listings

    def _extract_from_ld_item(self, item: Any, page_url: str) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        offer_type = item.get("@type", "")
        if offer_type not in ("Offer", "Product", "Apartment", "Place"):
            return None

        url = item.get("url", "")
        if not url:
            return None

        price_info = item.get("offers", item)
        price_raw = price_info.get("price") or price_info.get("lowPrice", "")
        try:
            price = float(price_raw) if price_raw else None
        except (ValueError, TypeError):
            price = None

        name = item.get("name", "")

        return {
            "url": url,
            "price": price,
            "rooms": _parse_rooms(name),
            "total_area": None,
            "floor": None,
            "floors_total": None,
            "address": item.get("address", {}).get("streetAddress", "")
            if isinstance(item.get("address"), dict)
            else str(item.get("address", "")),
            "city": self._city_name_ru,
            "deal_type": self.deal_type,
            "scraped_at": datetime.now().isoformat(),
            "source_url": url,
            "source_city": self.city,
            "parser_version": PARSER_VERSION,
        }

    def _parse_embedded_json(self, soup: BeautifulSoup, page_url: str) -> List[Dict[str, Any]]:
        """Try to find listing data in embedded JavaScript variables."""
        listings: List[Dict[str, Any]] = []

        patterns = [
            r"window\.__initialData__\s*=\s*({.+?});\s*</script>",
            r"window\._cianConfig\s*=\s*({.+?});\s*</script>",
            r'"offersSerialized"\s*:\s*(\[.+?\])\s*[,}]',
            r'"offers"\s*:\s*(\[.+?\])\s*[,}]',
        ]

        full_html = str(soup)
        for pattern in patterns:
            match = re.search(pattern, full_html, re.DOTALL)
            if not match:
                continue
            raw = match.group(1)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            offer_list = data if isinstance(data, list) else self._dig_offers(data)
            for offer in offer_list:
                listing = self._extract_from_offer_json(offer)
                if listing:
                    listings.append(listing)
            if listings:
                break

        return listings

    def _dig_offers(self, data: Any, depth: int = 0) -> List[Any]:
        """Recursively search a nested dict/list structure for offer arrays."""
        if depth > 5:
            return []
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if "price" in data[0] or "totalArea" in data[0] or "offerType" in data[0]:
                return data
        if isinstance(data, dict):
            for key in ("offers", "offersSerialized", "items", "data", "results"):
                if key in data:
                    result = self._dig_offers(data[key], depth + 1)
                    if result:
                        return result
        return []

    def _extract_from_offer_json(self, offer: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(offer, dict):
            return None

        offer_id = offer.get("id") or offer.get("cianId")
        url = offer.get("fullUrl") or offer.get("url", "")
        if not url and offer_id:
            url = f"{_CIAN_BASE}/sale/flat/{offer_id}/"

        if not url:
            return None

        price_raw = (
            offer.get("bargainTerms", {}).get("price")
            or offer.get("price")
            or offer.get("priceRub")
        )
        try:
            price = float(price_raw) if price_raw is not None else None
        except (ValueError, TypeError):
            price = None

        rooms_raw = offer.get("roomsCount") or offer.get("rooms")
        rooms: Optional[int] = None
        if rooms_raw is not None:
            try:
                rooms = int(rooms_raw)
            except (ValueError, TypeError):
                rooms = _parse_rooms(str(rooms_raw))

        area_raw = offer.get("totalArea") or offer.get("area")
        total_area: Optional[float] = None
        if area_raw is not None:
            try:
                total_area = float(area_raw)
            except (ValueError, TypeError):
                pass

        floor_raw = offer.get("floorNumber") or offer.get("floor")
        floor: Optional[int] = None
        if floor_raw is not None:
            try:
                floor = int(floor_raw)
            except (ValueError, TypeError):
                pass

        building = offer.get("building") or {}
        floors_total_raw = building.get("floorsCount") if isinstance(building, dict) else None
        floors_total: Optional[int] = None
        if floors_total_raw is not None:
            try:
                floors_total = int(floors_total_raw)
            except (ValueError, TypeError):
                pass

        geo = offer.get("geo") or {}
        address = ""
        if isinstance(geo, dict):
            addr_parts = [a.get("name", "") for a in geo.get("address", []) if isinstance(a, dict)]
            address = ", ".join(filter(None, addr_parts))

        return {
            "url": url,
            "price": price,
            "rooms": rooms,
            "total_area": total_area,
            "floor": floor,
            "floors_total": floors_total,
            "address": address,
            "city": self._city_name_ru,
            "deal_type": self.deal_type,
            "scraped_at": datetime.now().isoformat(),
            "source_url": url,
            "source_city": self.city,
            "parser_version": PARSER_VERSION,
        }

    def _parse_article_tags(self, soup: BeautifulSoup, page_url: str) -> List[Dict[str, Any]]:
        """HTML fallback: parse <article> elements as listing cards."""
        listings: List[Dict[str, Any]] = []

        # CIAN wraps each listing in an <article> or a div with data-name="CardComponent"
        cards = soup.find_all("article") or soup.find_all(
            "div", attrs={"data-name": "CardComponent"}
        )

        if not cards:
            # Try generic listing card selectors
            cards = soup.select("[class*='card'], [class*='listing-item'], [class*='offer-item']")

        logger.debug("Found %d card elements via article-tag strategy.", len(cards))

        for card in cards:
            listing = self._parse_single_card(card, page_url)
            if listing:
                listings.append(listing)

        return listings

    def _parse_single_card(self, card: Any, page_url: str) -> Optional[Dict[str, Any]]:
        # URL
        link_tag = card.find("a", href=True)
        url = ""
        if link_tag:
            href = link_tag["href"]
            if href.startswith("http"):
                url = href
            else:
                url = urljoin(_CIAN_BASE, href)

        if not url:
            return None

        # Price — look for elements containing ₽ or "руб"
        price: Optional[float] = None
        price_candidates = card.find_all(string=re.compile(r"[\d\s]+[₽р]|руб", re.IGNORECASE))
        for cand in price_candidates:
            parsed = _parse_price(str(cand))
            if parsed and parsed > 100_000:
                price = parsed
                break

        # Rooms & area from title / heading
        title_tag = card.find(["h1", "h2", "h3", "span"], string=re.compile(r"комн|студ", re.I))
        rooms: Optional[int] = None
        total_area: Optional[float] = None
        if title_tag:
            rooms = _parse_rooms(title_tag.get_text())
            total_area = _parse_area(title_tag.get_text())

        # Floor info
        floor_tag = card.find(string=re.compile(r"\d+\s*/\s*\d+\s*эт", re.I))
        floor, floors_total = _parse_floor_info(str(floor_tag) if floor_tag else "")

        # Address
        addr_tag = card.find(["address", "span"], string=re.compile(r"ул\.|пр\.|пер\.|д\.", re.I))
        address = addr_tag.get_text(strip=True) if addr_tag else ""

        return {
            "url": url,
            "price": price,
            "rooms": rooms,
            "total_area": total_area,
            "floor": floor,
            "floors_total": floors_total,
            "address": address,
            "city": self._city_name_ru,
            "deal_type": self.deal_type,
            "scraped_at": datetime.now().isoformat(),
            "source_url": url,
            "source_city": self.city,
            "parser_version": PARSER_VERSION,
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get_with_retry(
        self,
        url: str,
        max_retries: int = 3,
        base_delay: float = 10.0,
    ) -> Optional["_ResponseLike"]:
        """GET *url* with exponential backoff retry logic.

        Delegates the actual retry/backoff/give-up policy to
        ``network.fetch_with_retry()`` (Phase 1 parser-architecture
        refactor — network I/O is now a standalone, independently testable
        module, see ``src/data_collection/network.py``). This method's
        external contract (return value has ``.text``/``.status_code``, or
        ``None`` on failure) is unchanged, so existing callers and tests
        that monkeypatch ``_get_with_retry`` keep working unmodified.
        """
        result = _fetch_with_retry(
            self._session, url, max_retries=max_retries, base_delay=base_delay
        )
        if not result.ok or result.text is None:
            return None
        return _ResponseLike(text=result.text, status_code=result.status_code or 200)

    def _scrape_page_with_retry(self, url: str) -> List[Dict[str, Any]]:
        """Wrapper that calls :meth:`scrape_page` and returns an empty list on error."""
        try:
            return self.scrape_page(url)
        except Exception as exc:
            logger.error("Unexpected error scraping %s: %s", url, exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _build_page_url(self, base_url: Optional[str], page_num: int) -> str:
        if base_url:
            # Replace existing &p=N or append
            if re.search(r"[&?]p=\d+", base_url):
                return re.sub(r"([&?]p=)\d+", rf"\g<1>{page_num}", base_url)
            sep = "&" if "?" in base_url else "?"
            return f"{base_url}{sep}p={page_num}"
        return (
            f"{_CIAN_BASE}/cat.php"
            f"?deal_type={self.deal_type}"
            f"&engine_version=2"
            f"&offer_type=flat"
            f"&region={self._region_id}"
            f"&p={page_num}"
        )

    def _deduplicate(self, listings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove listings whose URL has already been seen in this session."""
        unique: List[Dict[str, Any]] = []
        for listing in listings:
            url = listing.get("url", "")
            if url and url not in self._seen_urls:
                self._seen_urls.add(url)
                unique.append(listing)
        return unique

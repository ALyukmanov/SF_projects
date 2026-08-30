"""
Selenium-based CIAN scraper for Russian real estate listings.

Handles JavaScript-rendered pages using Chrome WebDriver managed by
webdriver-manager.  Uses explicit waits, lazy-loading scroll, and
StaleElementReferenceException retry logic.
"""

import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from src.data_collection.parsing_utils import (
    CITY_NAMES_RU as _CITY_NAMES_RU,
    PARSER_VERSION,
    parse_area as _parse_area,
    parse_floor_info as _parse_floor_info,
    parse_price as _parse_price,
    parse_rooms as _parse_rooms,
    random_user_agent as _random_ua,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CSS selectors tried in order when looking for listing cards
_CARD_SELECTORS: List[str] = [
    "[data-name='CardComponent']",
    "article[data-name]",
    "article",
    "._93444fe79c--card--ibP9o",
    "[class*='card--']",
    "[class*='offer-card']",
    "[class*='listing-item']",
]

_CITY_URLS: Dict[str, str] = {
    "moskva": "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region=1&p={page}",
    "spb": "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region=2&p={page}",
    "ekaterinburg": "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region=4743&p={page}",
    "novosibirsk": "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region=4897&p={page}",
    "kazan": "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region=4777&p={page}",
    "nizhniy-novgorod": "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region=4885&p={page}",
    "samara": "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region=4966&p={page}",
    "krasnodar": "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&region=4820&p={page}",
}

# ---------------------------------------------------------------------------
# Parsing helpers
#
# _parse_price/_parse_rooms/_parse_area/_parse_floor_info and _CITY_NAMES_RU
# used to be independently duplicated here and in cian_scraper.py (they had
# already silently drifted — this scraper's _parse_rooms fallback regex
# required a trailing '-', the requests scraper's didn't). Both now import
# the same functions from src.data_collection.parsing_utils, the single
# source of truth.
# ---------------------------------------------------------------------------


def _safe_text(element) -> str:
    """Return stripped text from a WebElement, or empty string on failure."""
    try:
        return element.text.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main scraper class
# ---------------------------------------------------------------------------


class CianSeleniumScraper:
    """Chrome WebDriver-based scraper for cian.ru apartment listings.

    Args:
        city:     City key matching :data:`_CITY_URLS` (default: ``'moskva'``).
        headless: Run Chrome in headless mode (default: ``True``).
        delay_min: Minimum inter-page delay in seconds (default: 3.0).
        delay_max: Maximum inter-page delay in seconds (default: 7.0).
    """

    def __init__(
        self,
        city: str = "moskva",
        headless: bool = True,
        delay_min: float = 3.0,
        delay_max: float = 7.0,
    ) -> None:
        self.city = city.lower()
        self.headless = headless
        self._delay_min = delay_min
        self._delay_max = delay_max
        self._city_name_ru = _CITY_NAMES_RU.get(self.city, self.city)
        self._url_template = _CITY_URLS.get(
            self.city,
            "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2"
            "&offer_type=flat&region=1&p={page}",
        )
        self._driver: Optional[webdriver.Chrome] = None
        self._seen_urls: set = set()

        self._driver = self._build_driver()
        logger.info("CianSeleniumScraper initialised | city=%s | headless=%s", self.city, headless)

    # ------------------------------------------------------------------
    # Driver setup
    # ------------------------------------------------------------------

    def _build_driver(self) -> webdriver.Chrome:
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--lang=ru-RU")

        ua = _random_ua()
        options.add_argument(f"--user-agent={ua}")

        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
        except Exception as exc:
            logger.warning("webdriver-manager failed (%s). Trying system chromedriver…", exc)
            driver = webdriver.Chrome(options=options)

        # Stealth tweak — remove webdriver property
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    window.chrome = { runtime: {} };
                """
            },
        )
        logger.debug("Chrome WebDriver started.")
        return driver

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def scrape_pages(self, max_pages: int = 10) -> List[Dict[str, Any]]:
        """Scrape up to *max_pages* listing pages and return all listings.

        Args:
            max_pages: Maximum number of pages to process (default: 10).

        Returns:
            Deduplicated list of listing dicts.
        """
        all_listings: List[Dict[str, Any]] = []
        self._seen_urls.clear()

        for page_num in range(1, max_pages + 1):
            url = self._url_template.format(page=page_num)
            logger.info("Scraping page %d/%d → %s", page_num, max_pages, url)

            try:
                listings = self.scrape_page(url)
            except Exception as exc:
                logger.error("Fatal error on page %d: %s", page_num, exc, exc_info=True)
                break

            if not listings:
                logger.warning("No listings on page %d — stopping early.", page_num)
                break

            # Deduplicate across pages
            new_listings = [item for item in listings if item.get("url") not in self._seen_urls]
            for item in new_listings:
                self._seen_urls.add(item.get("url", ""))
            all_listings.extend(new_listings)

            logger.info(
                "Page %d: +%d new listings (total: %d)",
                page_num,
                len(new_listings),
                len(all_listings),
            )

            if page_num < max_pages:
                delay = random.uniform(self._delay_min, self._delay_max)
                logger.debug("Sleeping %.1f s…", delay)
                time.sleep(delay)

        logger.info("Selenium scraping done. Total listings: %d", len(all_listings))
        return all_listings

    def scrape_page(self, url: str) -> List[Dict[str, Any]]:
        """Load a single search-results page and extract all listing cards.

        Args:
            url: Full URL of the CIAN search-results page.

        Returns:
            List of listing dicts extracted from visible cards.
        """
        if not self._driver:
            logger.error("WebDriver not initialised.")
            return []

        try:
            self._driver.get(url)
        except WebDriverException as exc:
            logger.error("Failed to load %s: %s", url, exc)
            return []

        # Wait for page content
        cards_located = self._wait_for_cards()
        if not cards_located:
            logger.warning("No listing cards appeared at %s", url)
            return []

        # Scroll to trigger lazy loading
        self._scroll_to_bottom()

        # Give dynamic content a moment to settle
        time.sleep(1.5)

        return self._extract_all_cards()

    def close(self) -> None:
        """Quit the Chrome WebDriver instance."""
        if self._driver:
            try:
                self._driver.quit()
                logger.info("WebDriver closed.")
            except Exception as exc:
                logger.warning("Error closing WebDriver: %s", exc)
            finally:
                self._driver = None

    # Context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_cards(self, timeout: int = 20) -> bool:
        """Wait until at least one listing card appears in the DOM.

        Tries each selector in :data:`_CARD_SELECTORS` in order.

        Returns:
            ``True`` if cards were found within *timeout* seconds.
        """
        for selector in _CARD_SELECTORS:
            try:
                WebDriverWait(self._driver, timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                logger.debug("Cards found with selector: %s", selector)
                return True
            except TimeoutException:
                continue
        return False

    def _scroll_to_bottom(self) -> None:
        """Scroll the page to the bottom to trigger lazy-loaded images/data."""
        try:
            last_height = self._driver.execute_script("return document.body.scrollHeight")
            for _ in range(5):
                self._driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.6)
                new_height = self._driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
        except Exception as exc:
            logger.debug("Scroll error (non-fatal): %s", exc)

    def _extract_all_cards(self) -> List[Dict[str, Any]]:
        """Find all card elements using known selectors and extract data."""
        cards = []
        for selector in _CARD_SELECTORS:
            try:
                found = self._driver.find_elements(By.CSS_SELECTOR, selector)
                if found:
                    cards = found
                    logger.debug("Using selector '%s' — found %d cards.", selector, len(cards))
                    break
            except Exception:
                continue

        listings: List[Dict[str, Any]] = []
        for card in cards:
            listing = self._extract_card_data(card)
            if listing:
                listings.append(listing)
        return listings

    def _extract_card_data(self, card, retries: int = 2) -> Optional[Dict[str, Any]]:
        """Extract structured data from a single card WebElement.

        Retries on :class:`StaleElementReferenceException`.
        """
        for attempt in range(retries + 1):
            try:
                return self._parse_card(card)
            except StaleElementReferenceException:
                if attempt < retries:
                    time.sleep(0.3)
                    continue
                logger.debug("StaleElementReferenceException — skipping card.")
                return None
            except Exception as exc:
                logger.debug("Card parse error: %s", exc)
                return None
        return None

    def _parse_card(self, card) -> Optional[Dict[str, Any]]:
        """Extract listing fields from a card WebElement."""
        # --- URL -------------------------------------------------------
        url = ""
        try:
            link = card.find_element(By.TAG_NAME, "a")
            url = link.get_attribute("href") or ""
        except NoSuchElementException:
            pass

        if not url:
            return None

        # --- Price -------------------------------------------------------
        price: Optional[float] = None
        price_selectors = [
            "[data-name='PriceInfo']",
            "[class*='price']",
            "[class*='Price']",
        ]
        for sel in price_selectors:
            try:
                price_el = card.find_element(By.CSS_SELECTOR, sel)
                price_text = _safe_text(price_el)
                parsed = _parse_price(price_text)
                if parsed and parsed > 100_000:
                    price = parsed
                    break
            except NoSuchElementException:
                continue

        # --- Title / rooms / area ----------------------------------------
        rooms: Optional[int] = None
        total_area: Optional[float] = None
        title_selectors = [
            "[data-name='GeneralInfoSectionRowComponent']",
            "[class*='title']",
            "[class*='header']",
            "h3",
            "h2",
        ]
        for sel in title_selectors:
            try:
                title_el = card.find_element(By.CSS_SELECTOR, sel)
                title_text = _safe_text(title_el)
                if not title_text:
                    continue
                if rooms is None:
                    rooms = _parse_rooms(title_text)
                if total_area is None:
                    total_area = _parse_area(title_text)
                if rooms is not None and total_area is not None:
                    break
            except NoSuchElementException:
                continue

        # --- Floor -------------------------------------------------------
        floor: Optional[int] = None
        floors_total: Optional[int] = None
        floor_selectors = [
            "[data-name='FloorInfo']",
            "[class*='floor']",
            "[class*='Floor']",
        ]
        for sel in floor_selectors:
            try:
                floor_el = card.find_element(By.CSS_SELECTOR, sel)
                floor_text = _safe_text(floor_el)
                floor, floors_total = _parse_floor_info(floor_text)
                if floor is not None:
                    break
            except NoSuchElementException:
                continue

        # If still no floor, try searching card text
        if floor is None:
            full_text = _safe_text(card)
            floor, floors_total = _parse_floor_info(full_text)

        # --- Address -------------------------------------------------------
        address = ""
        addr_selectors = [
            "[data-name='AddressContainer']",
            "[class*='address']",
            "[class*='Address']",
        ]
        for sel in addr_selectors:
            try:
                addr_el = card.find_element(By.CSS_SELECTOR, sel)
                address = _safe_text(addr_el)
                if address:
                    break
            except NoSuchElementException:
                continue

        return {
            "url": url,
            "price": price,
            "rooms": rooms,
            "total_area": total_area,
            "floor": floor,
            "floors_total": floors_total,
            "address": address,
            "city": self._city_name_ru,
            "scraped_at": datetime.now().isoformat(),
            "source_url": url,
            "source_city": self.city,
            "parser_version": PARSER_VERSION,
        }

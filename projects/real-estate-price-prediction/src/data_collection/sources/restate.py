"""
Restate.ru scraper -- the project's PRIMARY real-data source.

Selected 2026-08-24 after live probes confirmed:
  - CIAN: HTTP 200 -> immediate redirect to /cian-captcha/, confirmed on
    repeated checks. Confirmed dead end, no bypass attempted. See
    src/data_collection/cian_scraper.py, which is kept dormant, not deleted.
  - Restate.ru: msk.restate.ru/choice/moscow/<category> returns HTTP 200
    with a real schema.org `ItemList` of `RealEstateListing` items embedded
    as JSON-LD -- real prices, areas, rooms, floors, addresses, listing
    URLs. No CAPTCHA encountered. robots.txt for msk.restate.ru explicitly
    does NOT disallow `/choice/` (it disallows `/search/`, `/list/`,
    `/complex_show`, `/material/`, and several other paths, but not the
    path this scraper uses).
  - Raw HTML/robots.txt from that probe were saved as evidence at the time.

Scope decision -- detail pages are NOT fetched in bulk
--------------------------------------------------------
Each individual listing detail page (`https://www.restate.ru/base/<id>.html`)
carries `<meta name="robots" content="noindex, nofollow">`. robots.txt does
not forbid fetching these pages, but that meta tag is a clear, deliberate
site-owner signal that they are not meant to be crawled/indexed at scale --
unlike the `/choice/...` search-results pages, whose rich JSON-LD markup is
obviously built to be crawled (it is textbook SEO rich-snippet structured
data). This scraper respects that distinction: every field it produces
comes from the search-results JSON-LD, and fields that are genuinely only
available on detail pages (coordinates, kitchen area, exact publish date)
are left ``None`` rather than fetched. See the project's DATA_CARD.md /
README.md before changing this. (`building_type` and `ceiling_height` turned
out NOT to be detail-page-only -- see ``normalize.py``: the search-results
``description`` text itself contains "тип дома: ..." / "высота потолков ..."
for many listings, discovered 2026-08-24 while sampling live category pages.)

Scope expansion (2026-08-25) -- categories, second city, pagination
-----------------------------------------------------------------------------
Live-probed directly (single sparing GETs per URL, politeness delay between
each; the full evidence trail was recorded at the time):

- Three more *_sale category slugs, confirmed real and reachable, linked
  from the site's own navigation on the pages already scraped:
  ``new_erect_flats_sale`` (new-build flats), ``cottages_sale``
  (houses/cottages -- ``itemOffered.@type`` is schema.org ``House``, not
  ``Apartment``), ``room_sale`` (individual rooms within an apartment).
  ``owners_flats_sale``/``lands_sale``/``offices_*``/``warehouses_*`` were
  also found in the nav but declined: the first is a same-listings
  "from owner" filter over categories already scraped (would only add
  wasted requests, not new items), the rest are non-residential or raw
  land, out of this project's scope.
- A second city subdomain, ``spb.restate.ru``, confirmed live with slug
  ``petersburg`` (NOT ``spb`` -- that 404s; ``petersburg`` was found by
  grepping the site's own internal nav links). "Санкт-Петербург" is already
  one of the fixed 8 cities in ``feature_engineering.CITY_SLUGS``, so no ML
  contract change is needed to consume this city's data.
- **Pagination depth cap, found by direct probing**: regardless of the
  ``numberOfItems`` a category reports (e.g. 11,507 for Moscow 1-room
  flats), ``?page=N`` past roughly page 24 silently starts returning
  page 1's content again -- HTTP 200, well-formed markup, no error signal.
  The scraper must treat "this page's items are all URLs we've already
  seen" as an end-of-category signal, not just "this page had 0 items"
  (see ``ScrapeStats`` / the loop in :meth:`RestateScraper.scrape`).
- The site's own sort-order control (a real ``<select name="o">`` on every
  category page, not a guessed parameter) reorders the underlying result
  set before the same ~24-page truncation is applied, so sweeping a
  category under a few different sort orders (default / price ascending /
  price descending) surfaces a different, mostly non-overlapping slice of
  its inventory each time -- see :data:`RESTATE_SORT_ORDERS`.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from src.data_collection.network import (
    build_session as _build_session_impl,
    fetch_with_retry as _fetch_with_retry,
    rotate_user_agent as _rotate_user_agent,
)
from src.data_collection.persistence import JsonlCheckpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)

PARSER_VERSION = "1.1"
SOURCE_NAME = "restate"

# Only cities actually confirmed live are listed here. Do not add a city
# without first confirming its subdomain/slug with a real request (see
# README.md "Adding a new city to the restate.ru source"). This is a
# registry, not an if/elif chain -- adding a city is a data change here,
# never a new branch in the scraping/normalization logic.
RESTATE_CITIES: Dict[str, Dict[str, str]] = {
    "moscow": {"subdomain": "msk", "slug": "moscow", "name_ru": "Москва"},
    "petersburg": {"subdomain": "spb", "slug": "petersburg", "name_ru": "Санкт-Петербург"},
}


@dataclass(frozen=True)
class CategorySpec:
    """One /choice/<city>/<slug> sale category, plus the property-type
    metadata needed to tag every listing it yields honestly.

    ``property_type`` reflects which category URL a listing was found
    under (a SOURCE-level fact -- which page we fetched it from), not a
    guess from free text. ``market_segment`` is only set to a specific
    value when the category itself unambiguously means that (currently
    only ``new_erect_flats_sale`` -> "new_build"); it stays ``None`` for
    categories where the site does not disambiguate secondary vs. new-build
    (the plain room-count categories contain a mix of both).
    """

    slug: str
    property_type: str  # "flat" | "house" | "room"
    market_segment: Optional[str] = None


# Sale categories confirmed reachable 2026-08-24/2026-08-25 (linked from the
# site's own navigation, under msk.restate.ru/choice/moscow/... and
# spb.restate.ru/choice/petersburg/...). See the module docstring for what
# was found and declined.
RESTATE_SALE_CATEGORIES: List[CategorySpec] = [
    CategorySpec("studio_flats_sale", property_type="flat"),
    CategorySpec("1_rooms_flats_sale", property_type="flat"),
    CategorySpec("2_rooms_flats_sale", property_type="flat"),
    CategorySpec("3_rooms_flats_sale", property_type="flat"),
    CategorySpec("new_erect_flats_sale", property_type="flat", market_segment="new_build"),
    CategorySpec("cottages_sale", property_type="house"),
    CategorySpec("room_sale", property_type="room"),
]

# Values taken verbatim from the live <select name="o"> sort control found
# on msk.restate.ru/choice/moscow/1_rooms_flats_sale (confirmed 2026-08-25).
# Only a subset is used by default -- see RestateScraper.scrape's
# `sort_orders` argument.
RESTATE_SORT_ORDERS: Dict[str, int] = {
    "default": 0,
    "price_asc": 1,
    "price_desc": 2,
}


@dataclass
class ScrapeStats:
    """Per-run counters, surfaced by the CLI's final report."""

    pages_fetched: int = 0
    pages_failed: int = 0
    items_seen_on_pages: int = 0
    items_new: int = 0
    items_duplicate: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)


class RestateScraper:
    """Fetches restate.ru `/choice/<city>/<category>` search-results pages
    and extracts the embedded schema.org JSON-LD `ItemList`.

    This class returns RAW schema.org item dicts (plus minimal capture
    metadata) -- it does NOT parse `name`/`description` free text into
    structured fields (area, floor, seller type, ...). That is a separate,
    independently testable step: see `src/data_collection/normalize.py`.
    Keeping raw capture and normalization separate means the raw archive
    genuinely reflects what the site returned, and normalization logic can
    be re-run/fixed later without re-fetching anything.

    Args:
        city: Key into :data:`RESTATE_CITIES` (default: ``"moscow"``).
        categories: :class:`CategorySpec` list to sweep (default: all of
            :data:`RESTATE_SALE_CATEGORIES`).
        sort_orders: Names (keys of :data:`RESTATE_SORT_ORDERS`) to sweep
            per category (default: just ``["default"]``, the original
            behaviour). Pass e.g. ``["default", "price_asc",
            "price_desc"]`` to access more of a large category's inventory
            past the ~24-page crawl depth cap -- see the module docstring.
        delay_min / delay_max: Jittered inter-request delay in seconds
            between pages (politeness -- see module docstring).
        checkpoint_path: Optional path to a :class:`JsonlCheckpoint`. When
            set, already-checkpointed listing URLs are skipped on
            (re)construction -- this is single-run crash/resume support,
            the same contract as ``CianScraper``, not cross-run price-change
            tracking (that lives in the CLI's incremental-report logic).
    """

    def __init__(
        self,
        city: str = "moscow",
        categories: Optional[List[CategorySpec]] = None,
        sort_orders: Optional[List[str]] = None,
        delay_min: float = 1.5,
        delay_max: float = 3.5,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        if city not in RESTATE_CITIES:
            raise ValueError(
                f"Unknown/unconfirmed restate.ru city '{city}'. "
                f"Known: {sorted(RESTATE_CITIES)}. Confirm with a live request before adding one."
            )
        self.city = city
        self.categories = categories or list(RESTATE_SALE_CATEGORIES)
        self.sort_orders = sort_orders or ["default"]
        unknown_sorts = set(self.sort_orders) - set(RESTATE_SORT_ORDERS)
        if unknown_sorts:
            raise ValueError(
                f"Unknown sort order(s) {sorted(unknown_sorts)}. Known: {sorted(RESTATE_SORT_ORDERS)}."
            )
        self._delay_min = delay_min
        self._delay_max = delay_max
        self._city_cfg = RESTATE_CITIES[city]

        self._session = _build_session_impl()
        self._seen_urls: set[str] = set()

        self._checkpoint = JsonlCheckpoint(checkpoint_path) if checkpoint_path else None
        if self._checkpoint is not None:
            self._seen_urls |= self._checkpoint.load_seen_urls()
            if self._seen_urls:
                logger.info(
                    "Resuming from checkpoint %s: %d URL(s) already collected this run, will be skipped.",
                    checkpoint_path,
                    len(self._seen_urls),
                )

        logger.info(
            "RestateScraper initialised | city=%s | categories=%s | sort_orders=%s",
            city,
            [c.slug for c in self.categories],
            self.sort_orders,
        )

    # ------------------------------------------------------------------
    def category_url(self, category: str, page: int = 1, sort: str = "default") -> str:
        cfg = self._city_cfg
        url = f"https://{cfg['subdomain']}.restate.ru/choice/{cfg['slug']}/{category}"
        params = []
        if page > 1:
            params.append(f"page={page}")
        sort_value = RESTATE_SORT_ORDERS[sort]
        if sort_value != RESTATE_SORT_ORDERS["default"]:
            params.append(f"o={sort_value}")
        if params:
            url += "?" + "&".join(params)
        return url

    # ------------------------------------------------------------------
    def scrape(
        self,
        max_pages_per_category: int = 3,
        limit: Optional[int] = None,
    ) -> tuple[List[Dict[str, Any]], ScrapeStats]:
        """Sweep configured categories x sort orders, page by page.

        Stops a (category, sort) sweep early when either:
        - a page yields 0 parsed items (assumed end of that sweep's
          results), or
        - a page's items are ALL already in ``_seen_urls`` (assumed the
          site's ~24-page crawl-depth cap was hit -- past that point
          restate.ru silently repeats page 1's content instead of erroring,
          see the module docstring). This is a heuristic: on a large
          category's very first page it is vanishingly unlikely all 20
          items already happen to be duplicates from an earlier sweep, so
          treating it as "cap reached" rather than "coincidence" is safe in
          practice, and a false-early-stop here only means slightly fewer
          items collected, not a correctness bug.

        Stops the whole run once *limit* new (non-duplicate) items have
        been collected, if given.

        Returns:
            ``(raw_records, stats)`` -- raw_records are ready to append to
            a :class:`JsonlCheckpoint` / write to ``data/raw/...``.
        """
        collected: List[Dict[str, Any]] = []
        stats = ScrapeStats()

        for category in self.categories:
            for sort in self.sort_orders:
                for page in range(1, max_pages_per_category + 1):
                    if limit is not None and len(collected) >= limit:
                        logger.info("Reached limit=%d, stopping run.", limit)
                        return collected[:limit], stats

                    url = self.category_url(category.slug, page, sort=sort)
                    page_items, fetch_ok, fail_reason = self._fetch_and_parse(url, category)

                    if not fetch_ok:
                        stats.pages_failed += 1
                        stats.errors.append({"url": url, "reason": fail_reason})
                        logger.error("Failed %s: %s", url, fail_reason)
                        break  # don't keep paging a sweep we can't reach

                    stats.pages_fetched += 1
                    stats.items_seen_on_pages += len(page_items)

                    new_items = []
                    for raw in page_items:
                        if raw["url"] in self._seen_urls:
                            stats.items_duplicate += 1
                            continue
                        self._seen_urls.add(raw["url"])
                        new_items.append(raw)
                    stats.items_new += len(new_items)

                    if self._checkpoint is not None and new_items:
                        self._checkpoint.append(new_items)
                    collected.extend(new_items)

                    logger.info(
                        "%s/%s page %d: %d item(s) on page, %d new (run total: %d)",
                        category.slug,
                        sort,
                        page,
                        len(page_items),
                        len(new_items),
                        len(collected),
                    )

                    if not page_items:
                        logger.info(
                            "No items parsed on %s/%s page %d -- assuming end of sweep.",
                            category.slug,
                            sort,
                            page,
                        )
                        break

                    if page_items and not new_items:
                        logger.info(
                            "%s/%s page %d returned only already-seen URLs -- assuming crawl-depth "
                            "cap reached, stopping this sweep.",
                            category.slug,
                            sort,
                            page,
                        )
                        break

                    if page < max_pages_per_category:
                        delay = random.uniform(self._delay_min, self._delay_max)
                        time.sleep(delay)
                        _rotate_user_agent(self._session)

        return collected[:limit] if limit is not None else collected, stats

    # ------------------------------------------------------------------
    def _fetch_and_parse(
        self, url: str, category: Optional[CategorySpec] = None
    ) -> tuple[List[Dict[str, Any]], bool, Optional[str]]:
        result = _fetch_with_retry(self._session, url, max_retries=3, base_delay=8.0)
        if not result.ok or result.text is None:
            return [], False, result.gave_up_reason
        try:
            items = self._extract_item_list(result.text, url, category=category)
        except Exception as exc:  # pragma: no cover - defensive, logged not swallowed silently
            logger.error("Unexpected parse error for %s: %s", url, exc, exc_info=True)
            return [], False, f"parse_error:{exc}"
        return items, True, None

    def _extract_item_list(
        self, html: str, page_url: str, category: Optional[CategorySpec] = None
    ) -> List[Dict[str, Any]]:
        """Return raw schema.org `RealEstateListing` item dicts found in
        this page's JSON-LD `ItemList`, each wrapped with capture metadata.
        """
        soup = BeautifulSoup(html, "html.parser")
        fetched_at = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []

        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
            except (json.JSONDecodeError, TypeError):
                continue
            graph = data.get("@graph") if isinstance(data, dict) else None
            if not graph:
                continue
            for node in graph:
                if not isinstance(node, dict) or node.get("@type") != "ItemList":
                    continue
                for li in node.get("itemListElement", []):
                    item = li.get("item") if isinstance(li, dict) else None
                    if not isinstance(item, dict) or item.get("@type") != "RealEstateListing":
                        continue
                    url = item.get("url")
                    if not url:
                        continue
                    records.append(
                        {
                            "url": url,
                            "source": SOURCE_NAME,
                            "source_page_url": page_url,
                            "fetched_at": fetched_at,
                            "parser_version": PARSER_VERSION,
                            "raw_item": item,  # verbatim schema.org node, as emitted by the site
                            # Which /choice/ category this listing was found under -- a
                            # SOURCE fact (which page we fetched), not a text-based guess.
                            "source_category": category.slug if category else None,
                            "property_type": category.property_type if category else None,
                            "market_segment": category.market_segment if category else None,
                        }
                    )
        return records

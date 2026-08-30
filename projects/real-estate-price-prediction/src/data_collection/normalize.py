"""
Normalization: raw source-specific records -> the project's flat listing
schema (see the field list in ``README.md`` / ``DATA_CARD.md``).

Kept separate from ``sources/restate.py`` on purpose: the scraper's job is
"fetch + extract whatever structured data the site embedded, unmodified";
this module's job is "turn that into consistent typed fields." If a parsing
rule turns out wrong, fixing it here and re-running against the existing
raw JSONL archive does not require re-fetching anything from the live site.

Every field that cannot be genuinely determined from the source is left
``None`` -- nothing here invents a plausible-looking value.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.features.feature_engineering import BUILDING_TYPE_ALIASES

_AREA_RE = re.compile(r"([\d]+(?:[.,]\d+)?)\s*м²")
_LIVING_AREA_RE = re.compile(r"жилая площадь\s*([\d]+(?:[.,]\d+)?)\s*кв\.?\s*м", re.I)
_FLOOR_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*эт")
_ROOMS_STUDIO_RE = re.compile(r"студ", re.I)
_ROOMS_N_RE = re.compile(r"(\d+)[-\s]*комн", re.I)
_OWNER_HINT_RE = re.compile(r"без\s+посредников|от\s+собственник", re.I)
_AGENCY_HINT_RE = re.compile(r"от\s+агентств|риэлтор|агентство\s+недвижимост", re.I)
_LISTING_ID_RE = re.compile(r"/base/(\d+)\.html")
# "тип дома: панельный" / "материал: газобетон" -- found live in the same
# search-results `description` text as area/floor (2026-08-25 sampling), not
# a detail-page-only field as originally assumed. Mapped through the
# project's existing BUILDING_TYPE_ALIASES (fixed ML-contract vocabulary);
# an unrecognised value (e.g. "газобетон") is kept as a free-text field
# rather than forced into a wrong bucket -- see `building_type_raw` below.
_BUILDING_TYPE_RE = re.compile(r"(?:тип дома|материал)\s*:\s*([а-яё]+)", re.I)
_CEILING_HEIGHT_RE = re.compile(r"высота потолков\s*([\d]+(?:[.,]\d+)?)\s*м", re.I)

REQUIRED_RAW_KEYS = ("url", "source", "raw_item")


def _to_float(text: str, pattern: re.Pattern) -> Optional[float]:
    if not text:
        return None
    m = pattern.search(text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


def _parse_rooms(text: str) -> Optional[int]:
    if not text:
        return None
    if _ROOMS_STUDIO_RE.search(text):
        return 0
    m = _ROOMS_N_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def _parse_floor(text: str) -> tuple[Optional[int], Optional[int]]:
    if not text:
        return None, None
    m = _FLOOR_RE.search(text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _infer_seller_type(text: str) -> Optional[str]:
    if not text:
        return None
    if _OWNER_HINT_RE.search(text):
        return "owner"
    if _AGENCY_HINT_RE.search(text):
        return "agency"
    return None


def _listing_id(url: str) -> Optional[str]:
    m = _LISTING_ID_RE.search(url or "")
    return m.group(1) if m else None


def _infer_building_type(text: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(building_type, building_type_raw)``.

    ``building_type`` is only set when the raw Russian word maps onto the
    project's fixed ML-contract vocabulary (``BUILDING_TYPE_ALIASES``);
    otherwise it stays ``None`` and the untranslated word is preserved in
    ``building_type_raw`` instead of being force-mapped to a wrong bucket.
    """
    if not text:
        return None, None
    m = _BUILDING_TYPE_RE.search(text)
    if not m:
        return None, None
    raw = m.group(1).lower()
    return BUILDING_TYPE_ALIASES.get(raw), raw


def _infer_ceiling_height(text: str) -> Optional[float]:
    return _to_float(text, _CEILING_HEIGHT_RE)


def normalize_restate_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize one raw Restate capture record (as produced by
    ``RestateScraper``) into the project's flat listing schema.

    Returns ``None`` (rather than a half-filled row) if the record is
    missing the bare minimum to be a listing at all (no URL, or the
    embedded item isn't actually a ``RealEstateListing``).
    """
    if not all(k in raw for k in REQUIRED_RAW_KEYS):
        return None
    item = raw["raw_item"]
    if not isinstance(item, dict) or item.get("@type") != "RealEstateListing":
        return None

    url = raw["url"]
    name = item.get("name", "") or ""
    description = item.get("description", "") or ""
    combined = f"{name} {description}"

    offers = item.get("offers") or {}
    price_raw = offers.get("price")
    try:
        price = float(price_raw) if price_raw is not None else None
    except (TypeError, ValueError):
        price = None

    total_area = _to_float(name, _AREA_RE) or _to_float(description, _AREA_RE)
    floor, floors_total = _parse_floor(combined)
    building_type, building_type_raw = _infer_building_type(description)

    price_per_sqm: Optional[float] = None
    if price is not None and total_area:
        price_per_sqm = round(price / total_area, 2)

    return {
        "source": raw.get("source"),
        "source_listing_id": _listing_id(url),
        "url": url,
        "canonical_url": url,
        "listing_type": "sale",
        # SOURCE field: which /choice/ category this listing was found
        # under (see sources/restate.py CategorySpec). Falls back to "flat"
        # only for raw records captured before this field existed (older
        # checkpoint files / the pre-2026-08-25 fixture) so re-normalizing
        # an existing raw archive doesn't turn every row into an unknown
        # property type.
        "property_type": raw.get("property_type") or "flat",
        "market_segment": raw.get(
            "market_segment"
        ),  # SOURCE field: from category, e.g. "new_build"
        "source_category": raw.get("source_category"),
        "title": name,
        "price": price,
        "currency": offers.get("priceCurrency"),
        "price_per_sqm": price_per_sqm,
        "total_area": total_area,
        "living_area": _to_float(description, _LIVING_AREA_RE),
        "kitchen_area": None,  # detail-page only field, not fetched -- see sources/restate.py
        "rooms": _parse_rooms(name),
        "bedrooms": None,
        "bathrooms": None,
        "floor": floor,
        "floors_total": floors_total,
        "year_built": None,  # detail-page only field, not fetched
        # DERIVED field: regex-mapped from the "тип дома: .../материал: ..."
        # phrase in `description` (a SOURCE field) onto the fixed ML-contract
        # vocabulary. building_type_raw keeps the untranslated word when the
        # mapping doesn't recognise it (e.g. "газобетон").
        "building_type": building_type,
        "building_type_raw": building_type_raw,
        "ceiling_height": _infer_ceiling_height(description),  # DERIVED from `description`
        "address": name,
        "city": None,  # filled by the caller from the scraper's city config
        "district": None,
        "region": None,
        "latitude": None,  # detail-page only field, not fetched
        "longitude": None,  # detail-page only field, not fetched
        "published_at": None,  # not present on the search-results JSON-LD
        "updated_at": None,
        "scraped_at": raw.get("fetched_at") or datetime.now(timezone.utc).isoformat(),
        "seller_type": _infer_seller_type(combined),
        "description": description,
        "image_count": 1 if item.get("image") else 0,
        "parser_version": raw.get("parser_version"),
        "source_page_url": raw.get("source_page_url"),
    }


def normalize_restate_records(
    raw_records: List[Dict[str, Any]],
    *,
    city_name: Optional[str] = None,
    region_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Normalize a batch, filling in ``city``/``region`` from the scrape
    context (these aren't reliably present in every item's free text)."""
    out = []
    for raw in raw_records:
        norm = normalize_restate_record(raw)
        if norm is None:
            continue
        if city_name is not None:
            norm["city"] = city_name
        if region_name is not None:
            norm["region"] = region_name
        out.append(norm)
    return out

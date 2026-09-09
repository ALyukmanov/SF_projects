"""
Tests for the restate.ru source adapter, against a sanitized real-markup
fixture -- see ``tests/data_collection/fixtures/restate_html/search_results_page.html``'s
own header comment for provenance (captured live 2026-08-24, trimmed to 3
items, no personal data). No network access; these never hit restate.ru.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data_collection.normalize import normalize_restate_record, normalize_restate_records
from src.data_collection.sources.restate import CategorySpec, RestateScraper

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "restate_html" / "search_results_page.html"


@pytest.fixture
def fixture_html() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


@pytest.fixture
def scraper() -> RestateScraper:
    return RestateScraper(city="moscow")


class TestExtractItemList:
    def test_extracts_all_real_items_from_fixture(self, scraper, fixture_html):
        records = scraper._extract_item_list(
            fixture_html, "https://msk.restate.ru/choice/moscow/1_rooms_flats_sale"
        )
        assert len(records) == 3

    def test_each_record_has_capture_metadata(self, scraper, fixture_html):
        records = scraper._extract_item_list(fixture_html, "https://example-page/")
        for r in records:
            assert r["source"] == "restate"
            assert r["source_page_url"] == "https://example-page/"
            assert r["fetched_at"]
            assert r["parser_version"]
            assert r["url"].startswith("https://www.restate.ru/base/")

    def test_raw_item_preserves_schema_org_fields(self, scraper, fixture_html):
        records = scraper._extract_item_list(fixture_html, "https://x/")
        item = records[0]["raw_item"]
        assert item["@type"] == "RealEstateListing"
        assert item["offers"]["price"] == 14500000
        assert item["offers"]["priceCurrency"] == "RUB"

    def test_ignores_faqpage_and_breadcrumb_blocks(self, scraper, fixture_html):
        # The fixture also contains a FAQPage JSON-LD block and a
        # BreadcrumbList node in the @graph -- neither should leak into the
        # extracted listings.
        records = scraper._extract_item_list(fixture_html, "https://x/")
        urls = {r["url"] for r in records}
        assert all("/base/" in u for u in urls)

    def test_empty_html_yields_no_records(self, scraper):
        assert scraper._extract_item_list("<html></html>", "https://x/") == []

    def test_malformed_json_ld_does_not_raise(self, scraper):
        html = '<script type="application/ld+json">{not valid json</script>'
        assert scraper._extract_item_list(html, "https://x/") == []


class TestNormalizeRestateRecord:
    def test_normalizes_real_captured_item(self, scraper, fixture_html):
        raw = scraper._extract_item_list(fixture_html, "https://x/")[0]
        norm = normalize_restate_record(raw)

        assert norm["source"] == "restate"
        assert norm["source_listing_id"] == "1442804407"
        assert norm["price"] == 14500000.0
        assert norm["currency"] == "RUB"
        assert norm["total_area"] == pytest.approx(38.70)
        assert norm["living_area"] == pytest.approx(20.00)
        assert norm["rooms"] == 1
        assert norm["floor"] == 2
        assert norm["floors_total"] == 14
        assert norm["price_per_sqm"] == pytest.approx(14500000 / 38.70, rel=1e-6)
        assert norm["listing_type"] == "sale"

    def test_fields_not_on_search_results_page_are_null_not_invented(self, scraper, fixture_html):
        raw = scraper._extract_item_list(fixture_html, "https://x/")[0]
        norm = normalize_restate_record(raw)
        for field in ("kitchen_area", "year_built", "latitude", "longitude", "published_at"):
            assert norm[field] is None

    def test_building_type_extracted_from_description(self, scraper, fixture_html):
        # Fixture item 0's description contains "тип дома: панельный" --
        # confirmed present in live search-results markup, not detail-page
        # only (see normalize.py's _BUILDING_TYPE_RE comment).
        raw = scraper._extract_item_list(fixture_html, "https://x/")[0]
        norm = normalize_restate_record(raw)
        assert norm["building_type"] == "panel"
        assert norm["building_type_raw"] == "панельный"

    def test_unmapped_building_material_kept_as_raw_only(self):
        raw = {
            "url": "https://www.restate.ru/base/998.html",
            "source": "restate",
            "raw_item": {
                "@type": "RealEstateListing",
                "name": "Коттедж, 200 м²",
                "description": "материал: газобетон, площадь участка 3 сот",
                "offers": {"price": 20000000, "priceCurrency": "RUB"},
            },
        }
        norm = normalize_restate_record(raw)
        assert norm["building_type"] is None
        assert norm["building_type_raw"] == "газобетон"

    def test_ceiling_height_extracted_when_present(self):
        raw = {
            "url": "https://www.restate.ru/base/997.html",
            "source": "restate",
            "raw_item": {
                "@type": "RealEstateListing",
                "name": "1-комн квартира, 43.20 м²",
                "description": "27 этаж из 28, высота потолков 2.85 м",
                "offers": {"price": 15000000, "priceCurrency": "RUB"},
            },
        }
        norm = normalize_restate_record(raw)
        assert norm["ceiling_height"] == pytest.approx(2.85)

    def test_property_type_defaults_to_flat_for_older_raw_records(self, scraper, fixture_html):
        # Raw records captured before source_category/property_type existed
        # (older checkpoints) must still normalize to a sane property_type,
        # not None/crash.
        raw = scraper._extract_item_list(fixture_html, "https://x/")[0]
        assert raw.get("property_type") is None  # fixture predates CategorySpec tagging
        norm = normalize_restate_record(raw)
        assert norm["property_type"] == "flat"

    def test_property_type_and_market_segment_come_from_category_spec(self):
        from src.data_collection.sources.restate import CategorySpec

        cat = CategorySpec("cottages_sale", property_type="house")
        raw = {
            "url": "https://www.restate.ru/base/996.html",
            "source": "restate",
            "source_category": cat.slug,
            "property_type": cat.property_type,
            "market_segment": cat.market_segment,
            "raw_item": {
                "@type": "RealEstateListing",
                "name": "Коттедж/дом, 225 м²",
                "description": "материал: газобетон",
                "offers": {"price": 33000000, "priceCurrency": "RUB"},
            },
        }
        norm = normalize_restate_record(raw)
        assert norm["property_type"] == "house"
        assert norm["source_category"] == "cottages_sale"
        assert norm["market_segment"] is None

    def test_studio_style_room_count(self):
        raw = {
            "url": "https://www.restate.ru/base/999.html",
            "source": "restate",
            "raw_item": {
                "@type": "RealEstateListing",
                "name": "Студия, Москва г., Тестовая ул., 25.00 м², 3/10 эт.",
                "description": "Продается студия, 25.00 кв. м, жилая площадь 12.00 кв. м",
                "offers": {"price": 10000000, "priceCurrency": "RUB"},
            },
        }
        norm = normalize_restate_record(raw)
        assert norm["rooms"] == 0

    def test_missing_raw_item_type_returns_none(self):
        assert (
            normalize_restate_record(
                {"url": "https://x/", "source": "restate", "raw_item": {"@type": "Other"}}
            )
            is None
        )

    def test_missing_required_key_returns_none(self):
        assert normalize_restate_record({"url": "https://x/"}) is None

    def test_batch_fills_city_and_region(self, scraper, fixture_html):
        raw_records = scraper._extract_item_list(fixture_html, "https://x/")
        normalized = normalize_restate_records(
            raw_records, city_name="Москва", region_name="Москва"
        )
        assert len(normalized) == 3
        assert all(r["city"] == "Москва" for r in normalized)


class TestCategoryTagging:
    """CategorySpec metadata must flow raw -> normalized untouched."""

    def test_extract_item_list_tags_records_with_category_metadata(self, scraper, fixture_html):
        cat = CategorySpec("cottages_sale", property_type="house")
        records = scraper._extract_item_list(fixture_html, "https://x/", category=cat)
        assert all(r["source_category"] == "cottages_sale" for r in records)
        assert all(r["property_type"] == "house" for r in records)
        assert all(r["market_segment"] is None for r in records)

    def test_extract_item_list_without_category_leaves_fields_none(self, scraper, fixture_html):
        records = scraper._extract_item_list(fixture_html, "https://x/")
        assert all(r["source_category"] is None for r in records)
        assert all(r["property_type"] is None for r in records)

    def test_new_erect_category_tags_new_build_market_segment(self, scraper, fixture_html):
        cat = CategorySpec("new_erect_flats_sale", property_type="flat", market_segment="new_build")
        records = scraper._extract_item_list(fixture_html, "https://x/", category=cat)
        assert all(r["market_segment"] == "new_build" for r in records)


class TestSecondCityAndSortOrders:
    def test_petersburg_city_uses_correct_subdomain_and_slug(self):
        scraper = RestateScraper(city="petersburg")
        url = scraper.category_url("1_rooms_flats_sale", page=1)
        assert url == "https://spb.restate.ru/choice/petersburg/1_rooms_flats_sale"

    def test_unknown_city_still_rejected(self):
        with pytest.raises(ValueError):
            RestateScraper(city="novosibirsk")

    def test_category_url_default_sort_omits_o_param(self, scraper):
        url = scraper.category_url("1_rooms_flats_sale", page=1, sort="default")
        assert "o=" not in url

    def test_category_url_price_asc_adds_o_param(self, scraper):
        url = scraper.category_url("1_rooms_flats_sale", page=1, sort="price_asc")
        assert url.endswith("o=1")

    def test_category_url_combines_page_and_sort(self, scraper):
        url = scraper.category_url("1_rooms_flats_sale", page=3, sort="price_desc")
        assert "page=3" in url and "o=2" in url

    def test_unknown_sort_order_rejected(self):
        with pytest.raises(ValueError):
            RestateScraper(city="moscow", sort_orders=["not_a_real_sort"])

    def test_default_sort_orders_is_just_default(self, scraper):
        assert scraper.sort_orders == ["default"]


class TestPaginationDepthCap:
    """restate.ru silently repeats page-1 content past its crawl-depth cap
    instead of returning an empty page or an error -- scrape() must detect
    an all-duplicate page and stop that sweep instead of looping until
    max_pages_per_category (see the module docstring / scrape()'s
    docstring for the live-probed evidence)."""

    def test_scrape_stops_sweep_when_page_returns_only_duplicates(self, monkeypatch):
        scraper = RestateScraper(
            city="moscow", categories=[CategorySpec("1_rooms_flats_sale", "flat")]
        )

        same_page_items = [
            {"url": f"https://www.restate.ru/base/{i}.html", "source": "restate", "raw_item": {}}
            for i in range(3)
        ]

        call_count = {"n": 0}

        def fake_fetch(url, category=None):
            call_count["n"] += 1
            return list(same_page_items), True, None

        monkeypatch.setattr(scraper, "_fetch_and_parse", fake_fetch)
        collected, stats = scraper.scrape(max_pages_per_category=10)

        # Page 1 yields 3 new items; every subsequent page repeats the same
        # 3 URLs, so the sweep must stop at page 2, not run all 10 pages.
        assert len(collected) == 3
        assert call_count["n"] == 2
        assert stats.pages_fetched == 2

    def test_scrape_sweeps_multiple_sort_orders_independently(self, monkeypatch):
        cat = CategorySpec("1_rooms_flats_sale", "flat")
        scraper = RestateScraper(
            city="moscow", categories=[cat], sort_orders=["default", "price_asc"]
        )

        def fake_fetch(url, category=None):
            # Each sort order's first page returns disjoint URLs so both
            # sweeps should contribute distinct items.
            sort_tag = "asc" if "o=1" in url else "default"
            items = [
                {
                    "url": f"https://www.restate.ru/base/{sort_tag}-{i}.html",
                    "source": "restate",
                    "raw_item": {},
                }
                for i in range(2)
            ]
            return items, True, None

        monkeypatch.setattr(scraper, "_fetch_and_parse", fake_fetch)
        collected, stats = scraper.scrape(max_pages_per_category=1)

        urls = {r["url"] for r in collected}
        assert (
            len(urls) == 4
        )  # 2 from "default" sweep + 2 from "price_asc" sweep, no cross-contamination

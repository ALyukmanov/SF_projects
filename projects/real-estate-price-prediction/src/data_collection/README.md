# src/data_collection

## Status: restate.ru is primary, CIAN is dormant

`sources/restate.py` is the active, working, real-data source. CLI:

```bash
python -m src.data_collection probe --source restate --city moscow
python -m src.data_collection probe --source cian          # re-checks CIAN's status
python -m src.data_collection scrape --source restate --city moscow --limit 50
python -m src.data_collection scrape --source restate --city moscow --max-pages-per-category 15
python -m src.data_collection scrape --source restate --city moscow --resume <run_dir>
python -m src.data_collection scrape --source restate --city moscow --incremental
```

`sources/restate.py` fetches `/choice/<city>/<category>` search-results
pages and parses the schema.org `RealEstateListing` `ItemList` embedded as
JSON-LD — no HTML-scraping heuristics needed for price/area/rooms/floor/
address/url. The bulk `scrape` step does **not** open individual detail
pages. `normalize.py` turns the raw captured JSON-LD items into the
project's flat listing schema, kept separate from fetching so a
parsing-rule fix doesn't require re-scraping.

`scripts/enrich_coordinates.py` is a separate, opt-in step that *does* fetch
each listing's detail page (`/base/<id>.html`, allowed by robots.txt) once,
slowly, to read the `data-lat`/`data-lng` map coordinates that the
search-results JSON-LD doesn't include. It caches every fetched page
(`data/raw/restate_details/coordinates.jsonl`) and can be stopped and
resumed. Output goes to `restate_<city>_listings_geo.csv` next to the
originals, which are left untouched.

The CIAN scrapers below are kept but no longer used: every recent request
returned a CAPTCHA page and 0 listings. `python -m src.data_collection probe
--source cian` re-checks this if CIAN's behaviour changes.

---

# CIAN scrapers (dormant — see status above)

Two scraper implementations for the same site: `cian_scraper.py` (requests +
BeautifulSoup, no browser) is the primary path; `cian_selenium_scraper.py`
(Selenium/Chrome) is a fallback for pages that only render listings via
client-side JavaScript, which the requests parser's three extraction
strategies (JSON-LD → embedded JS JSON → `<article>` HTML) cannot see.

## Shared, network-free parsing logic — `parsing_utils.py`

`parse_price`, `parse_rooms`, `parse_area`, `parse_floor_info`,
`random_user_agent`, `CITY_REGION_MAP`, `CITY_NAMES_RU`, `CIAN_BASE`, and
`PARSER_VERSION` used to be duplicated almost verbatim in both scraper
files — and had already silently drifted (the Selenium copy of
`_parse_rooms` required a trailing `-` in its fallback regex; the requests
copy didn't). Both scrapers now import these from `parsing_utils.py`, the
single source of truth. Bump `PARSER_VERSION` whenever parsing *behaviour*
changes (a regex gets stricter/looser, a new extraction strategy is added),
so listings collected under different parser versions can be told apart —
every extracted listing dict carries its own `parser_version`, `source_url`,
and `source_city` fields.

These functions take/return plain strings (or a `BeautifulSoup` object
already constructed from local HTML, in the case of the requests scraper's
`_parse_json_ld` / `_parse_embedded_json` / `_parse_article_tags` methods) —
**no network access is required to test them.** See
`tests/data_collection/test_parsing_utils.py` and
`tests/data_collection/test_cian_scraper_parsing.py`, which run against the
de-identified HTML fixtures in `tests/data_collection/fixtures/` (hand-written
to exercise specific edge cases — normal card, missing field, studio, price
with spaces/₽, floor info, empty results page, malformed/truncated HTML —
not copies of real listing markup).

## Extraction priority (requests scraper)

1. **JSON-LD** (`<script type="application/ld+json">`) — structured,
   schema.org-tagged data when present; most stable across CSS/markup
   changes.
2. **Embedded JS JSON** (`window.__initialData__` / `window._cianConfig` /
   inline `offersSerialized` blocks) — CIAN's own client-side hydration
   data; more fields than JSON-LD (floor, floors_total, building info) but
   more likely to change shape between site releases.
3. **`<article>` / `data-name="CardComponent"` HTML fallback** — last
   resort, most fragile to CSS/class-name churn; regex-based extraction of
   price/rooms/floor/address text from raw markup.

Each strategy is tried in order and the first one that yields any listings
wins — see `CianScraper.scrape_page()`.

## Architecture

Network I/O, parsing, and persistence are three separate, independently
testable modules, rather than interleaved inside `cian_scraper.py`:

| Concern | Module | Tested via |
|---|---|---|
| Parsing/normalization (pure functions) | `parsing_utils.py` | `tests/data_collection/test_parsing_utils.py` (no network) |
| HTML/JSON extraction strategies | `cian_scraper.py` (`_parse_json_ld`/`_parse_embedded_json`/`_parse_article_tags`) | `tests/test_cian_scraper_parsing.py`, fixture HTML, no network |
| Network fetching + retry/backoff policy | `network.py` (`fetch_with_retry`, `build_session`) | `tests/data_collection/test_network.py` — mocked `requests.Session`, injectable `sleep_fn`, no real waiting or network |
| Persistence / checkpointing / resume | `persistence.py` (`JsonlCheckpoint`) | `tests/data_collection/test_persistence.py`, `tests/test_cian_scraper_checkpoint.py` |

`CianScraper._get_with_retry()` is a thin adapter over
`network.fetch_with_retry()`; the retry/backoff logic lives in `network.py`
and is tested there directly.

## Resilience / politeness features (requests scraper)

- Configurable inter-page delay range (`delay_min`/`delay_max`, jittered via
  `random.uniform`) plus User-Agent rotation between pages.
- Exponential backoff with jitter on HTTP 429 and on timeouts/connection
  errors (`network.fetch_with_retry`, up to 3 attempts by default).
- Immediate give-up (no retry) on HTTP 403/404 — retrying those wastes
  requests and does not help.
- Per-run URL deduplication (`_deduplicate`) so a listing appearing on
  two overlapping result pages is only kept once — resume-aware
  (see "Checkpointing/resume" below): URLs already in a checkpoint file are
  treated as already-seen from the start of a new run.
- 30-second request timeout.

## Checkpointing / resume (`persistence.py`)

`CianScraper(..., checkpoint_path="data/interim/run.jsonl")` opts into
append-only JSONL checkpointing — each page's listings are written
immediately after parsing, and constructing a new `CianScraper` with the
same `checkpoint_path` pre-loads already-collected URLs into the
deduplication set, so a restarted run skips them rather than re-fetching the
same pages. Off by default (`checkpoint_path=None`). A flat JSONL file, not a
database.

## What this module does not do

- No CAPTCHA or anti-bot bypass. If CIAN serves a CAPTCHA page, the scraper
  gives up the same way it does on a 403/404.
- No mass/continuous scraping: `max_pages` defaults to 2, there is no
  scheduler or "keep scraping" mode.
- No proxy rotation beyond the polite delay/backoff above.

## CIAN status

A single check request to CIAN's first search-results page returns HTTP 200
with a CAPTCHA page (`<title>Captcha - база объявлений ЦИАН</title>`), not
listings, so none of the extraction strategies find anything. CIAN is not
usable as a data source without solving the CAPTCHA, which this project does
not do. restate.ru is used instead.

Note: `requirements.txt` pins `brotli` — CIAN responses use Brotli
compression, and without a decoder `requests` silently returns undecodable
bytes rather than raising.

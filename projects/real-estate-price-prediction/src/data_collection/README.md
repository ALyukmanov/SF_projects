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

The CIAN scrapers documented below are **not deleted, but not being
invested in further**: every live contact attempt so far, including from
this project's own network via CIAN's own robots.txt-allowed API path,
returned a CAPTCHA challenge page, 0 listings. `python -m
src.data_collection probe --source cian` re-checks this without writing new
code, if CIAN's posture ever changes.

---

# CIAN scrapers (dormant — see status above)

Two independent scraper implementations for the same site, kept side by side
on purpose (not because one is legacy): `cian_scraper.py` (requests +
BeautifulSoup, fast, no browser) is the primary path; `cian_selenium_scraper.py`
(Selenium/Chrome) is a fallback for pages that only render listings via
client-side JavaScript that the requests-based parser's three extraction
strategies (JSON-LD → embedded JS JSON → `<article>`/CardComponent HTML
fallback) cannot see.

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
`network.fetch_with_retry()` — kept as a method (not inlined) specifically so
existing tests that monkeypatch `scraper._get_with_retry` continue to work
unmodified; the actual retry/backoff decisions live in `network.py` and are
tested there directly, without needing `CianScraper` in the loop at all.

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
deduplication set, so a restarted run skips them rather than re-fetching
(and re-risking a rate-limit/block on) the same pages. Off by default
(`checkpoint_path=None`). Deliberately a flat JSONL file, not a
database/queue — see `persistence.py`'s module docstring for why, given
this project's data volumes.

## What this module intentionally does NOT do

- **No CAPTCHA or anti-bot bypass of any kind.** If CIAN serves a
  CAPTCHA/challenge page, the scraper has no special handling for it beyond
  the existing 403/404 give-up path — this is deliberate, per repository
  policy, and (see "Site-liveness" below) is not a hypothetical case.
- **No mass/continuous scraping built into this codebase.** `max_pages`
  defaults to a small number (2 for the requests scraper); there is no
  scheduler, queue, or "keep scraping forever" mode.
- No proxy rotation, no residential-IP infrastructure, no attempt to defeat
  rate limiting beyond the polite delay/backoff behaviour above.

## Site-liveness (CIAN)

Per this project's standing "sparing 1-2 page checks only" policy, a single
`GET` to `CianScraper(city="moskva").base_url` (a normal, first-page search
results URL, no pagination beyond page 1) was made to check current
reachability and markup compatibility. Result: **HTTP 200, but the response
body is a CAPTCHA challenge page** (`<title>Captcha - база объявлений
ЦИАН</title>`), not a search-results page — none of the three extraction
strategies (JSON-LD / embedded JS JSON / article-tag fallback) find anything
because there is nothing listing-shaped to find. Per policy, this was not
worked around, retried with different headers/proxies, or investigated
further — one request was enough to answer "is this currently viable
without solving a CAPTCHA," and the answer is no. **Practical conclusion:
CIAN cannot currently be used as a real-data source by this project without
crossing the explicit CAPTCHA-bypass boundary, which will not happen.**

A genuinely separate, unrelated bug was found and fixed alongside this
check: the scraper's request headers advertise `Accept-Encoding: gzip,
deflate, br`, but the `brotli` package was not a project dependency. CIAN's
response used Brotli compression; without a Brotli decoder installed,
`requests`/`urllib3` silently returned undecodable garbage instead of
either decompressing correctly or raising an error — which would have been
easy to misdiagnose as "the site changed its markup" or "we got blocked,"
rather than "we're missing a dependency for encoding we ourselves
requested." Fixed by adding `brotli==1.1.0` to `requirements.txt`.

Домклик, Авито Недвижимость, and Яндекс.Недвижимость are not checked here —
no scraper code exists for them in this project, and adding new site
support is out of scope for this module. See the project root
`DATA_CARD.md` / `README.md` roadmap for whether/how alternative,
legitimately-licensed data sources are pursued instead of scraping
additional commercial sites.

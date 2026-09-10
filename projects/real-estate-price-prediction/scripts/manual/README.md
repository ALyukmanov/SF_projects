# scripts/manual/ — old manual dev scripts

These predate the `tests/` suite and the
`run_feature_engineering_real.py` / `run_model_training_real.py` pipeline.
They are not part of the test suite or CI and are kept only for occasional
manual use.

| File | What it does | Superseded by |
|---|---|---|
| `collect_real_data.py` | One-off Selenium scrape of 5 cities | `src/data_collection/cian_selenium_scraper.py` + a real invocation of `python scripts/run_feature_engineering_real.py` once raw data exists |
| `collect_real_data_combo.py` | Larger one-off scrape (8 cities, target 5000+ listings) | same as above |
| `test_cbr_integration.py` | Manual smoke check of the CBR API integration | `tests/integration/test_cbr_api.py` (proper pytest, marked `integration`, network-only) |
| `test_cian_scraper.py` | Self-described `DEPRECATED` manual integration script | `tests/integration/test_cian_scraper.py` (network) and `tests/data_collection/test_cian_scraper_parsing.py` (offline, fixture-based) |

Do not run `collect_real_data.py`/`collect_real_data_combo.py` without a
specific, deliberate reason — they perform live scraping of CIAN.ru at a
scale well beyond the "1-2 pages, sparing" limit used elsewhere in this
project's development process. See `DATA_CARD.md` for the project's current
policy on data collection.

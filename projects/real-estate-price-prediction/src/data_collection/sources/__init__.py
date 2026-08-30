"""Per-source ingestion adapters.

Each module here targets exactly one real-estate site and returns raw
schema.org/JSON-LD (or equivalent) item dicts, close to what the site
itself emitted -- see ``src/data_collection/normalize.py`` for the
separate step that turns those raw records into the project's flat
normalized listing schema.
"""

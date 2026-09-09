"""
Build a small local POI table from OpenStreetMap for the geo features.

What it does, once, on your machine:

1. downloads the two regional OSM extracts that cover Moscow and Saint
   Petersburg — Central and Northwestern Federal District from Geofabrik
   (*not* the whole of Russia);
2. reads each one clipped to the city's bounding box and keeps only seven
   POI kinds that plausibly matter for flat prices:
   metro stations, railway stations, schools, kindergartens,
   hospitals/clinics, supermarkets, parks;
3. writes ``data/external/osm_poi.csv`` (category, name, lat, lon, city) plus
   ``data/external/osm_poi_manifest.json`` recording where the data came
   from, when, and the OSM tags used.

The raw ``.osm.pbf`` files stay under ``data/external/osm/`` and are
git-ignored. Re-running skips extracts that are already downloaded.

Data © OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright).

Downloads are ~900 MB (Central FD) + ~500 MB (Northwestern FD); reading them
clipped to the city bbox needs roughly 2-4 GB RAM and a few minutes each.
Smaller city-only extracts exist on BBBike
(https://download.bbbike.org/osm/bbbike/Moscow/Moscow.osm.pbf and
.../Sankt-Petersburg/Sankt-Petersburg.osm.pbf) — pass those via
``--moscow-url`` / ``--spb-url`` if the big downloads are inconvenient.

Requires ``pyrosm`` (``pip install pyrosm``). On Windows, if the pip build
fails, ``conda install -c conda-forge pyrosm`` is the reliable route.

Caching: a normal run does nothing if ``data/external/osm_poi.csv`` and its
manifest already exist — reading the ~1.5 GB of federal-district PBF takes
about 1.5 hours and the CSV is the only thing downstream code needs. Pass
``--rebuild`` to re-parse the PBF (extracts stay cached on disk), or
``--force`` to also re-download the extracts.

Usage:
    python scripts/prepare_osm_poi.py                  # no-op if osm_poi.csv exists
    python scripts/prepare_osm_poi.py --rebuild        # re-parse cached PBF -> osm_poi.csv
    python scripts/prepare_osm_poi.py --force          # re-download extracts, then rebuild
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("prepare_osm_poi")

EXTERNAL_DIR = _PROJECT_ROOT / "data" / "external"
PBF_DIR = EXTERNAL_DIR / "osm"
POI_CSV = EXTERNAL_DIR / "osm_poi.csv"
MANIFEST_JSON = EXTERNAL_DIR / "osm_poi_manifest.json"

# Regional extracts from Geofabrik. Override with --moscow-url / --spb-url
# to use a smaller source (e.g. BBBike city extracts — see the docstring).
EXTRACTS = {
    "Москва": "https://download.geofabrik.de/russia/central-fed-district-latest.osm.pbf",
    "Санкт-Петербург": "https://download.geofabrik.de/russia/northwestern-fed-district-latest.osm.pbf",
}

# Bounding boxes [min_lon, min_lat, max_lon, max_lat] used to clip each
# regional extract down to the city (keeps memory and POI counts sane).
CITY_BBOX = {
    "Москва": [36.7, 55.4, 38.0, 56.05],
    "Санкт-Петербург": [29.4, 59.7, 30.8, 60.15],
}

# OSM tags we ask pyrosm for, then sort into our seven categories.
OSM_TAG_FILTER = {
    "amenity": ["school", "kindergarten", "hospital", "clinic"],
    "shop": ["supermarket"],
    "leisure": ["park"],
    "railway": ["station", "halt", "subway_entrance"],
    "station": ["subway"],
}


def _categorise(row: pd.Series) -> str | None:
    railway = row.get("railway")
    if row.get("station") == "subway" or railway == "subway_entrance":
        return "metro"
    if railway in ("station", "halt"):
        return "rail_station"
    amenity = row.get("amenity")
    if amenity == "school":
        return "school"
    if amenity == "kindergarten":
        return "kindergarten"
    if amenity in ("hospital", "clinic"):
        return "hospital"
    if row.get("shop") == "supermarket":
        return "supermarket"
    if row.get("leisure") == "park":
        return "park"
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        logger.info("Already downloaded: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest.name)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        next_mark = 20 * 1024 * 1024
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done >= next_mark:
                    pct = f" ({100 * done / total:.0f}%)" if total else ""
                    logger.info("  %.0f MB%s", done / 1e6, pct)
                    next_mark += 20 * 1024 * 1024
    logger.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def _extract_pois(pbf_path: Path, city_name: str) -> pd.DataFrame:
    from pyrosm import OSM  # imported here so --help works without pyrosm installed

    logger.info("Reading POIs from %s (clipped to %s) ...", pbf_path.name, city_name)
    osm = OSM(str(pbf_path), bounding_box=CITY_BBOX[city_name])
    pois = osm.get_pois(custom_filter=OSM_TAG_FILTER)
    if pois is None or pois.empty:
        logger.warning("No POIs returned for %s", city_name)
        return pd.DataFrame(columns=["category", "name", "lat", "lon", "city"])

    for col in ("railway", "station", "amenity", "shop", "leisure", "name"):
        if col not in pois.columns:
            pois[col] = None

    pois["category"] = pois.apply(_categorise, axis=1)
    pois = pois[pois["category"].notna()].copy()

    points = pois.geometry.representative_point()
    pois["lat"] = points.y.round(6)
    pois["lon"] = points.x.round(6)
    pois["city"] = city_name

    out = pois[["category", "name", "lat", "lon", "city"]].reset_index(drop=True)
    out = out.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["category", "lat", "lon"])
    logger.info("  %s: %d POIs %s", city_name, len(out), out["category"].value_counts().to_dict())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the OSM extracts (implies --rebuild).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-parse the cached PBF extracts into osm_poi.csv even if it already exists.",
    )
    parser.add_argument("--moscow-url", default=EXTRACTS["Москва"])
    parser.add_argument("--spb-url", default=EXTRACTS["Санкт-Петербург"])
    args = parser.parse_args()

    if POI_CSV.is_file() and MANIFEST_JSON.is_file() and not (args.rebuild or args.force):
        existing = pd.read_csv(POI_CSV)
        print(
            f"{POI_CSV.relative_to(_PROJECT_ROOT)} already exists "
            f"({len(existing)} rows) — nothing to do.\n"
            "Pass --rebuild to re-parse the cached PBF, or --force to re-download first."
        )
        return 0

    urls = {"Москва": args.moscow_url, "Санкт-Петербург": args.spb_url}

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    manifest_sources = []

    for city_name, url in urls.items():
        pbf_path = PBF_DIR / url.rsplit("/", 1)[-1].replace("-latest", "")
        try:
            _download(url, pbf_path, force=args.force)
        except Exception as exc:  # noqa: BLE001 — show the URL so the user can fix it
            print(
                f"\nDownload failed for {city_name}:\n  {url}\n  {exc}\n\n"
                "Check the URL (BBBike area names change occasionally) or pass\n"
                "--moscow-url / --spb-url with a working link, e.g. a Geofabrik\n"
                "regional extract.",
                file=sys.stderr,
            )
            return 1

        frames.append(_extract_pois(pbf_path, city_name))
        manifest_sources.append(
            {
                "city": city_name,
                "url": url,
                "file": pbf_path.name,
                "sha256": _sha256(pbf_path),
                "size_bytes": pbf_path.stat().st_size,
                "downloaded_at": datetime.fromtimestamp(
                    pbf_path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )

    poi = pd.concat(frames, ignore_index=True)
    poi.to_csv(POI_CSV, index=False, encoding="utf-8-sig")

    MANIFEST_JSON.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "attribution": "© OpenStreetMap contributors, ODbL — https://www.openstreetmap.org/copyright",
                "osm_tag_filter": OSM_TAG_FILTER,
                "categories": sorted(poi["category"].unique().tolist()),
                "row_count": int(len(poi)),
                "counts_by_category": poi["category"].value_counts().to_dict(),
                "sources": manifest_sources,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("OSM POI SUMMARY")
    print("=" * 60)
    print(poi.groupby(["city", "category"]).size().to_string())
    print(f"\nPOI table : {POI_CSV}  ({len(poi)} rows)")
    print(f"Manifest  : {MANIFEST_JSON}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

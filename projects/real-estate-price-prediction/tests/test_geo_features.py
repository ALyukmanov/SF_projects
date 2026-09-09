"""
Tests for src/features/geo_features.py — the OSM geo-feature builder.

Uses a tiny hand-built POI table (not the real 27k-row osm_poi.csv) so the
distances/counts are checkable by hand, and so the test does not depend on
data/external/ being present.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.geo_features import (  # noqa: E402
    GEO_FEATURE_COLUMNS,
    NEAREST_SENTINEL_M,
    GeoFeatureBuilder,
    _dedupe_points,
)


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


@pytest.fixture
def poi_csv(tmp_path: Path) -> Path:
    """A minimal POI table: a couple of POI per category in each city, with
    coordinates far enough apart that nothing dedupes by accident."""
    rows = [
        # Moscow — around 55.75 / 37.62
        ("metro", "M-A", 55.7500, 37.6200, "Москва"),
        ("metro", "M-B", 55.7600, 37.6400, "Москва"),
        ("school", "S-A", 55.7520, 37.6210, "Москва"),
        ("supermarket", "Sup-A", 55.7505, 37.6205, "Москва"),
        ("supermarket", "Sup-B", 55.7560, 37.6320, "Москва"),  # ~800 m from Sup-A
        ("kindergarten", "K-A", 55.7510, 37.6220, "Москва"),
        ("hospital", "H-A", 55.7700, 37.6600, "Москва"),
        ("park", "P-A", 55.7480, 37.6100, "Москва"),
        ("rail_station", "R-A", 55.8000, 37.7000, "Москва"),
        # SPb — around 59.93 / 30.36; deliberately different values
        ("metro", "SM-A", 59.9300, 30.3600, "Санкт-Петербург"),
        ("school", "SS-A", 59.9310, 30.3610, "Санкт-Петербург"),
        ("supermarket", "SSup-A", 59.9305, 30.3605, "Санкт-Петербург"),
        ("kindergarten", "SK-A", 59.9320, 30.3620, "Санкт-Петербург"),
        ("hospital", "SH-A", 59.9400, 30.3700, "Санкт-Петербург"),
        ("park", "SP-A", 59.9280, 30.3550, "Санкт-Петербург"),
        ("rail_station", "SR-A", 59.9500, 30.4000, "Санкт-Петербург"),
    ]
    df = pd.DataFrame(rows, columns=["category", "name", "lat", "lon", "city"])
    path = tmp_path / "osm_poi.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def builder(poi_csv: Path) -> GeoFeatureBuilder:
    return GeoFeatureBuilder(poi_csv=poi_csv)


def test_builder_available(builder: GeoFeatureBuilder):
    assert builder.available
    assert set(builder.poi_summary()) == {"moscow", "spb"}


def test_all_geo_columns_present_after_transform(builder: GeoFeatureBuilder):
    df = pd.DataFrame([{"city": "Москва", "latitude": 55.7500, "longitude": 37.6200}])
    out = builder.transform(df)
    for col in GEO_FEATURE_COLUMNS:
        assert col in out.columns


def test_nearest_distance_matches_hand_haversine(builder: GeoFeatureBuilder):
    """A listing right next to Moscow metro M-A: nearest_metro_station_distance_m
    must equal the hand-computed haversine distance to M-A (the closer of the
    two Moscow metro POI)."""
    lat, lon = 55.7502, 37.6203
    out = builder.transform(
        pd.DataFrame([{"city": "Москва", "latitude": lat, "longitude": lon}])
    )
    expected = _haversine_m(lat, lon, 55.7500, 37.6200)
    assert out.loc[0, "nearest_metro_station_distance_m"] == pytest.approx(expected, abs=1.0)
    assert out.loc[0, "has_coordinates"] == 1


def test_counts_within_radius(builder: GeoFeatureBuilder):
    """From the Moscow anchor both supermarkets are within 1 km; only the
    nearer one is within 500 m."""
    out = builder.transform(
        pd.DataFrame([{"city": "Москва", "latitude": 55.7505, "longitude": 37.6205}])
    )
    assert out.loc[0, "supermarkets_1000m"] == 2
    assert out.loc[0, "supermarkets_500m"] == 1


def test_no_coordinates_gets_sentinel_and_flag(builder: GeoFeatureBuilder):
    out = builder.transform(
        pd.DataFrame([{"city": "Москва", "latitude": np.nan, "longitude": np.nan}])
    )
    assert out.loc[0, "has_coordinates"] == 0
    assert out.loc[0, "nearest_metro_station_distance_m"] == NEAREST_SENTINEL_M
    assert out.loc[0, "supermarkets_1000m"] == 0


def test_row_count_and_no_nan_preserved(builder: GeoFeatureBuilder):
    df = pd.DataFrame(
        [
            {"city": "Москва", "latitude": 55.75, "longitude": 37.62},
            {"city": "Москва", "latitude": None, "longitude": None},
            {"city": "Санкт-Петербург", "latitude": 59.93, "longitude": 30.36},
        ]
    )
    out = builder.transform(df)
    assert len(out) == len(df)
    assert out[GEO_FEATURE_COLUMNS].isna().sum().sum() == 0


def test_cities_do_not_mix(builder: GeoFeatureBuilder):
    """A Moscow listing must be scored against Moscow POI only. If the SPb
    POI leaked in, the nearest-metro distance for a point near the SPb metro
    coordinates but tagged 'Москва' would be tiny; instead it should be the
    (large) distance Moscow<->SPb."""
    out = builder.transform(
        pd.DataFrame([{"city": "Москва", "latitude": 59.9300, "longitude": 30.3600}])
    )
    # Moscow's nearest metro POI is ~600 km away from the SPb coordinates.
    assert out.loc[0, "nearest_metro_station_distance_m"] > 500_000
    assert out.loc[0, "supermarkets_1000m"] == 0


def test_unsupported_city_gets_sentinel(builder: GeoFeatureBuilder):
    out = builder.transform(
        pd.DataFrame([{"city": "Казань", "latitude": 55.79, "longitude": 49.12}])
    )
    assert out.loc[0, "has_coordinates"] == 0
    assert out.loc[0, "nearest_school_distance_m"] == NEAREST_SENTINEL_M


def test_dedupe_collapses_near_points():
    # three points within a few metres + one far away -> 2 survivors
    coords = np.radians(
        np.array(
            [
                [55.7500, 37.6200],
                [55.75001, 37.62001],
                [55.75002, 37.62000],
                [55.8000, 37.7000],
            ]
        )
    )
    kept = _dedupe_points(coords, radius_m=100.0)
    assert len(kept) == 2


def test_missing_poi_file_is_not_fatal(tmp_path: Path):
    builder = GeoFeatureBuilder(poi_csv=tmp_path / "does_not_exist.csv")
    assert not builder.available
    out = builder.transform(
        pd.DataFrame([{"city": "Москва", "latitude": 55.75, "longitude": 37.62}])
    )
    # every row falls back to sentinel/zero, nothing raised, columns present
    assert out.loc[0, "has_coordinates"] == 0
    assert list(out.columns[-len(GEO_FEATURE_COLUMNS):]) != []  # geo cols added


def test_geo_columns_registered_as_optional_model_features():
    """feature_engineering must know these column names, otherwise a
    geo-enriched training frame would silently drop them."""
    from src.features.feature_engineering import _OPTIONAL_NUMERIC_FEATURES

    for col in GEO_FEATURE_COLUMNS:
        assert col in _OPTIONAL_NUMERIC_FEATURES


class TestPredictorGeoIntegration:
    """The live inference path (Predictor._build_feature_vector) must only
    emit geo columns when the loaded model expects them, and when it does
    they must match GeoFeatureBuilder run on the same coordinates."""

    def _predictor(self, poi_csv: Path, feature_names):
        from src.inference.predictor import Predictor

        p = Predictor(model_path="models")
        p._feature_names = list(feature_names)
        p._geo_builder = GeoFeatureBuilder(poi_csv=poi_csv)
        return p

    def test_no_geo_emitted_when_model_has_no_geo_features(self, poi_csv: Path):
        p = self._predictor(poi_csv, ["rooms", "total_area", "city_moskva"])
        vec = p._build_feature_vector(
            {"rooms": 2, "total_area": 55.0, "floor": 4, "floors_total": 10, "city": "Москва"}
        )
        assert not any(c in vec for c in GEO_FEATURE_COLUMNS)

    def test_geo_emitted_and_matches_builder(self, poi_csv: Path):
        p = self._predictor(
            poi_csv, ["rooms", "total_area", *GEO_FEATURE_COLUMNS]
        )
        req = {
            "rooms": 2,
            "total_area": 55.0,
            "floor": 4,
            "floors_total": 10,
            "city": "Москва",
            "latitude": 55.7505,
            "longitude": 37.6205,
        }
        vec = p._build_feature_vector(req)
        ref = GeoFeatureBuilder(poi_csv=poi_csv).transform(
            pd.DataFrame([{"city": "Москва", "latitude": 55.7505, "longitude": 37.6205}])
        ).iloc[0]
        for col in GEO_FEATURE_COLUMNS:
            assert vec[col] == pytest.approx(float(ref[col]))

    def test_geo_request_without_coordinates_uses_sentinel(self, poi_csv: Path):
        p = self._predictor(poi_csv, ["rooms", *GEO_FEATURE_COLUMNS])
        vec = p._build_feature_vector(
            {"rooms": 2, "total_area": 55.0, "floor": 4, "floors_total": 10, "city": "Москва"}
        )
        assert vec["has_coordinates"] == 0
        assert vec["nearest_metro_station_distance_m"] == NEAREST_SENTINEL_M


class TestPredictorOsmDependency:
    """A pre-geo model must not require OSM data; a geo model must fail with
    a clear message (not mysteriously mid-predict) when osm_poi.csv is gone."""

    def _fake_geo_artifact(self, tmp_path: Path, feature_names, poi_csv: Path | None):
        import joblib
        from sklearn.dummy import DummyRegressor

        model = DummyRegressor(strategy="constant", constant=16.0).fit(
            [[0.0]] * len(feature_names), [16.0] * len(feature_names)
        )
        art = {
            "model": model,
            "scaler": None,
            "model_type": "dummy",
            "trained_at": "2026-09-09T00:00:00",
            "metrics": {"r2": 0.5},
            "feature_names": list(feature_names),
        }
        mdir = tmp_path / "models"
        mdir.mkdir()
        joblib.dump(art, mdir / "m.pkl")
        (mdir / "current_model.json").write_text('{"filename": "m.pkl"}', encoding="utf-8")
        return mdir

    def test_pre_geo_model_loads_without_osm(self, tmp_path: Path, monkeypatch):
        from src.inference import predictor as predmod

        monkeypatch.setattr(
            predmod, "GeoFeatureBuilder", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
        )
        mdir = self._fake_geo_artifact(tmp_path, ["rooms", "total_area"], None)
        p = predmod.Predictor(model_path=str(mdir))
        assert p.load() is True
        assert p._expects_geo_features is False
        assert p._geo_unavailable is False

    def test_geo_model_without_osm_gives_clear_error(self, tmp_path: Path, monkeypatch):
        from src.features import geo_features as gf
        from src.inference import predictor as predmod

        # a GeoFeatureBuilder that never finds a POI table
        monkeypatch.setattr(
            predmod, "GeoFeatureBuilder", lambda *a, **k: gf.GeoFeatureBuilder(poi_csv=tmp_path / "nope.csv")
        )
        mdir = self._fake_geo_artifact(
            tmp_path, ["rooms", *GEO_FEATURE_COLUMNS], poi_csv=None
        )
        p = predmod.Predictor(model_path=str(mdir))
        assert p.load() is True
        assert p._geo_unavailable is True
        assert p.model_info["geo_enabled"] is True
        assert p.model_info["geo_poi_available"] is False
        with pytest.raises(RuntimeError, match="osm_poi.csv"):
            p.predict({"rooms": 2, "total_area": 55.0, "floor": 4, "floors_total": 10, "city": "Москва"})

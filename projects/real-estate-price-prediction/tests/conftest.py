"""
Shared pytest fixtures for the Real Estate Price Prediction test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Return a DataFrame with 20 rows of realistic raw real-estate data."""
    rng = np.random.default_rng(0)

    cities = [
        "москва",
        "москва",
        "санкт-петербург",
        "санкт-петербург",
        "екатеринбург",
        "екатеринбург",
        "новосибирск",
        "казань",
        "нижний новгород",
        "самара",
        "краснодар",
        "москва",
        "санкт-петербург",
        "екатеринбург",
        "казань",
        "новосибирск",
        "самара",
        "краснодар",
        "нижний новгород",
        "москва",
    ]

    base_ppm = {
        "москва": 280_000,
        "санкт-петербург": 180_000,
        "екатеринбург": 90_000,
        "новосибирск": 85_000,
        "казань": 80_000,
        "нижний новгород": 75_000,
        "самара": 72_000,
        "краснодар": 78_000,
    }

    n = 20
    rooms = rng.integers(1, 5, size=n).tolist()
    total_area = (rng.uniform(30, 120, size=n)).round(1).tolist()
    floors_total = rng.integers(5, 20, size=n).tolist()
    floor = [int(rng.integers(1, ft + 1)) for ft in floors_total]
    year_built = rng.integers(1970, 2023, size=n).tolist()
    building_types = ["panel", "brick", "monolith", "block"]
    building_type = rng.choice(building_types, size=n).tolist()

    price = [int(base_ppm[c] * a * rng.uniform(0.85, 1.15)) for c, a in zip(cities, total_area)]

    days_offset = rng.integers(0, 600, size=n)
    base_date = pd.Timestamp("2023-01-01")
    date_published = [str(base_date + pd.Timedelta(days=int(d)))[:10] for d in days_offset]
    scraped_at = date_published[:]

    urls = [f"https://cian.ru/sale/flat/{200_000 + i}/" for i in range(n)]
    addresses = [f"ул. Ленина, д. {rng.integers(1, 100)}" for _ in range(n)]

    return pd.DataFrame(
        {
            "price": price,
            "rooms": rooms,
            "total_area": total_area,
            "floor": floor,
            "floors_total": floors_total,
            "address": addresses,
            "city": cities,
            "year_built": year_built,
            "building_type": building_type,
            "url": urls,
            "scraped_at": scraped_at,
            "date_published": date_published,
        }
    )


@pytest.fixture
def sample_cleaned_df(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return an already-cleaned DataFrame (runs DataCleaner on sample_raw_df)."""
    from src.preprocessing.cleaner import DataCleaner

    cleaner = DataCleaner()
    return cleaner.clean(sample_raw_df)


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for model artefacts."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

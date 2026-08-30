"""
Integration test for CBR API access via EconomicFeatureEngineer.

IMPORTANT: This test makes real HTTP requests to cbr.ru.
Run explicitly with:
    pytest tests/integration/test_cbr_api.py -v

Do NOT include in the standard CI suite.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

pytestmark = pytest.mark.integration


@pytest.mark.integration
def test_cbr_api_vs_fallback():
    """Verify that live CBR data and fallback data produce the same column structure."""
    from src.preprocessing.economic_features import EconomicFeatureEngineer

    dates = pd.date_range(start="2024-01-01", periods=5, freq="60D")
    df = pd.DataFrame(
        {
            "date_published": dates,
            "city": ["Москва"] * 5,
            "price": [10_000_000] * 5,
        }
    )

    eng_fallback = EconomicFeatureEngineer(use_api=False)
    result = eng_fallback.add_economic_features(df)

    for col in ("key_rate", "usd_rate", "inflation_rate", "rate_change_6m"):
        assert col in result.columns, f"Column '{col}' missing from result"
        assert result[col].notna().all(), f"Column '{col}' has NaN values"

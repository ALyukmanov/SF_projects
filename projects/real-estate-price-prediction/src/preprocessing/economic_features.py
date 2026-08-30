"""
Economic feature engineering module.

Fetches and attaches macroeconomic indicators (CBR key rate, USD/RUB
exchange rate, inflation) to a listings DataFrame.  Falls back to
hard-coded historical data when the CBR API is unavailable.
"""

from __future__ import annotations

import datetime
from typing import Dict, Optional

import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Fallback historical data (year → value)
# ---------------------------------------------------------------------------

_KEY_RATES: Dict[int, float] = {
    2018: 7.75,
    2019: 6.25,
    2020: 4.25,
    2021: 8.50,
    2022: 7.50,
    2023: 16.00,
    2024: 21.00,
    2025: 21.00,
}

_USD_RATES: Dict[int, float] = {
    2018: 62.7,
    2019: 64.7,
    2020: 73.5,
    2021: 73.0,
    2022: 68.5,
    2023: 88.5,
    2024: 100.0,
    2025: 90.0,
}

_INFLATION_RATES: Dict[int, float] = {
    2018: 4.3,
    2019: 3.0,
    2020: 4.9,
    2021: 8.4,
    2022: 11.9,
    2023: 7.4,
    2024: 8.5,
    2025: 8.5,
}

# CBR REST API endpoints
_CBR_KEY_RATE_URL = (
    "https://www.cbr.ru/hd_base/KeyRate/?UniDbQuery.Posted=True"
    "&UniDbQuery.From=01.01.{year}&UniDbQuery.To=31.12.{year}"
)
_CBR_CURRENCY_URL = (
    "https://www.cbr.ru/scripts/XML_dynamic.asp"
    "?date_req1=01/01/{year}&date_req2=31/12/{year}&VAL_NM_RQ=R01235"
)


# ---------------------------------------------------------------------------
# Helper: live data fetchers
# ---------------------------------------------------------------------------


def _fetch_cbr_key_rate(year: int, timeout: int = 10) -> Optional[float]:
    """Attempt to fetch the end-of-year key rate from CBR HTML table.

    Returns the last recorded rate for *year*, or ``None`` on failure.
    """
    try:
        url = _CBR_KEY_RATE_URL.format(year=year)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        # CBR HTML table — look for the last numeric value in the table
        import re

        matches = re.findall(r"(\d+[.,]\d+)\s*%", response.text)
        if matches:
            return float(matches[-1].replace(",", "."))
    except Exception as exc:
        logger.debug("CBR key rate fetch failed for %d: %s", year, exc)
    return None


def _fetch_cbr_usd_rate(year: int, timeout: int = 10) -> Optional[float]:
    """Attempt to fetch the average annual USD/RUB rate from the CBR XML API.

    Returns the average rate, or ``None`` on failure.
    """
    try:
        url = _CBR_CURRENCY_URL.format(year=year)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        import xml.etree.ElementTree as ET

        root = ET.fromstring(response.text)
        values = []
        for record in root.findall("Record"):
            val_el = record.find("Value")
            if val_el is not None and val_el.text:
                try:
                    values.append(float(val_el.text.replace(",", ".")))
                except ValueError:
                    continue
        if values:
            return round(sum(values) / len(values), 2)
    except Exception as exc:
        logger.debug("CBR USD rate fetch failed for %d: %s", year, exc)
    return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EconomicFeatureEngineer:
    """Adds macroeconomic indicator columns to a real-estate DataFrame.

    Args:
        use_api: If ``True``, attempt to fetch live data from the CBR API
                 before falling back to cached historical values.
    """

    def __init__(self, use_api: bool = True) -> None:
        self.use_api = use_api
        self._key_rate_cache: Dict[int, float] = {}
        self._usd_rate_cache: Dict[int, float] = {}
        logger.info("EconomicFeatureEngineer initialised (use_api=%s)", use_api)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach economic indicator columns to *df* and return the result.

        New columns added:
        - ``key_rate``       — CBR key rate (%) for the listing year
        - ``usd_rate``       — Average USD/RUB rate for the listing year
        - ``inflation_rate`` — Annual CPI inflation rate (%) for the listing year
        - ``rate_change_6m`` — Approximate 6-month key-rate change

        Args:
            df: Input DataFrame.  May contain a ``date_published`` column
                (str or datetime) from which the year is extracted.
                If absent, the current year is used for all rows.

        Returns:
            DataFrame with four new columns appended.
        """
        df = df.copy()

        # --- Extract year from date_published ----------------------------
        if "date_published" in df.columns:
            df["_year"] = self._extract_year(df["date_published"])
        else:
            current_year = datetime.datetime.now().year
            logger.info("No 'date_published' column found. Using current year (%d).", current_year)
            df["_year"] = current_year

        # --- Map economic indicators per year ----------------------------
        unique_years = df["_year"].dropna().unique().astype(int)
        logger.info("Fetching economic data for years: %s", sorted(unique_years))

        key_rate_map: Dict[int, float] = {}
        usd_rate_map: Dict[int, float] = {}
        inflation_map: Dict[int, float] = {}

        for year in unique_years:
            key_rate_map[year] = self._get_key_rate(year)
            usd_rate_map[year] = self._get_usd_rate(year)
            inflation_map[year] = _INFLATION_RATES.get(year, _INFLATION_RATES[2025])

        df["key_rate"] = df["_year"].map(key_rate_map)
        df["usd_rate"] = df["_year"].map(usd_rate_map)
        df["inflation_rate"] = df["_year"].map(inflation_map)

        # --- 6-month rate change (approx: diff between year and year-1) --
        df["rate_change_6m"] = df["_year"].apply(
            lambda y: self._rate_change_6m(int(y)) if pd.notna(y) else 0.0
        )

        df.drop(columns=["_year"], inplace=True)

        logger.info(
            "Economic features added. Columns: key_rate, usd_rate, inflation_rate, rate_change_6m"
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_year(self, series: pd.Series) -> pd.Series:
        """Parse a mixed date Series and return the year as int."""

        def to_year(val):
            if pd.isna(val):
                return datetime.datetime.now().year
            if isinstance(val, (int, float)):
                # Assume it's already a year if 2000-2030
                if 2000 <= int(val) <= 2030:
                    return int(val)
            if isinstance(val, (datetime.datetime, datetime.date)):
                return val.year
            # Try to parse string
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d.%m.%Y"):
                try:
                    return datetime.datetime.strptime(str(val), fmt).year
                except ValueError:
                    continue
            # Last resort: extract 4-digit year
            import re

            match = re.search(r"(20\d{2})", str(val))
            if match:
                return int(match.group(1))
            return datetime.datetime.now().year

        return series.apply(to_year)

    def _get_key_rate(self, year: int) -> float:
        if year in self._key_rate_cache:
            return self._key_rate_cache[year]

        value: Optional[float] = None
        if self.use_api:
            value = _fetch_cbr_key_rate(year)

        if value is None:
            value = _KEY_RATES.get(year, _KEY_RATES[2025])
            logger.debug("Using fallback key rate for %d: %.2f%%", year, value)
        else:
            logger.debug("Live key rate for %d: %.2f%%", year, value)

        self._key_rate_cache[year] = value
        return value

    def _get_usd_rate(self, year: int) -> float:
        if year in self._usd_rate_cache:
            return self._usd_rate_cache[year]

        value: Optional[float] = None
        if self.use_api:
            value = _fetch_cbr_usd_rate(year)

        if value is None:
            value = _USD_RATES.get(year, _USD_RATES[2025])
            logger.debug("Using fallback USD rate for %d: %.2f", year, value)
        else:
            logger.debug("Live USD rate for %d: %.2f", year, value)

        self._usd_rate_cache[year] = value
        return value

    def _rate_change_6m(self, year: int) -> float:
        """Approximate 6-month key rate change as half the annual change."""
        current = self._get_key_rate(year)
        prev = self._get_key_rate(year - 1)
        return round((current - prev) / 2, 4)

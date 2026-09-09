"""
Data schema / contract validation for real-estate listing DataFrames.

Why a hand-rolled validator instead of Pandera?
------------------------------------------------
Pandera was evaluated but not
added as a dependency in this pass: the validation rules needed here are a
short, fixed list of row-level numeric-range and referential checks with
custom, readable Russian-friendly error messages, which a ~150-line module
with no new third-party dependency covers adequately for a first-year
project's scope. This is a documented, reversible choice — swapping in
Pandera later is straightforward if the schema grows materially (e.g. many
more columns, dtype coercion pipelines). See DATA_CARD.md.

Usage
-----
    from src.data.schema import validate_listings_df, SchemaValidationError

    try:
        report = validate_listings_df(df)
    except SchemaValidationError as exc:
        print(exc.report)  # list of human-readable problems
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import pandas as pd

from src.features.feature_engineering import BUILDING_TYPE_ALIASES, BUILDING_TYPE_SLUGS, CITY_SLUGS

REQUIRED_COLUMNS: Sequence[str] = ("price", "total_area", "floor", "floors_total", "city")

_MIN_PRICE = 100_000  # RUB — anything lower is almost certainly a data error, not a real listing
_MAX_PRICE = 500_000_000
_MIN_AREA = 5.0
_MAX_AREA = 1000.0
_MIN_FLOOR = 1
_MAX_FLOOR = 150
_MIN_YEAR_BUILT = 1850
_MAX_ROOMS = 15
_KNOWN_CITIES = set(CITY_SLUGS.keys())
_KNOWN_BUILDING_TYPES = set(BUILDING_TYPE_SLUGS.keys()) | set(BUILDING_TYPE_ALIASES.keys())
_PRICE_PER_SQM_MIN = 5_000.0  # RUB/sqm — implausibly cheap even for the cheapest regional market
_PRICE_PER_SQM_MAX = 3_000_000.0  # RUB/sqm — implausibly expensive even for prime Moscow


@dataclass
class SchemaValidationReport:
    """Structured result of :func:`validate_listings_df`."""

    n_rows: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def __str__(self) -> str:  # pragma: no cover - convenience only
        lines = [f"SchemaValidationReport(n_rows={self.n_rows}, valid={self.is_valid})"]
        for e in self.errors:
            lines.append(f"  ERROR:   {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


class SchemaValidationError(ValueError):
    """Raised by :func:`validate_listings_df` when *strict=True* and errors are found."""

    def __init__(self, report: SchemaValidationReport) -> None:
        self.report = report
        super().__init__(str(report))


def validate_listings_df(
    df: pd.DataFrame,
    *,
    strict: bool = False,
    year_upper_bound: int = 2027,
) -> SchemaValidationReport:
    """Validate a raw/cleaned real-estate listings DataFrame.

    Checks performed (each failure is reported by ROW COUNT, not by raising
    on the first bad row, so a single call gives a complete picture):

    - required columns present;
    - ``price > 0`` (and within a plausible range);
    - ``total_area > 0`` (and within a plausible range);
    - ``floor >= 1``;
    - ``floors_total >= floor``;
    - ``year_built`` within ``[1850, year_upper_bound]`` when present;
    - ``rooms`` within ``[0, 15]`` when present;
    - ``city`` is one of the 8 known cities (warning, not error — new cities
      may legitimately be added later);
    - ``building_type`` is a known value/alias when present (warning);
    - duplicate ``url`` values (error — indicates the same listing scraped twice);
    - anomalous ``price_per_sqm`` (price / total_area) outside a plausible
      RUB/sqm range (warning — flags likely data-entry errors without being
      so strict it rejects legitimate outliers).

    Args:
        df: DataFrame to validate.
        strict: If True, raise :class:`SchemaValidationError` when any error
            (not warning) is found. If False (default), return the report
            and let the caller decide.
        year_upper_bound: Reject/flag ``year_built`` values after this year
            (default 2027, one year ahead of "today" to tolerate
            end-of-year listings for buildings still under construction).

    Returns:
        A :class:`SchemaValidationReport` with ``errors``/``warnings`` lists.

    Raises:
        SchemaValidationError: If ``strict=True`` and any error was found.
    """
    n_rows = len(df)
    report = SchemaValidationReport(n_rows=n_rows)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report.errors.append(f"Missing required columns: {missing_cols}")
        # Can't safely check row-level constraints on columns that don't exist.
        if strict and report.errors:
            raise SchemaValidationError(report)
        return report

    _check_range(report, df, "price", _MIN_PRICE, _MAX_PRICE, allow_missing=False)
    _check_range(report, df, "total_area", _MIN_AREA, _MAX_AREA, allow_missing=False)
    _check_min(report, df, "floor", _MIN_FLOOR)
    _check_max(report, df, "floor", _MAX_FLOOR, level="warnings")

    if "floors_total" in df.columns and "floor" in df.columns:
        bad = (
            df["floors_total"].notna()
            & df["floor"].notna()
            & (
                pd.to_numeric(df["floors_total"], errors="coerce")
                < pd.to_numeric(df["floor"], errors="coerce")
            )
        )
        if bad.any():
            report.errors.append(f"{int(bad.sum())} row(s) have floors_total < floor (impossible).")

    if "year_built" in df.columns:
        year = pd.to_numeric(df["year_built"], errors="coerce")
        bad = year.notna() & ((year < _MIN_YEAR_BUILT) | (year > year_upper_bound))
        if bad.any():
            report.errors.append(
                f"{int(bad.sum())} row(s) have year_built outside "
                f"[{_MIN_YEAR_BUILT}, {year_upper_bound}]."
            )

    if "rooms" in df.columns:
        rooms = pd.to_numeric(df["rooms"], errors="coerce")
        bad = rooms.notna() & ((rooms < 0) | (rooms > _MAX_ROOMS))
        if bad.any():
            report.errors.append(f"{int(bad.sum())} row(s) have rooms outside [0, {_MAX_ROOMS}].")

    if "city" in df.columns:
        cities = df["city"].fillna("").astype(str).str.strip().str.lower()
        unknown = ~cities.isin(_KNOWN_CITIES) & (cities != "")
        if unknown.any():
            report.warnings.append(
                f"{int(unknown.sum())} row(s) have a city not in the known set "
                f"({sorted(_KNOWN_CITIES)}): {sorted(cities[unknown].unique().tolist())[:10]}"
            )

    if "building_type" in df.columns:
        bt = df["building_type"].fillna("").astype(str).str.strip().str.lower()
        unknown = ~bt.isin(_KNOWN_BUILDING_TYPES) & (bt != "")
        if unknown.any():
            report.warnings.append(
                f"{int(unknown.sum())} row(s) have a building_type outside the known "
                f"vocabulary: {sorted(bt[unknown].unique().tolist())[:10]}"
            )

    if "url" in df.columns:
        dup_mask = df["url"].notna() & df.duplicated(subset=["url"], keep=False)
        if dup_mask.any():
            n_dup_groups = df.loc[dup_mask, "url"].nunique()
            report.errors.append(
                f"{int(dup_mask.sum())} row(s) share a duplicate url "
                f"across {n_dup_groups} distinct URL(s) — same listing scraped twice?"
            )

    if "price" in df.columns and "total_area" in df.columns:
        price = pd.to_numeric(df["price"], errors="coerce")
        area = pd.to_numeric(df["total_area"], errors="coerce").replace(0, pd.NA)
        ppsqm = price / area
        bad = ppsqm.notna() & ((ppsqm < _PRICE_PER_SQM_MIN) | (ppsqm > _PRICE_PER_SQM_MAX))
        if bad.any():
            report.warnings.append(
                f"{int(bad.sum())} row(s) have an implausible price_per_sqm outside "
                f"[{_PRICE_PER_SQM_MIN:,.0f}, {_PRICE_PER_SQM_MAX:,.0f}] RUB/sqm."
            )

    if all(c in df.columns for c in ("address", "rooms", "floor", "price", "total_area")):
        near_dupes = find_near_duplicate_candidates(df)
        if not near_dupes.empty:
            report.warnings.append(
                f"{near_dupes['url'].nunique()} row(s) across {near_dupes['group_id'].nunique()} "
                "group(s) are NEAR-DUPLICATE CANDIDATES (same normalized address+rooms+floor, "
                "price/area within tolerance) — not auto-removed, see "
                "find_near_duplicate_candidates() for the full candidates report."
            )

    if strict and report.errors:
        raise SchemaValidationError(report)
    return report


def find_near_duplicate_candidates(
    df: pd.DataFrame,
    *,
    price_tolerance: float = 0.02,
    area_tolerance: float = 0.02,
) -> pd.DataFrame:
    """Find near-duplicate LISTING candidates — rows that are not exact
    ``url`` duplicates but plausibly describe the same underlying property
    (e.g. the same apartment re-listed under a second category sweep, or by
    a different agency).

    This NEVER deletes, merges, or otherwise modifies rows — it returns a
    candidates report for a human/downstream process to review, per the
    project's "don't auto-remove ambiguous matches" data-quality policy.

    Matching key: normalized address (lowercased, whitespace-collapsed) +
    ``rooms`` + ``floor``, with ``price`` and ``total_area`` both required to
    be within ``price_tolerance``/``area_tolerance`` (relative) of each
    other. The address-text match alone is intentionally loose — restate.ru
    addresses are street-level, so many genuinely distinct apartments
    legitimately share one street address — it's numeric agreement on
    price/area that does the actual discriminating; rows sharing an address
    but with very different price/area are correctly NOT flagged.

    Args:
        df: Listings DataFrame. Returns an empty result (not an error) if
            any of the required columns is absent, or the DataFrame is empty.
        price_tolerance: Max relative price difference to still count as a
            match (default 2%).
        area_tolerance: Max relative area difference to still count as a
            match (default 2%).

    Returns:
        DataFrame with one row per candidate listing, columns ``group_id``
        (candidates sharing a ``group_id`` are mutually near-duplicate),
        ``url``, ``address_normalized``, ``rooms``, ``floor``, ``price``,
        ``total_area``. Empty (zero rows) if no candidates were found.
    """
    required = ("address", "rooms", "floor", "price", "total_area")
    empty_result = pd.DataFrame(
        columns=[
            "group_id",
            "df_index",
            "url",
            "address_normalized",
            "rooms",
            "floor",
            "price",
            "total_area",
        ]
    )
    if df.empty or not all(c in df.columns for c in required):
        return empty_result

    work = df.copy()
    work["_addr_norm"] = (
        work["address"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["total_area"] = pd.to_numeric(work["total_area"], errors="coerce")

    candidates: List[dict] = []
    group_id = 0
    group_cols = ["_addr_norm", "rooms", "floor"]
    for key, group in work.groupby(group_cols, dropna=False):
        addr_norm = key[0] if isinstance(key, tuple) else key
        if not addr_norm or len(group) < 2:
            continue
        remaining = group.copy()
        while len(remaining) > 1:
            base = remaining.iloc[0]
            if pd.isna(base["price"]) or pd.isna(base["total_area"]):
                remaining = remaining.iloc[1:]
                continue
            price_ref = max(abs(base["price"]), 1.0)
            area_ref = max(abs(base["total_area"]), 1.0)
            same = remaining[
                remaining["price"].notna()
                & remaining["total_area"].notna()
                & (remaining["price"].sub(base["price"]).abs() <= price_tolerance * price_ref)
                & (
                    remaining["total_area"].sub(base["total_area"]).abs()
                    <= area_tolerance * area_ref
                )
            ]
            if len(same) > 1:
                group_id += 1
                for _, row in same.iterrows():
                    candidates.append(
                        {
                            "group_id": group_id,
                            "df_index": row.name,
                            "url": row.get("url"),
                            "address_normalized": addr_norm,
                            "rooms": key[1] if isinstance(key, tuple) else None,
                            "floor": key[2] if isinstance(key, tuple) else None,
                            "price": row["price"],
                            "total_area": row["total_area"],
                        }
                    )
            remaining = remaining.drop(same.index)

    return pd.DataFrame(candidates) if candidates else empty_result


def build_split_groups(df: pd.DataFrame) -> pd.Series:
    """Return a group-id Series (indexed like *df*) suitable for
    ``sklearn.model_selection.GroupShuffleSplit``/``GroupKFold``, so that
    near-duplicate listings (see :func:`find_near_duplicate_candidates`)
    always land entirely in train or entirely in test — never split across
    both, which would let the model evaluate on a listing it effectively
    already trained on (37.5% of near-duplicate groups leak across a plain
    random split on this project's dataset).

    Rows that are NOT part of any near-duplicate group each get their own
    unique group id (their own ``url``, or positional index if ``url`` is
    absent) so they behave exactly like a normal random split would for the
    ~97% of the dataset with no near-duplicate twin.

    Returns:
        A ``str`` Series, one group id per row of *df*, indexed like *df*.
    """
    if "url" in df.columns:
        groups = df["url"].astype(str).copy()
    else:
        groups = pd.Series(df.index.astype(str), index=df.index)

    cands = find_near_duplicate_candidates(df)
    if cands.empty:
        return groups

    # Join on the original DataFrame index (df_index), not `url` -- this
    # must work even when `url` is absent from *df* (e.g. a feature-only
    # DataFrame), otherwise near-duplicate rows would silently each get
    # their own unique group id instead of sharing one, defeating the
    # entire point of this function. Found via a test with no `url` column.
    index_to_group_id = dict(
        zip(cands["df_index"], cands["group_id"].astype(str).radd("near_dup_"))
    )
    mapped = pd.Series(df.index, index=df.index).map(index_to_group_id)
    groups = groups.where(mapped.isna(), mapped)
    return groups


def build_location_groups(df: pd.DataFrame, coord_decimals: int = 4) -> pd.Series:
    """Return a *building-level* group-id Series for
    ``GroupShuffleSplit``/``GroupKFold`` — stricter than
    :func:`build_split_groups`.

    Two rows share a location group when EITHER

    * they are near-duplicate listings (:func:`build_split_groups` — the
      same listing re-posted, or two listings with the same
      address+rooms+floor and price/area within 2%), OR
    * both carry coordinates that round to the same
      ``(latitude, longitude)`` at ``coord_decimals`` places.

    ``coord_decimals=4`` is ~11 m. The restate.ru listing pages emit
    coordinates at **at most 4 decimal places** (this dataset: 90% of
    coordinate rows are 4-dp, the rest 3-dp or coarser), so this is not a
    lossy down-round: it groups at the granularity the source geocoder
    actually resolves, which is the building / address point. Listings in
    the same building therefore always land on the same side of a split.

    Rows without coordinates keep their :func:`build_split_groups` id
    (listing / near-duplicate level). These are ~8% of this dataset and
    also cannot carry OSM geo features (``has_coordinates=0``), so any
    residual same-building pairing among them affects a geo model and its
    pre-geo baseline equally and does not inflate the measured geo uplift.

    Returns:
        A ``str`` Series, one group id per row of *df*, indexed like *df*.
    """
    base = build_split_groups(df).astype(str)
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return base

    lat = pd.to_numeric(df["latitude"], errors="coerce").round(coord_decimals)
    lon = pd.to_numeric(df["longitude"], errors="coerce").round(coord_decimals)
    has_coord = lat.notna() & lon.notna()
    if not has_coord.any():
        return base

    # Union-find over row positions: connect rows sharing a base group and
    # rows sharing a rounded coordinate, then relabel by component root.
    n = len(df)
    parent = list(range(n))

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    pos = {idx: p for p, idx in enumerate(df.index)}
    for _key, members in base.groupby(base).groups.items():
        it = iter(members)
        first = pos[next(it)]
        for m in it:
            union(first, pos[m])

    coord_key = lat.astype("string") + "," + lon.astype("string")
    for _key, members in coord_key[has_coord].groupby(coord_key[has_coord]).groups.items():
        it = iter(members)
        first = pos[next(it)]
        for m in it:
            union(first, pos[m])

    roots = [find(p) for p in range(n)]
    return pd.Series([f"loc_{r}" for r in roots], index=df.index)


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------


def _check_range(
    report: SchemaValidationReport,
    df: pd.DataFrame,
    col: str,
    lo: float,
    hi: float,
    allow_missing: bool,
) -> None:
    if col not in df.columns:
        if not allow_missing:
            report.errors.append(f"Required column '{col}' missing.")
        return
    values = pd.to_numeric(df[col], errors="coerce")
    bad = values.notna() & ((values <= 0) | (values < lo) | (values > hi))
    if bad.any():
        report.errors.append(
            f"{int(bad.sum())} row(s) have '{col}' outside plausible range "
            f"({lo:,.0f}, {hi:,.0f}] or <= 0."
        )


def _check_min(report: SchemaValidationReport, df: pd.DataFrame, col: str, lo: float) -> None:
    if col not in df.columns:
        return
    values = pd.to_numeric(df[col], errors="coerce")
    bad = values.notna() & (values < lo)
    if bad.any():
        report.errors.append(f"{int(bad.sum())} row(s) have '{col}' < {lo}.")


def _check_max(
    report: SchemaValidationReport, df: pd.DataFrame, col: str, hi: float, level: str = "errors"
) -> None:
    if col not in df.columns:
        return
    values = pd.to_numeric(df[col], errors="coerce")
    bad = values.notna() & (values > hi)
    if bad.any():
        getattr(report, level).append(f"{int(bad.sum())} row(s) have '{col}' > {hi}.")

"""Проверка сырых данных: схема и логические инварианты.

Хардкодятся структура и логика формата, а не точные количества строк.
Запуск:  python -m second_year.data.validate
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import pandas as pd

from second_year.config import Config, load_config
from second_year.data.load import iter_item_properties, load_category_tree, load_events

EVENT_TYPES = {"view", "addtocart", "transaction"}
EVENTS_COLUMNS = ["timestamp", "visitorid", "event", "itemid", "transactionid"]
PROPERTIES_COLUMNS = ["timestamp", "itemid", "property", "value"]
CATEGORY_COLUMNS = ["categoryid", "parentid"]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name, passed, detail))

    @property
    def ok(self) -> bool:
        return all(r.passed for r in self.results)

    def render(self) -> str:
        lines = []
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"[{mark}] {r.name}" + (f" — {r.detail}" if r.detail else ""))
        lines.append("")
        lines.append(f"ИТОГ: {'PASS' if self.ok else 'FAIL'} "
                     f"({sum(r.passed for r in self.results)}/{len(self.results)} проверок)")
        return "\n".join(lines)


def validate_events(cfg: Config, report: ValidationReport) -> None:
    df = load_events(cfg)
    report.add("events: набор колонок", list(df.columns) == EVENTS_COLUMNS, str(list(df.columns)))
    report.add("events: timestamp целочисленный", pd.api.types.is_integer_dtype(df["timestamp"]))
    report.add("events: visitorid без пропусков", int(df["visitorid"].isna().sum()) == 0)
    report.add("events: itemid без пропусков", int(df["itemid"].isna().sum()) == 0)

    bad_types = set(df["event"].dropna().unique()) - EVENT_TYPES
    report.add("events: event в {view, addtocart, transaction}", not bad_types, f"лишние: {bad_types}")

    is_txn = df["event"] == "transaction"
    txn_without_id = int((is_txn & df["transactionid"].isna()).sum())
    nontxn_with_id = int((~is_txn & df["transactionid"].notna()).sum())
    report.add("events: transactionid только у transaction",
               txn_without_id == 0 and nontxn_with_id == 0,
               f"transaction без id: {txn_without_id}, не-transaction с id: {nontxn_with_id}")

    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    plausible = (ts.dt.year >= 2014).all() and (ts.dt.year <= 2016).all()
    report.add("events: timestamp в разумном диапазоне (2014–2016)", bool(plausible),
               f"{ts.min()} .. {ts.max()}")

    full_dups = int(df.duplicated().sum())
    report.add("events: полные дубликаты строк (информационно, не блокер)", True,
               f"{full_dups} шт.")


def validate_category_tree(cfg: Config, report: ValidationReport) -> None:
    df = load_category_tree(cfg)
    report.add("category_tree: набор колонок", list(df.columns) == CATEGORY_COLUMNS, str(list(df.columns)))
    report.add("category_tree: categoryid уникален", int(df["categoryid"].duplicated().sum()) == 0)

    known = set(df["categoryid"].tolist())
    parents = set(df["parentid"].dropna().astype("int64").tolist())
    missing_parents = parents - known
    report.add("category_tree: все parentid присутствуют как categoryid",
               not missing_parents, f"отсутствуют: {sorted(missing_parents)[:10]}")

    parent_map = dict(zip(df["categoryid"], df["parentid"]))

    def has_cycle(start: int) -> bool:
        seen: set[int] = set()
        node = start
        while node in parent_map and pd.notna(parent_map[node]):
            if node in seen:
                return True
            seen.add(node)
            node = int(parent_map[node])
        return False

    cyclic = [c for c in df["categoryid"] if has_cycle(int(c))]
    report.add("category_tree: отсутствие циклов", not cyclic, f"циклические: {cyclic[:10]}")


def validate_properties(cfg: Config, report: ValidationReport) -> None:
    schema_ok = True
    first_schema: list[str] | None = None
    n_rows = 0
    bad_itemid = 0
    key_parts: list[pd.DataFrame] = []

    for chunk in iter_item_properties(cfg):
        cols = list(chunk.columns)
        if first_schema is None:
            first_schema = cols
        schema_ok = schema_ok and cols == PROPERTIES_COLUMNS
        n_rows += len(chunk)
        bad_itemid += int(chunk["itemid"].isna().sum())
        key_parts.append(chunk[["timestamp", "itemid", "property"]])

    keys = pd.concat(key_parts, ignore_index=True)
    key_dups = int(keys.duplicated().sum())
    ts = pd.to_datetime([int(keys["timestamp"].min()), int(keys["timestamp"].max())],
                        unit="ms", utc=True)

    report.add("properties: единая схема part1/part2", schema_ok, str(first_schema))
    report.add("properties: itemid без пропусков", bad_itemid == 0)
    report.add("properties: нет дублей по ключу (timestamp, itemid, property)", key_dups == 0,
               f"{key_dups} шт.")
    report.add("properties: timestamp в разумном диапазоне", bool((ts.year >= 2014).all()),
               f"{ts[0]} .. {ts[1]}")
    report.add("properties: всего строк (информационно)", True, f"{n_rows}")


def run_validation(cfg: Config | None = None) -> ValidationReport:
    cfg = cfg or load_config()
    report = ValidationReport()
    validate_events(cfg, report)
    validate_category_tree(cfg, report)
    validate_properties(cfg, report)
    return report


def main() -> int:
    report = run_validation()
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

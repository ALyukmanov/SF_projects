"""Загрузка конфигурации и разрешение путей относительно корня проекта."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# .../retail-recommender-system/src/retail_recommender/config.py -> корень проекта
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class SplitBounds:
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    paths: Paths
    event_weights: dict[str, int]
    strong_events: list[str]
    transaction_events: list[str]
    bot_event_threshold: int
    split: SplitBounds
    properties_first_snapshot: str
    feature_engineering: dict[str, Any]
    random_seed: int
    raw_files: dict[str, Any] = field(default_factory=dict)

    # --- удобные производные пути ---
    @property
    def events_csv(self) -> Path:
        return self.paths.raw_dir / self.raw_files["events"]

    @property
    def category_tree_csv(self) -> Path:
        return self.paths.raw_dir / self.raw_files["category_tree"]

    @property
    def item_property_csvs(self) -> list[Path]:
        return [self.paths.raw_dir / name for name in self.raw_files["item_properties"]]


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    root = PROJECT_ROOT
    p = raw["paths"]
    paths = Paths(
        root=root,
        raw_dir=root / p["raw_dir"],
        interim_dir=root / p["interim_dir"],
        processed_dir=root / p["processed_dir"],
        artifacts_dir=root / p["artifacts_dir"],
        reports_dir=root / p["reports_dir"],
    )
    return Config(
        raw=raw,
        paths=paths,
        raw_files=raw["raw_files"],
        event_weights=raw["event_weights"],
        strong_events=list(raw["strong_events"]),
        transaction_events=list(raw["transaction_events"]),
        bot_event_threshold=int(raw["bot_event_threshold"]),
        split=SplitBounds(**raw["split"]),
        properties_first_snapshot=raw["properties_first_snapshot"],
        feature_engineering=raw["feature_engineering"],
        random_seed=int(raw["random_seed"]),
    )

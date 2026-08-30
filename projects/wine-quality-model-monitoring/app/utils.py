"""Мелкие общие утилиты."""
from __future__ import annotations

import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

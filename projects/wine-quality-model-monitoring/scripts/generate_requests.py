#!/usr/bin/env python
"""Генератор трафика: шлёт случайные строки датасета в POST /predict.

Запуск:
    python scripts/generate_requests.py --count 100 --delay 0.2
    python scripts/generate_requests.py --url http://localhost:8000 --count 50
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data" / "winequality-red.csv"


def load_rows(dataset_path: Path) -> list[dict]:
    df = pd.read_csv(dataset_path, sep=";")
    return df[FEATURE_NAMES].to_dict(orient="records")


def main() -> int:
    parser = argparse.ArgumentParser(description="Отправка тестового трафика в /predict")
    parser.add_argument("--url", default="http://localhost:8000", help="Базовый URL API")
    parser.add_argument("--count", type=int, default=100, help="Сколько запросов отправить")
    parser.add_argument("--delay", type=float, default=0.2, help="Задержка между запросами, в секундах")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Путь к CSV с данными")
    parser.add_argument("--timeout", type=float, default=5.0, help="Таймаут одного запроса, в секундах")
    args = parser.parse_args()

    try:
        rows = load_rows(args.dataset)
    except Exception as exc:  # noqa: BLE001
        print(f"Не удалось загрузить датасет из {args.dataset}: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("В датасете нет строк для выборки", file=sys.stderr)
        return 1

    endpoint = args.url.rstrip("/") + "/predict"
    successes = 0
    failures = 0
    latencies: list[float] = []

    print(f"Отправляю {args.count} запросов на {endpoint} (задержка={args.delay}с)")

    for i in range(args.count):
        payload = random.choice(rows)
        started = time.perf_counter()
        try:
            response = requests.post(endpoint, json=payload, timeout=args.timeout)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            if response.status_code == 200:
                successes += 1
            else:
                failures += 1
                print(f"[{i + 1}/{args.count}] HTTP {response.status_code}: {response.text[:200]}")
        except requests.RequestException as exc:
            failures += 1
            print(f"[{i + 1}/{args.count}] запрос не прошёл: {exc}")

        if args.delay > 0 and i < args.count - 1:
            time.sleep(args.delay)

    print("\n--- Итоги ---")
    print(f"Всего запросов   : {args.count}")
    print(f"Успешных         : {successes}")
    print(f"Неудачных        : {failures}")
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"Средняя задержка : {avg_latency * 1000:.1f} мс")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

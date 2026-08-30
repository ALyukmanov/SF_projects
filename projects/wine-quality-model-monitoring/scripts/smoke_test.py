#!/usr/bin/env python
"""Smoke-тест поднятого стека.

Проверяет запущенный `docker compose up`:
  - API здоров и модель загружена;
  - /model/info отдаёт нужные поля;
  - /predict возвращает валидное предсказание;
  - /metrics отдаёт формат Prometheus;
  - Prometheus готов, target ml-model-api в статусе UP;
  - Grafana отвечает health-check'ом.

Запуск:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --api-url http://localhost:8000 --prometheus-url http://localhost:9090 --grafana-url http://localhost:3000
"""
from __future__ import annotations

import argparse
import sys

import requests

SAMPLE_WINE = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11,
    "total sulfur dioxide": 34,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


class Check:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.detail = ""

    def ok(self, detail: str = "") -> "Check":
        self.passed = True
        self.detail = detail
        return self

    def fail(self, detail: str) -> "Check":
        self.passed = False
        self.detail = detail
        return self


def check_health(api_url: str, timeout: float) -> Check:
    check = Check("API /health")
    try:
        resp = requests.get(f"{api_url}/health", timeout=timeout)
        body = resp.json()
        if resp.status_code == 200 and body.get("status") == "ok" and body.get("model_loaded") is True:
            return check.ok(f"model_version={body.get('model_version')}")
        return check.fail(f"неожиданный ответ: {resp.status_code} {body}")
    except Exception as exc:  # noqa: BLE001
        return check.fail(str(exc))


def check_model_info(api_url: str, timeout: float) -> Check:
    check = Check("API /model/info")
    try:
        resp = requests.get(f"{api_url}/model/info", timeout=timeout)
        body = resp.json()
        required = {"model_version", "model_type", "feature_names", "accuracy", "f1_score"}
        if resp.status_code == 200 and required.issubset(body.keys()):
            return check.ok(f"version={body['model_version']} accuracy={body['accuracy']:.3f}")
        return check.fail(f"неожиданный ответ: {resp.status_code} {body}")
    except Exception as exc:  # noqa: BLE001
        return check.fail(str(exc))


def check_predict(api_url: str, timeout: float) -> Check:
    check = Check("API /predict")
    try:
        resp = requests.post(f"{api_url}/predict", json=SAMPLE_WINE, timeout=timeout)
        body = resp.json()
        if resp.status_code == 200 and "prediction" in body and "good_quality_probability" in body:
            return check.ok(f"prediction={body['prediction']} proba={body['good_quality_probability']:.3f}")
        return check.fail(f"неожиданный ответ: {resp.status_code} {body}")
    except Exception as exc:  # noqa: BLE001
        return check.fail(str(exc))


def check_metrics(api_url: str, timeout: float) -> Check:
    check = Check("API /metrics")
    try:
        resp = requests.get(f"{api_url}/metrics", timeout=timeout)
        text = resp.text
        required_metrics = ["ml_model_accuracy", "ml_predictions_total", "ml_prediction_duration_seconds"]
        missing = [m for m in required_metrics if m not in text]
        if resp.status_code == 200 and not missing:
            return check.ok(f"строк в ответе: {len(text.splitlines())}")
        return check.fail(f"status={resp.status_code} missing={missing}")
    except Exception as exc:  # noqa: BLE001
        return check.fail(str(exc))


def check_prometheus_ready(prometheus_url: str, timeout: float) -> Check:
    check = Check("Готовность Prometheus")
    try:
        resp = requests.get(f"{prometheus_url}/-/ready", timeout=timeout)
        if resp.status_code == 200:
            return check.ok()
        return check.fail(f"status={resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        return check.fail(str(exc))


def check_prometheus_target_up(prometheus_url: str, timeout: float) -> Check:
    check = Check("Prometheus target ml-model-api (UP)")
    try:
        resp = requests.get(f"{prometheus_url}/api/v1/targets", timeout=timeout)
        data = resp.json()
        active = data.get("data", {}).get("activeTargets", [])
        target = next((t for t in active if t.get("labels", {}).get("job") == "ml-model-api"), None)
        if target is None:
            return check.fail("job=ml-model-api не найден среди активных targets")
        if target.get("health") == "up":
            return check.ok()
        return check.fail(f"health={target.get('health')} lastError={target.get('lastError')}")
    except Exception as exc:  # noqa: BLE001
        return check.fail(str(exc))


def check_grafana_health(grafana_url: str, timeout: float) -> Check:
    check = Check("Здоровье Grafana")
    try:
        resp = requests.get(f"{grafana_url}/api/health", timeout=timeout)
        body = resp.json()
        if resp.status_code == 200 and body.get("database") == "ok":
            return check.ok()
        return check.fail(f"неожиданный ответ: {resp.status_code} {body}")
    except Exception as exc:  # noqa: BLE001
        return check.fail(str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-тест запущенного docker compose стека")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--grafana-url", default="http://localhost:3000")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    checks = [
        check_health(args.api_url, args.timeout),
        check_model_info(args.api_url, args.timeout),
        check_predict(args.api_url, args.timeout),
        check_metrics(args.api_url, args.timeout),
        check_prometheus_ready(args.prometheus_url, args.timeout),
        check_prometheus_target_up(args.prometheus_url, args.timeout),
        check_grafana_health(args.grafana_url, args.timeout),
    ]

    print("Результаты smoke-теста:")
    all_passed = True
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        all_passed = all_passed and check.passed
        print(f"  [{status}] {check.name} - {check.detail}")

    print("\nИтог:", "PASS" if all_passed else "FAIL")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

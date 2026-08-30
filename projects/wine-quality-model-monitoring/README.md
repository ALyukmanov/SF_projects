# Мониторинг ML-модели: качество вина

Учебный проект про то, как следить за ML-моделью в проде: не только обучить её, но и отдавать предсказания через API, собирать метрики в Prometheus и смотреть на них в Grafana. Плюс — переобучать модель без остановки сервиса.

## Что реализовано

- Модель `RandomForestClassifier` (в `Pipeline` со `StandardScaler`) классифицирует вино на «хорошее» / «плохое» по химическому составу — датасет [Wine Quality с UCI](data/README.md).
- FastAPI-сервис: предсказание, batch-предсказание, переобучение, информация о модели.
- Метрики модели и запросов экспортируются в формате Prometheus.
- Grafana-дашборд с 15 панелями: качество модели, трафик, латентность, ошибки, версия модели.
- Весь стек поднимается одной командой через Docker Compose.

## Стек

Python, scikit-learn, FastAPI + Pydantic v2, `prometheus-client`, Prometheus, Grafana, Docker Compose, pytest.

## Как запустить

```bash
docker compose up --build
```

При первом запуске модель обучается автоматически, если артефактов ещё нет.

| Сервис | Адрес |
|---|---|
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Метрики | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

Остановить: `docker compose down`. Удалить volumes (данные Prometheus/Grafana и обученную модель): `docker compose down -v`.

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0,
    "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11,
    "total sulfur dioxide": 34, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4
  }'
```

Ответ:

```json
{
  "prediction": 0,
  "label": "low_quality",
  "good_quality_probability": 0.27,
  "model_version": "20260726T095837365876Z"
}
```

Эндпоинты: `/health`, `/model/info`, `/predict`, `/predict/batch`, `/retrain`, `/metrics`. Полная схема — в Swagger UI.

Сгенерировать тестовый трафик:

```bash
python scripts/generate_requests.py --count 100 --delay 0.2
```

## Мониторинг

Prometheus опрашивает `/metrics` каждые 5 секунд. Проверить target: http://localhost:9090/targets — job `ml-model-api` должен быть `UP`.

Grafana подключена автоматически (datasource и дашборд заведены через provisioning) — сразу открывайте http://localhost:3000 → Dashboards → **ML Model Monitoring**.

На дашборде: accuracy/precision/recall/F1/ROC AUC модели, количество и скорость предсказаний, латентность (p50/p95/p99), распределение классов, ошибки, статус сервиса, версия и метаданные модели.

## Переобучение модели

```bash
curl -X POST http://localhost:8000/retrain
# либо локально: python -m app.training  /  make train
```

Чтобы показать переобучение на новых данных: допишите строки в `data/winequality-red.csv` (тот же формат) и вызовите `/retrain`. Модель обучается заново на всём датасете — это полный re-fit, не `partial_fit` (`RandomForestClassifier` инкрементальное обучение не поддерживает). После переобучения `model_version` меняется, и новые запросы к `/predict` сразу используют новую модель.

Повторный `/retrain`, пока уже идёт обучение, получает `409 Conflict`. Если обучение упадёт с ошибкой, старая модель продолжает отвечать — артефакты подменяются атомарно и только после успешного завершения.

## Результаты модели

| Метрика | Значение |
|---|---:|
| Accuracy | 0.819 |
| Precision | 0.834 |
| Recall | 0.825 |
| F1 | 0.829 |
| ROC AUC | 0.907 |

Метрики — на отложенной тестовой выборке (20% данных, стратифицированное разбиение, `random_state=42`). Цифры немного плавают от переобучения к переобучению.

## Тесты

```bash
python -m pytest -q
```

28 тестов: обучение и сохранение модели, API, метрики, конкурентный retrain, откат при неудачном обучении.

Smoke-тест поднятого стека (API, Prometheus, Grafana):

```bash
python scripts/smoke_test.py
```

## Структура проекта

```text
app/          — API, обучение и работа с моделью
tests/        — тесты
grafana/      — дашборд и provisioning
prometheus/   — конфигурация Prometheus
scripts/      — smoke-test и генератор трафика
data/         — датасет
artifacts/    — модель и метаданные (создаются при обучении, в git не попадают)
```

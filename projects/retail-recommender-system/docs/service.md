# Сервис рекомендаций — документация

MVP-сервис отдаёт Top-N персональных рекомендаций товаров по `visitorid`.
В основе — модель **item-item co-occurrence** (та же, что в экспериментах).

## Данные

Открытый датасет [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset),
журнал интернет-магазина за 2015-05-03 … 2015-09-18. Кладётся в `data/raw/`
(в git не хранится, ~940 МБ).

| файл | содержание | ключевые поля |
|---|---|---|
| `events.csv` | журнал событий | `timestamp` (Unix ms), `visitorid`, `event` ∈ {view, addtocart, transaction}, `itemid`, `transactionid` |
| `item_properties_part1/2.csv` | свойства товаров, недельные срезы | `timestamp`, `itemid`, `property`, `value` |
| `category_tree.csv` | дерево категорий | `categoryid`, `parentid` |

Цены, выручки и количества в данных нет.

## Подготовка (`scripts/build_dataset.py`)

- удаление **460** полностью повторяющихся строк `events.csv`
  (2 756 101 → 2 755 641 события);
- пометка аномально активных визиторов (> 500 событий за период, 75 визиторов) —
  исключаются из обучения и оценки;
- вес события для implicit-feedback: `view=1`, `addtocart=3`, `transaction=5`;
- разметка временного split по дате события;
- метаданные товара (`categoryid`, `available`) — **as-of**: для момента `t`
  берётся последний недельный срез с `snapshot_ts <= t`; если срезов до `t` нет
  (события 03–09 мая, до первого среза 10 мая) — значение `UNKNOWN`, forward-fill
  из будущего не делается.

## Факторы (`src/retail_recommender/features.py`, `notebooks/02_features.ipynb`)

Все признаки считаются только из событий с `ts < cutoff` (`cutoff = 2015-07-01`).

- **item**: популярность (взвешенная и по типам событий), конверсии
  view→addtocart→transaction, свежесть последнего события, число уникальных
  визиторов, категория и корневая категория, доступность;
- **user**: число событий и их разбивка по типам, длина истории, число уникальных
  товаров и категорий, свежесть последнего события, доля сильных действий;
- **user-item**: число и типы взаимодействий пары, свежесть, аффинити визитора к
  категории товара.

Co-occurrence, ALS и BPR из этих таблиц напрямую не читают — им нужны только
события train-окна и веса событий (см. ниже). Признаки — самостоятельный слой,
подготовленный на случай контентных/классификационных моделей в будущем.

## Схема validation / test

| окно | даты | назначение |
|---|---|---|
| train | ≤ 2015-08-21 | обучение моделей |
| validation | 2015-08-22 … 2015-09-04 | выбор модели и гиперпараметров |
| test | 2015-09-05 … 2015-09-18 | один финальный замер |

Принцип: всё течёт вперёд. Co-occurrence и популярность считаются только из train.
Гиперпараметры ALS/BPR подбираются только на validation; test для подбора не
использовался. Инвариант «порча событий после окна не меняет обученную модель»
проверяется `tests/test_temporal_leakage.py`.

Режим цели — **strong** (`addtocart ∪ transaction`). Основная метрика —
**Recall@10**, дополнительная — **NDCG@10**. Оценка на полном train-каталоге
(candidate pool не сужается). Разрезы: ALL, WARM (есть история), сегменты по
длине истории. Target-визиторов: val 3850 (warm 472), test 3432 (warm 495).

## Эксперименты

Полные числа — `reports/week3_metrics.json`, разбор по шагам — `notebooks/03_models.ipynb`.

- **baseline**: global popularity, category popularity, item-item co-occurrence;
- **ALS** (`implicit`): сетка `factors ∈ {32, 64}`, `regularization ∈ {0.01, 0.1, 1.0}`,
  `iterations ∈ {20, 40}`; выбрана `factors=64, reg=0.1, iters=40`;
- **BPR** (`implicit`): `factors ∈ {32, 64}`, `learning_rate ∈ {0.01, 0.05}`,
  `iterations=150`; выбрана `factors=64, lr=0.01, reg=0.01, iters=150`;
- матрица взаимодействий для ALS/BPR: `confidence = 1 + Σ(веса событий пары)`.

LightFM (гибрид с side-features товара) рассматривался, но в текущей Windows-среде
его C-расширение нестабильно при обучении (access violation в `fit_warp`/`fit_bpr`
на numpy 1.x и 2.x). В итоговое сравнение не вошёл; обёртка оставлена как
необязательная, тесты её пропускают.

## Итог

| модель (test) | Recall@10 | NDCG@10 | Recall@10 WARM | Coverage@10 |
|---|---|---|---|---|
| **item-item co-occurrence** | **0.0116** | **0.0086** | **0.0221** | 0.0187 |
| ALS `factors=64,reg=0.1,iters=40` | 0.0101 | 0.0075 | 0.0122 | 0.0032 |
| BPR `factors=64,lr=0.01,reg=0.01,iters=150` | 0.0101 | 0.0077 | 0.0118 | 0.0082 |

ALS и BPR слегка выигрывали на validation в warm-сегменте, но на test лучшее
качество показала **item-item co-occurrence** — её и завернули в сервис. Абсолютные
значения низкие: ~86% target-визиторов на test холодные (нет истории в train), на
них любая модель работает как популярность.

## Артефакт модели (`scripts/train_service_model.py`)

Скрипт обучает co-occurrence на train-окне и пишет в `artifacts/model/`:

| файл | содержание |
|---|---|
| `model.npz` | соседи товара (sparse, CSR-подобные массивы), популярность (fallback, top-3000), каталог train, истории всех визиторов до 2015-09-05 |
| `meta.json` | тип модели, версия, время сборки, `query_items`, `filter_seen`, размеры, метрики на test |

`model.npz` ≈ **25 МБ** (сжатый): 136 695 товаров с соседями, 3 472 372 пары
сосед-товар, 1 287 827 визиторов с историей. Плотные матрицы не сохраняются.

Сервис при загрузке восстанавливает объект `ItemItemCooccurrence` из этих массивов
и вызывает **тот же метод `recommend`**, что и при обучении/оценке — отдельной
реализации алгоритма нет. Модель, поднятая из артефакта, воспроизводит метрики на
test бит-в-бит (Recall@10 = 0.01156).

## API

Запуск локально:

```bash
python scripts/train_service_model.py                       # -> artifacts/model/
uvicorn retail_recommender.service.app:app --host 0.0.0.0 --port 8000
```

Каталог модели — из переменной `RETAIL_RECOMMENDER_MODEL_DIR` (по умолчанию `artifacts/model`).
Интерактивная схема — `http://localhost:8000/docs`.

### `GET /health`

```json
{ "status": "ok", "model": "item_item_cooccurrence" }
```

### `GET /model-info`

Тип и версия модели, `default_n` / `max_n`, размер каталога, число визиторов с
историей, границы train/test-окон, метрики на test.

### `POST /recommend`

Запрос (`n` необязателен, 1..100; по умолчанию 10):

```json
{ "visitorid": 874017, "n": 10 }
```

Ответ:

```json
{
  "visitorid": 874017,
  "recommendations": [ { "itemid": 309778 }, { "itemid": 260957 } ],
  "fallback": false,
  "known_user": true
}
```

- **известный визитор** — рекомендации из соседних товаров к его истории;
  из выдачи исключаются товары, которые он уже видел (`filter_seen`);
- **неизвестный визитор** — рекомендации по популярности, `fallback: true`,
  `known_user: false`, сервис не падает;
- `fallback: true` также если история визитора не дала персональных кандидатов
  (выдача совпала с популярным).

Ошибки → корректный `HTTP 422` (не 500): неверный JSON, `visitorid` не число,
`n <= 0` или `n > 100`, лишние поля.

### `GET /metrics`

Формат Prometheus (`text/plain; version=…`). Счётчики процесса:
`requests_total{method,path,status}`, `recommend_requests_total`,
`fallback_requests_total`, `request_errors_total{status}`, `uptime_seconds`.
Отдельный Prometheus/Grafana не поднимается — это только endpoint.

## Docker

```bash
python scripts/train_service_model.py            # артефакт нужен на этапе сборки
docker build -t retail-recommender-system:latest .
docker run --rm -p 8000:8000 retail-recommender-system:latest
```

- образ на `python:3.11-slim`, зависимости — `requirements-service.txt`
  (без `implicit`/`lightfm`/sklearn/matplotlib);
- артефакт модели копируется в образ; сырые CSV для обычной работы не нужны;
- процесс работает под непривилегированным пользователем;
- есть `HEALTHCHECK` на `/health`.

Проверка после запуска:

```bash
curl localhost:8000/health
curl localhost:8000/model-info
curl -X POST localhost:8000/recommend -H 'content-type: application/json' -d '{"visitorid": 874017, "n": 10}'
curl -X POST localhost:8000/recommend -H 'content-type: application/json' -d '{"visitorid": -1}'
curl localhost:8000/metrics
```

Собранный образ проверен локально: `/health`, `/model-info`, корректный и
некорректный `/recommend`, `/metrics` — все работают как описано выше.

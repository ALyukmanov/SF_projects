# Рекомендательная система для магазина

Учебный проект по Data Science: рекомендательная система интернет-магазина на
открытом датасете [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
(журнал событий за ~4.5 месяца 2015 года).

Сделано: разведочный анализ данных, подготовка признаков с целевой переменной,
сравнение baseline-моделей и коллаборативной фильтрации (ALS, BPR).

## Задача

- **пользователь** — визитор (`visitorid`), идентификатор браузера/сессии;
- **товар** — `itemid`;
- **взаимодействие** — событие `view` / `addtocart` / `transaction` (implicit feedback);
- **целевое действие** — сильное взаимодействие (`addtocart` или `transaction`);
- **что выдаём** — Top-N персональных рекомендаций: `recommend(visitorid) -> [itemid, ...]`;
- **что доступно на момент t** — события с `ts < t` и срез свойств товара на дату `<= t`.

Цен, выручки и количеств в датасете нет, поэтому денежный эффект здесь не
считается — только offline-метрики ранжирования (Recall@K, NDCG@K и др.,
обоснование в разделе 1 первого ноутбука).

## Данные

Кладутся в `data/raw/` (в git не хранятся, ~940 МБ):

| файл | содержание | ключевые поля |
|---|---|---|
| `events.csv` | журнал событий | `timestamp` (Unix ms), `visitorid`, `event`, `itemid`, `transactionid` |
| `category_tree.csv` | дерево категорий | `categoryid`, `parentid` |
| `item_properties_part1/2.csv` | свойства товаров, недельные срезы | `timestamp`, `itemid`, `property`, `value` |

## Ноутбуки

`notebooks/01_eda.ipynb` — разведочный анализ:

- постановка задачи и выбор метрик;
- проверка качества данных (пропуски, дубли, согласованность `transactionid`);
- аномально активные визиторы (порог 500 событий);
- покрытие товаров категорией;
- активность во времени, типы событий;
- распределение активности пользователей;
- популярность товаров (длинный хвост, item cold-start);
- транзакции, размер чека, воронка view → addtocart → transaction;
- временной split и точка отсечения 1 июля.

`notebooks/02_features.ipynb` — подготовка признаков:

- граница train / целевого окна (`cutoff = 2015-07-01`);
- признаки товаров (популярность, конверсии, свежесть, динамика, категория);
- признаки пользователей (активность, давность, разнообразие интересов);
- признаки пар пользователь–товар (история пары, доля в истории, affinity к категории);
- целевая переменная — сильное взаимодействие пары в целевом окне;
- проверки на утечки (признаки не используют события после cutoff).

`notebooks/03_models.ipynb` — модели:

- протокол оценки без утечки из test: train → выбор конфигурации на validation
  → один финальный замер на test;
- baseline: глобальная популярность, популярность в категории, item-item
  co-occurrence;
- коллаборативная фильтрация: ALS и BPR (`implicit`), матрица confidence из
  весов событий, подбор factors/regularization/iterations только по validation;
- сводная таблица качества (Recall@10, NDCG@10, Coverage@10) и разбор, почему
  победил baseline.

Ноутбуки генерируются скриптами `scripts/make_*_notebook.py`, чтобы код в них не
расходился с `src/`. Числа считаются кодом из `src/retail_recommender/`.

## Структура

```
configs/default.yaml          пути, веса событий, границы split, cutoff, целевое окно, протокол оценки
notebooks/
  01_eda.ipynb                разведочный анализ
  02_features.ipynb           признаки и целевая переменная
  03_models.ipynb             модели: baseline + ALS + BPR, выбор по validation, финальный test
src/retail_recommender/
  config.py                   загрузка конфигурации
  data/{load,validate}.py     чтение и проверка сырых данных
  preprocessing/              interactions, item metadata (as-of), category paths, top_properties
  split/temporal.py           временные окна
  features.py                 as-of категория, счётчики событий, целевая переменная
  eda.py                      агрегации для ноутбука
  evaluation/{metrics,protocol}.py   offline-метрики и протокол оценки моделей
  models/                     baseline (popularity, category-popularity, co-occurrence) + ALS/BPR
scripts/                      build_dataset, validate_data, run_model_experiments,
                               build_top_properties, make_*_notebook
tests/                        дедуп/веса/боты, дерево категорий, as-of, признаки, target,
                               метрики, protocol без утечки, модели (ALS/BPR)
```

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# положить сырые CSV в data/raw/

python scripts/validate_data.py    # проверка сырых данных
python scripts/build_dataset.py    # -> data/processed, data/interim
python -m pytest -q

python scripts/make_eda_notebook.py
python scripts/make_features_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_features.ipynb
```

## Модели

Сравнение в едином протоколе (`evaluation/protocol.py`): baseline (global /
category / item-item co-occurrence) против ALS и BPR (`implicit`), метрики
Recall@10 / NDCG@10, разрезы ALL / WARM. Гиперпараметры подбираются только на
validation, на test — один финальный замер выбранных конфигураций.

`implicit` стабильнее работает с `numpy<2`, поэтому для тяжёлых экспериментов —
отдельное окружение:

```bash
python -m venv .venv-models && source .venv-models/bin/activate   # Windows: .venv-models\Scripts\activate
pip install -e ".[dev]"
pip install "numpy<2" "implicit>=0.7,<0.8"

python scripts/build_top_properties.py      # -> data/interim/top_property_codes.json
python scripts/run_model_experiments.py     # -> reports/week3_metrics.json
python scripts/make_models_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/03_models.ipynb
```

Итог на test: лучший результат показал **item-item co-occurrence**
(Recall@10 = 0.0116, WARM 0.0221) — ALS и BPR на validation слегка опережали
baseline на warm-пользователях, но на test это не подтвердилось (подробный
разбор — в `notebooks/03_models.ipynb`, раздел 9).

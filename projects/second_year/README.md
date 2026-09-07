# Рекомендательная система для магазина

Учебный проект по Data Science: рекомендательная система интернет-магазина на
открытом датасете [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
(журнал событий за ~4.5 месяца 2015 года).

Сейчас выполнен первый этап — разведочный анализ данных. Сама модель
рекомендаций будет добавлена позже.

## Задача

- **пользователь** — визитор (`visitorid`), идентификатор браузера/сессии;
- **товар** — `itemid`;
- **взаимодействие** — событие `view` / `addtocart` / `transaction` (implicit feedback);
- **целевое действие** — сильное взаимодействие (`addtocart` или `transaction`);
- **что выдаём** — Top-N персональных рекомендаций: `recommend(visitorid) -> [itemid, ...]`;
- **что доступно на момент t** — события с `ts < t` и срез свойств товара на дату `<= t`.

Цен, выручки и количеств в датасете нет, поэтому денежный эффект здесь не
считается — только offline-метрики ранжирования (Recall@K, NDCG@K и др.,
обоснование в разделе 1 ноутбука).

## Данные

Кладутся в `data/raw/` (в git не хранятся, ~940 МБ):

| файл | содержание | ключевые поля |
|---|---|---|
| `events.csv` | журнал событий | `timestamp` (Unix ms), `visitorid`, `event`, `itemid`, `transactionid` |
| `category_tree.csv` | дерево категорий | `categoryid`, `parentid` |
| `item_properties_part1/2.csv` | свойства товаров, недельные срезы | `timestamp`, `itemid`, `property`, `value` |

## Что в ноутбуке

`notebooks/01_eda.ipynb`:

- постановка задачи и выбор метрик;
- проверка качества данных (пропуски, дубли, согласованность `transactionid`);
- аномально активные визиторы (порог 500 событий);
- покрытие товаров категорией;
- активность во времени, типы событий;
- распределение активности пользователей;
- популярность товаров (длинный хвост, item cold-start);
- транзакции, размер чека, воронка view → addtocart → transaction;
- временной split и точка отсечения 1 июля.

Все числа считаются кодом из `src/second_year/`, ноутбук генерируется скриптом
`scripts/make_eda_notebook.py`, чтобы не расходиться с исходниками.

## Структура

```
configs/default.yaml         пути, веса событий, границы split, cutoff
notebooks/01_eda.ipynb        разведочный анализ
src/second_year/
  config.py                   загрузка конфигурации
  data/{load,validate}.py     чтение и проверка сырых данных
  preprocessing/              interactions, item metadata (as-of), category paths
  split/temporal.py           временные окна
  eda.py                      агрегации для ноутбука
scripts/                      build_dataset, validate_data, make_eda_notebook
tests/                        дедуп/веса/боты, дерево категорий, as-of
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
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

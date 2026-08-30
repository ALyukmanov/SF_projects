# data/

## `winequality-red.csv`

- **Источник**: UCI Machine Learning Repository, датасет "Wine Quality" (красное вино), P. Cortez et al., 2009.
- **Формат**: CSV, разделитель `;`, 1599 строк + заголовок.
- **Колонки**: 11 числовых физико-химических признаков (`fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`) и целевая `quality` (оценка дегустаторов, 0–10).
- **Размер**: ~84 КБ — можно спокойно держать прямо в репозитории.
- **Лицензия**: датасет в открытом доступе для исследовательских и учебных целей.

## Как добавить данные для переобучения

Допишите строки в конец файла (тот же формат, `;`-разделитель, тот же набор колонок), затем вызовите `POST /retrain` или `python -m app.training`. Строки с пропусками или нечисловыми значениями отбрасываются автоматически перед обучением (см. `app/training.py::load_dataset`).

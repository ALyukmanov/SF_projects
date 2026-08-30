# Данные

## Источник

`winequality-red.csv` — классический датасет **Red Wine Quality**,
изначально переданный в репозиторий UCI Machine Learning (Cortez и др.,
2009) и широко зеркалируемый на Kaggle (например,
`uciml/red-wine-quality-cortez-et-al-2009`). Содержит физико-химические
измерения и сенсорную оценку качества для образцов португальского красного
вина «Vinho Verde».

- Страница UCI: https://archive.ics.uci.edu/dataset/186/wine+quality
- Зеркало на Kaggle: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

## Схема

CSV с разделителем `;`, 1 599 строк, 12 столбцов — 11 физико-химических
признаков плюс таргет:

| Столбец | Тип | Описание |
|---|---|---|
| `fixed acidity` | float | Концентрация винной кислоты (г/дм³) |
| `volatile acidity` | float | Концентрация уксусной кислоты (г/дм³); высокие значения дают неприятный уксусный привкус |
| `citric acid` | float | Концентрация лимонной кислоты (г/дм³); добавляет свежести |
| `residual sugar` | float | Остаточный сахар после брожения (г/дм³) |
| `chlorides` | float | Содержание соли (г/дм³) |
| `free sulfur dioxide` | float | Свободный SO2 (мг/дм³) |
| `total sulfur dioxide` | float | Общий (свободный + связанный) SO2 (мг/дм³) |
| `density` | float | Плотность вина (г/см³) |
| `pH` | float | Кислотность по шкале pH |
| `sulphates` | float | Добавка сульфата калия (г/дм³) |
| `alcohol` | float | Содержание алкоголя (% объёма) |
| `quality` | int | Сенсорная оценка качества, целое число 3–8, медиана минимум 3 экспертных оценок |

Пропусков ни в одном столбце нет. В датасете 240 точных дублей строк (из
1 599) — намеренно оставлены в ноутбуке, чтобы не сдвигать распределение
классов.

## Лицензия / атрибуция

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
*Modeling wine preferences by data mining from physicochemical properties.*
Decision Support Systems, 47(4), 547-553. Распространяется на условиях UCI
Machine Learning Repository для академического/исследовательского
использования.

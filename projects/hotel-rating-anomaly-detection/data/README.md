# Данные

## Источник

Датасет соревнования Kaggle **«SF Booking Hotels»**:
https://www.kaggle.com/competitions/sf-booking/data

## Файлы

| Файл | Назначение | Размер (локальная копия) |
|---|---|---|
| `hotels_train.csv` | Размеченные отзывы для обучения (есть `reviewer_score`) | ~178 МБ, 386 803 строки |
| `hotels_test.csv` | Неразмеченные отзывы для предсказания | ~59 МБ, 128 935 строк |
| `submission.csv` | Шаблон сабмита Kaggle (`id`, `reviewer_score`) | размер не проверялся |

## Схема (по фактическому заголовку `hotels_train.csv`)

| Столбец | Описание |
|---|---|
| `hotel_address` | Адрес отеля (улица, индекс, город, страна) |
| `additional_number_of_scoring` | Число оценок без письменного отзыва |
| `review_date` | Дата отзыва |
| `average_score` | Средняя оценка отеля (как отображается на Booking.com) |
| `hotel_name` | Название отеля |
| `reviewer_nationality` | Страна автора отзыва |
| `negative_review` | Текст негативного отзыва |
| `review_total_negative_word_counts` | Число слов в негативном отзыве |
| `total_number_of_reviews` | Всего отзывов у отеля |
| `positive_review` | Текст позитивного отзыва |
| `review_total_positive_word_counts` | Число слов в позитивном отзыве |
| `total_number_of_reviews_reviewer_has_given` | Всего отзывов, написанных этим автором |
| `reviewer_score` | Оценка от автора отзыва (целевая переменная) |
| `tags` | Теги: цель поездки, тип номера, число ночей и т.д. |
| `days_since_review` | Дней между датой отзыва и выгрузкой данных |
| `lat`, `lng` | Географические координаты отеля |

Данные предоставлены Kaggle для соревнования `sf-booking`.

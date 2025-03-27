# Проект 3. Анализ отзывов об отелях Booking.com

## Оглавление  

[1. Описание проекта](https://github.com/ALyukmanov/SF_Projects/tree/master/project_3/README.md#Описание-проекта)  
[2. Данные](https://github.com/ALyukmanov/SF_Projects/tree/master/project_3/README.md#Данные)  
[3. Структура проекта](https://github.com/ALyukmanov/SF_Projects/tree/master/project_3/README.md#Структура-проекта)  
[4. Библиотеки](https://github.com/ALyukmanov/SF_Projects/tree/master/project_3/README.md#Библиотеки)  
[5. Результаты](https://github.com/ALyukmanov/SF_Projects/tree/master/project_3/README.md#Результаты)  

## Описание проекта

Этот проект направлен на выявление отелей с потенциально накрученными рейтингами на платформе Booking.com. Мы строим модель машинного обучения, которая предсказывает рейтинг отеля на основе различных характеристик. Если предсказания модели значительно отличаются от фактического рейтинга, это может указывать на нечестные практики со стороны отеля.

## Данные

Данные были взяты с соревнования на kaggle.com:
- https://www.kaggle.com/competitions/sf-booking/data

В наборе данных содержатся следующие признаки:

| Признак | Описание |
|---------|----------|
| hotel_address | Адрес отеля (улица, индекс, город, страна) |
| additional_number_of_scoring | Количество оценок отеля без отзывов |
| review_date | Дата отзыва |
| average_score | Средний рейтинг отеля |
| hotel_name | Название отеля |
| reviewer_nationality | Страна рецензента |
| negative_review | Текст негативного отзыва |
| total_number_of_reviews | Общее количество отзывов об отеле |
| positive_review | Текст позитивного отзыва |
| review_total_positive_word_counts | Количество слов в позитивном отзыве |
| total_number_of_reviews_reviewer_has_given | Общее количество отзывов, оставленных рецензентом |
| reviewer_score | Оценка, поставленная рецензентом |
| tags | Теги, описывающие цель поездки, тип номера, количество ночей |
| days_since_review | Количество дней с момента отзыва до выгрузки данных |
| lat, lng | Географические координаты отеля |

## Структура проекта

1. **Исследование данных и проектирование признаков (Feature Engineering)**
   - Признаки, характеризующие отель
   - Информация о рецензенте
   - Обработка отзывов

2. **Анализ и отбор признаков (Feature Selection)**
   - Анализ мультиколлинеарности
   - Оценка значимости признаков

3. **Обучение модели и получение предсказаний (Model Building)**
   - Обучение модели RandomForestRegressor
   - Оценка качества модели
   - Получение предсказаний

## Библиотеки

- Pandas, NumPy
- Scikit-learn
- NLTK (для обработки естественного языка)
- Matplotlib, Seaborn (для визуализации)
- WordCloud (для визуализации текстовых данных)

## Результаты

Модель демонстрирует хорошую точность в предсказании рейтингов отелей. Основные метрики качества:
- Средняя абсолютная процентная ошибка (MAPE): 12.130274707035321

Наиболее важные признаки для предсказания:
1. review_positive_word_proportion
2. p_review_sentiments_compound
3. average_score 
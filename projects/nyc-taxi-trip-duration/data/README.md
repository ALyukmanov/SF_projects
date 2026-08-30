# Данные — длительность поездки такси в Нью-Йорке

## Источник

Датасет соревнования Kaggle **«New York City Taxi Trip Duration»**,
дополненный тремя вспомогательными источниками. Файлы не входят в
репозиторий (~900 МБ суммарно).

## Файлы

| Файл | Строк/размер | Роль | Схема (ключевые столбцы) |
|---|---|---|---|
| `train.csv` | ~1,46 млн строк, 191 МБ | Основные записи о поездках | `id`, `vendor_id`, `pickup_datetime`, `dropoff_datetime`, `passenger_count`, `pickup_longitude/latitude`, `dropoff_longitude/latitude`, `store_and_fwd_flag`, `trip_duration` (таргет, секунды) |
| `osrm_data_train.csv` | 702 МБ | Маршрут/расстояние от движка OSRM (Open Source Routing Machine) | `id`, `starting_street`, `end_street`, `total_distance`, `total_travel_time`, `number_of_steps` и детали по шагам маршрута |
| `weather_data.csv` | 823 КБ | Почасовая погода в Нью-Йорке | `time`, `temperature`, `windchill`, `humidity`, `pressure`, `dew Point`, `visibility`, `wind dir`, `wind speed`, `precip`, `conditions`, `date`, `hour` |
| `holiday_data.csv` | ~500 байт, разделитель `;` | Календарь праздников США на период поездок | `day`, `date`, `holiday` |

`test_data.csv` и `osrm_data_test.csv` нужны только для формирования
сабмита на Kaggle; метрики train/validation основаны только на
`train.csv`.

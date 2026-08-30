# artifacts/

Здесь появляются модель и её метаданные после обучения. В git не хранятся (см. `.gitignore`).

- `model.joblib` — обученный `Pipeline` (StandardScaler + RandomForestClassifier).
- `model_metadata.json` — метрики и метаданные текущей версии модели (схема — в README проекта).

Создать их можно любым способом:

```bash
python -m app.training
# или, пока API работает:
curl -X POST http://localhost:8000/retrain
# или через Docker Compose: ml-api обучает модель сам при первом запуске, если папка пустая.
```

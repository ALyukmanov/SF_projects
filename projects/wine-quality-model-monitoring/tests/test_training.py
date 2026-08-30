import json

import pandas as pd
import pytest

from app import config
from app.training import (
    DatasetValidationError,
    build_binary_target,
    load_dataset,
    save_model,
    train_and_save,
    train_model,
)


def test_load_dataset_success():
    df = load_dataset(config.DATASET_PATH)
    assert len(df) > 0
    for column in config.REQUIRED_COLUMNS:
        assert column in df.columns


def test_load_dataset_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("a;b;c\n1;2;3\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_dataset(bad_csv)


def test_load_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does-not-exist.csv")


def test_build_binary_target_threshold():
    df = pd.DataFrame({"quality": [3, 5, 6, 8]})
    for column in config.FEATURE_NAMES:
        df[column] = 0.0
    result = build_binary_target(df)
    assert list(result["is_good"]) == [0, 0, 1, 1]


def test_train_model_produces_reasonable_metrics():
    result = train_model()
    metadata = result.metadata
    assert metadata["dataset_rows"] > 0
    assert metadata["train_rows"] + metadata["test_rows"] == metadata["dataset_rows"]
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        assert 0.0 <= metadata[key] <= 1.0
    assert metadata["feature_names"] == config.FEATURE_NAMES
    assert metadata["positive_class_threshold"] == config.POSITIVE_CLASS_THRESHOLD


def test_save_model_creates_artifacts(isolated_artifacts):
    result = train_model()
    save_model(result)

    assert config.MODEL_PATH.exists()
    assert config.METADATA_PATH.exists()
    assert not config.MODEL_TMP_PATH.exists()
    assert not config.METADATA_TMP_PATH.exists()

    saved_metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    assert saved_metadata["model_version"] == result.metadata["model_version"]


def test_train_and_save_predicts_correctly(isolated_artifacts):
    import joblib

    metadata = train_and_save()
    model = joblib.load(config.MODEL_PATH)

    df = load_dataset(config.DATASET_PATH)
    sample = df[config.FEATURE_NAMES].iloc[[0]]
    prediction = model.predict(sample)
    probability = model.predict_proba(sample)[:, 1]

    assert prediction[0] in (0, 1)
    assert 0.0 <= probability[0] <= 1.0
    assert metadata["model_version"]


def test_model_version_changes_between_trainings(isolated_artifacts):
    first = train_and_save()
    second = train_and_save()
    assert first["model_version"] != second["model_version"]


def test_failed_save_leaves_previous_model_artifacts_untouched(isolated_artifacts, monkeypatch):
    """Ошибка записи новой модели не должна портить старую — это гарантия
    атомарной замены (os.replace), отдельная от блокировки конкурентного
    retrain (см. tests/test_api.py)."""
    import app.training as training_module

    baseline_metadata = train_and_save()
    baseline_model_bytes = config.MODEL_PATH.read_bytes()
    baseline_metadata_text = config.METADATA_PATH.read_text(encoding="utf-8")

    def failing_dump(*args, **kwargs):
        raise RuntimeError("simulated failure while writing the new model")

    monkeypatch.setattr(training_module.joblib, "dump", failing_dump)

    with pytest.raises(RuntimeError):
        train_and_save()

    assert config.MODEL_PATH.read_bytes() == baseline_model_bytes
    assert config.METADATA_PATH.read_text(encoding="utf-8") == baseline_metadata_text

    saved_metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    assert saved_metadata["model_version"] == baseline_metadata["model_version"]

    # После сорванной записи временных файлов не осталось.
    assert not config.MODEL_TMP_PATH.exists()
    assert not config.METADATA_TMP_PATH.exists()

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # noqa: E402


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    """Перенаправляет пути артефактов во временную папку, чтобы тесты
    не трогали настоящую artifacts/."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(config, "MODEL_PATH", artifacts_dir / "model.joblib")
    monkeypatch.setattr(config, "MODEL_TMP_PATH", artifacts_dir / "model.joblib.tmp")
    monkeypatch.setattr(config, "METADATA_PATH", artifacts_dir / "model_metadata.json")
    monkeypatch.setattr(config, "METADATA_TMP_PATH", artifacts_dir / "model_metadata.json.tmp")
    return artifacts_dir

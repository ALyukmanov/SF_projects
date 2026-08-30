"""
Explicit, fail-closed model promotion.

Training (scripts/run_model_training_real.py) always saves a candidate
artefact to models/*.pkl. Only this script may flip models/current_model.json
to point at a new artefact -- there is no other path in this repo that writes
that file for a *real*-data model except an explicit run of this command.
That closes an earlier auto-promotion bug (ModelTrainer.save()
used to default set_as_current=True, so a routine training run could
silently overwrite the reviewed production pointer).

Protocol: train -> evaluate -> candidate artefact -> validation report ->
explicit promote. This script implements the last two steps. Every check
below must pass before current_model.json is touched; any failure exits
with status 2 and writes nothing (fail closed).

Usage:
    python scripts/promote_model.py --candidate models/xgboost_20260827_091858.pkl
    python scripts/promote_model.py --candidate models/xgboost_....pkl --dry-run
    python scripts/promote_model.py --candidate models/....pkl --allow-synthetic-promotion
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import pandas as pd

from src.features.feature_engineering import FEATURE_SCHEMA_VERSION, FeatureEngineer
from src.preprocessing.economic_features import EconomicFeatureEngineer
from src.utils.logger import get_logger

logger = get_logger("promote_model")

_MODELS_DIR = _PROJECT_ROOT / "models"
_MANIFEST_PATH = _MODELS_DIR / "current_model.json"
_PROMOTION_LOG_PATH = _PROJECT_ROOT / "reports" / "promotion_log.jsonl"

_REQUIRED_METADATA_KEYS = (
    "model_type",
    "trained_at",
    "feature_names",
    "metrics",
    "feature_schema_version",
    "is_synthetic",
    "data_source",
)
_REQUIRED_METRIC_KEYS = ("mae", "rmse", "r2")


class ValidationError(Exception):
    """Raised when a candidate fails a promotion check. Caught in main()."""


def _expected_feature_names() -> List[str]:
    """Derive the live feature list by replaying the real training pipeline's
    feature-construction steps (economic enrichment, then feature
    engineering) on a tiny synthetic frame.

    This must mirror scripts/run_model_training_real.py's actual sequence --
    EconomicFeatureEngineer.add_economic_features() runs BEFORE
    FeatureEngineer.prepare_for_training(), because prepare_for_training()
    only *selects* whichever economic columns already exist in the frame
    (via _select_available_features); it does not add them itself. Skipping
    that step here previously made every real candidate look like a feature
    mismatch (23 vs 27 columns) purely from a validator bug, not an actual
    schema problem.
    """
    sample = pd.DataFrame(
        {
            "price": [10_000_000.0, 12_000_000.0],
            "total_area": [50.0, 60.0],
            "rooms": [1, 2],
            "floor": [3, 5],
            "floors_total": [9, 12],
            "city": ["Москва", "Санкт-Петербург"],
            "building_type": ["panel", "brick"],
            "year_built": [None, None],
            "source_category": ["1_rooms_flats_sale", "2_rooms_flats_sale"],
        }
    )
    eco_eng = EconomicFeatureEngineer(use_api=False)
    sample = eco_eng.add_economic_features(sample)

    engineer = FeatureEngineer()
    X, _ = engineer.prepare_for_training(sample)
    return list(X.columns)


def _load_manifest(manifest_path: Path = _MANIFEST_PATH) -> Dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def validate_candidate(
    candidate_path: Path,
    allow_synthetic_promotion: bool,
    manifest_path: Path = _MANIFEST_PATH,
) -> Tuple[Dict[str, Any], List[str]]:
    """Run every promotion check. Raises ValidationError on the first failure.

    Returns (artefact, warnings) on success. Warnings are non-fatal notes
    surfaced to the human reviewer (e.g. a metric that looks worse than the
    currently-promoted model), not promotion blockers.

    ``manifest_path`` defaults to the real ``models/current_model.json`` but
    is overridable so tests can validate against an empty/sandboxed manifest
    instead of whatever happens to be currently promoted on disk.
    """
    warnings: List[str] = []

    if not candidate_path.exists():
        raise ValidationError(f"Candidate artefact does not exist: {candidate_path}")
    if candidate_path.suffix != ".pkl":
        raise ValidationError(f"Candidate must be a .pkl artefact, got: {candidate_path.suffix}")

    try:
        # joblib.load() on a locally-produced artefact from models/ (this repo's own
        # training pipeline output, same trust boundary as trainer.py/predictor.py) --
        # not deserializing data from an external/untrusted source.
        artefact = joblib.load(candidate_path)
    except Exception as exc:  # noqa: BLE001 - any load failure must fail closed
        raise ValidationError(f"Candidate artefact failed to load: {exc}") from exc

    if not isinstance(artefact, dict):
        raise ValidationError("Candidate artefact is not the expected dict format.")

    model = artefact.get("model")
    if model is None or not hasattr(model, "predict"):
        raise ValidationError("Candidate artefact has no usable 'model' with .predict().")

    missing_keys = [k for k in _REQUIRED_METADATA_KEYS if artefact.get(k) is None]
    if missing_keys:
        raise ValidationError(f"Candidate metadata missing required fields: {missing_keys}")

    schema_version = artefact["feature_schema_version"]
    if schema_version != FEATURE_SCHEMA_VERSION:
        raise ValidationError(
            f"Candidate feature_schema_version={schema_version!r} does not match "
            f"the live FeatureEngineer's FEATURE_SCHEMA_VERSION={FEATURE_SCHEMA_VERSION!r}. "
            "Promoting would risk feeding the API's live feature vector into a model "
            "trained on a different schema."
        )

    feature_names = artefact["feature_names"]
    if not isinstance(feature_names, list) or not feature_names:
        raise ValidationError("Candidate feature_names is empty or not a list.")
    if len(feature_names) != len(set(feature_names)):
        raise ValidationError("Candidate feature_names contains duplicates.")

    expected_features = _expected_feature_names()
    if feature_names != expected_features:
        raise ValidationError(
            "Candidate feature_names does not match the live FeatureEngineer output "
            f"(order-sensitive). Candidate has {len(feature_names)} features, live "
            f"schema has {len(expected_features)}. Candidate={feature_names} "
            f"Expected={expected_features}"
        )

    metrics = artefact["metrics"]
    if not isinstance(metrics, dict):
        raise ValidationError("Candidate metrics is not a dict.")
    missing_metrics = [k for k in _REQUIRED_METRIC_KEYS if k not in metrics]
    if missing_metrics:
        raise ValidationError(f"Candidate metrics missing required keys: {missing_metrics}")
    for key in _REQUIRED_METRIC_KEYS:
        value = metrics[key]
        if value is None or not isinstance(value, (int, float)) or value != value:  # NaN check
            raise ValidationError(f"Candidate metric '{key}' is missing/NaN/non-numeric: {value!r}")
    if metrics["mae"] <= 0 or metrics["rmse"] <= 0:
        raise ValidationError(
            f"Candidate mae/rmse must be positive, got mae={metrics['mae']}, rmse={metrics['rmse']}."
        )
    if not (-5.0 <= metrics["r2"] <= 1.0):
        raise ValidationError(
            f"Candidate r2={metrics['r2']} is outside a sane range [-5, 1] -- likely a broken run."
        )

    dataset_path = artefact.get("dataset_path")
    if dataset_path:
        dataset_path_str = str(dataset_path)
        if (
            Path(dataset_path_str).is_absolute()
            or "Users" in dataset_path_str
            or "home" in dataset_path_str.lower()
        ):
            raise ValidationError(
                f"Candidate dataset_path looks absolute/leaks a local path: {dataset_path_str!r}. "
                "Training metadata must store a repo-relative path."
            )
    else:
        warnings.append("Candidate has no dataset_path recorded -- provenance is incomplete.")

    is_synthetic = artefact["is_synthetic"]
    if is_synthetic and not allow_synthetic_promotion:
        raise ValidationError(
            "Candidate is tagged is_synthetic=True. Promoting a synthetic model over a "
            "reviewed real-data (or other synthetic) production pointer requires the "
            "explicit --allow-synthetic-promotion flag, so this can't happen by accident."
        )

    current = _load_manifest(manifest_path)
    if current:
        if current.get("filename") == candidate_path.name:
            warnings.append(
                "Candidate is already the current promoted model -- this is a no-op re-promotion."
            )
        current_metrics = current.get("metrics") or {}
        for key in ("mae", "rmse", "r2"):
            cur_val = current_metrics.get(key)
            new_val = metrics.get(key)
            if cur_val is not None and new_val is not None:
                worse = (new_val > cur_val) if key in ("mae", "rmse") else (new_val < cur_val)
                if worse:
                    warnings.append(
                        f"Candidate {key}={new_val} is worse than current promoted "
                        f"{key}={cur_val} ({current.get('filename')})."
                    )
        if bool(current.get("is_synthetic")) != bool(is_synthetic):
            warnings.append(
                f"Provenance change: current is_synthetic={current.get('is_synthetic')} -> "
                f"candidate is_synthetic={is_synthetic}. This is exactly the synthetic-to-real "
                "transition this mechanism exists to make explicit and reviewed, not silent."
            )

    return artefact, warnings


def write_manifest(candidate_path: Path, artefact: Dict[str, Any]) -> Dict[str, Any]:
    previous = _load_manifest()
    manifest = {
        "filename": candidate_path.name,
        "model_type": artefact.get("model_type"),
        "trained_at": artefact.get("trained_at"),
        "is_synthetic": artefact.get("is_synthetic"),
        "data_source": artefact.get("data_source"),
        "feature_schema_version": artefact.get("feature_schema_version"),
        "split_strategy": artefact.get("split_strategy"),
        "dataset_path": artefact.get("dataset_path"),
        "hyperparameters": artefact.get("hyperparameters"),
        "tuned_from_study": artefact.get("tuned_from_study"),
        "metrics": artefact.get("metrics"),
        "artifact_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "set_current_at": datetime.now().isoformat(),
        "promoted_by": "scripts/promote_model.py",
        "promoted_from_git_revision": _git_revision(),
        "previous_current_filename": previous.get("filename"),
    }
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info("Promoted %s -> %s", candidate_path.name, _MANIFEST_PATH)
    return manifest


def append_promotion_log(manifest: Dict[str, Any], warnings: List[str], dry_run: bool) -> None:
    _PROMOTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "warnings": warnings,
        **{
            k: manifest.get(k)
            for k in (
                "filename",
                "model_type",
                "is_synthetic",
                "data_source",
                "metrics",
                "previous_current_filename",
                "promoted_from_git_revision",
            )
        },
    }
    with _PROMOTION_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--candidate", required=True, help="Path to the candidate .pkl artefact (in models/)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all validation checks and print the report, but do not write current_model.json.",
    )
    parser.add_argument(
        "--allow-synthetic-promotion",
        action="store_true",
        help="Required to promote a candidate tagged is_synthetic=True (extra guard).",
    )
    args = parser.parse_args()

    candidate_path = Path(args.candidate)
    if not candidate_path.is_absolute():
        candidate_path = _PROJECT_ROOT / candidate_path

    print("=" * 60)
    print("MODEL PROMOTION")
    print("=" * 60)
    print(f"Candidate : {candidate_path}")

    try:
        artefact, warnings = validate_candidate(candidate_path, args.allow_synthetic_promotion)
    except ValidationError as exc:
        print(f"\nVALIDATION FAILED -- refusing to promote (fail closed).\nReason: {exc}")
        return 2

    print("\nAll validation checks passed.")
    if warnings:
        print("\nWarnings (non-fatal, for human review):")
        for w in warnings:
            print(f"  - {w}")

    print("\nCandidate summary:")
    print(f"  model_type   : {artefact.get('model_type')}")
    print(f"  is_synthetic : {artefact.get('is_synthetic')}")
    print(f"  data_source  : {artefact.get('data_source')}")
    print(f"  metrics      : {artefact.get('metrics')}")
    print(f"  split        : {artefact.get('split_strategy')}")

    if args.dry_run:
        print("\n--dry-run set: current_model.json NOT modified.")
        return 0

    manifest = write_manifest(candidate_path, artefact)
    append_promotion_log(manifest, warnings, dry_run=False)
    print(f"\nPromoted. current_model.json now points at: {manifest['filename']}")
    print(f"Previous artefact: {manifest['previous_current_filename']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

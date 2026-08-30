"""
Model training module for Real Estate Price Prediction.

Supports XGBoost, LightGBM, CatBoost and scikit-learn RandomForest.
Handles train/test split, cross-validation, metric computation, and
artefact serialisation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Supported model types
# ---------------------------------------------------------------------------

_MODEL_TYPES = ("xgboost", "random_forest", "lightgbm", "catboost", "linear_regression", "ridge")


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (ignores zero targets)."""
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_residual_quantile_interval(
    y_true: np.ndarray, y_pred: np.ndarray, coverage: float = 0.90
) -> Dict[str, Any]:
    """Estimate an additive prediction interval from held-out residuals.

    This implements the "residual quantiles on a calibration/holdout set"
    approach: additive RUB offsets are taken from the empirical quantiles of
    ``y_true - y_pred`` so that, roughly, ``coverage`` fraction of the SAME
    holdout points fall inside ``[y_pred + lower_offset, y_pred + upper_offset]``.

    Honesty note (documented, not hidden): this is calibrated on the model's
    own evaluation holdout, not a separate third calibration split — the
    dataset (≈200 synthetic rows) is too small to support a clean
    train/calibration/holdout three-way split without each becoming
    unreasonably tiny. ``observed_coverage`` below is therefore an in-sample
    (holdout-fit) coverage estimate, not an out-of-sample conformal
    guarantee. This is called an "estimated range" (оценочный диапазон), not
    a statistical confidence interval, throughout the API/UI for this reason.

    Args:
        y_true: True target values (RUB) on the holdout set.
        y_pred: Point predictions (RUB) on the same holdout set.
        coverage: Target two-sided coverage, e.g. 0.90 for a 90% range.

    Returns:
        Dict with ``method``, ``coverage_target``, ``lower_offset``,
        ``upper_offset`` (RUB, signed), ``observed_coverage``, ``n_holdout``.
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    alpha = 1.0 - coverage
    lower_offset = float(np.quantile(residuals, alpha / 2))
    upper_offset = float(np.quantile(residuals, 1 - alpha / 2))

    covered = (residuals >= lower_offset) & (residuals <= upper_offset)
    observed_coverage = float(np.mean(covered))

    return {
        "method": "residual_quantile_holdout",
        "coverage_target": coverage,
        "lower_offset": lower_offset,
        "upper_offset": upper_offset,
        "observed_coverage": observed_coverage,
        "n_holdout": int(len(residuals)),
        "note": (
            "Calibrated on the evaluation holdout set, not an independent "
            "calibration split (dataset too small for a 3-way split). "
            "Report as an estimated range (оценочный диапазон), not a "
            "statistical confidence interval."
        ),
    }


def _build_model(model_type: str, params: Dict[str, Any]):
    """Instantiate the requested model with *params*."""
    mt = model_type.lower()
    if mt == "xgboost":
        try:
            from xgboost import XGBRegressor

            defaults = dict(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            )
            defaults.update(params)
            return XGBRegressor(**defaults)
        except ImportError:
            raise ImportError("xgboost is not installed.  Run: pip install xgboost")

    if mt == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            defaults = dict(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            defaults.update(params)
            return LGBMRegressor(**defaults)
        except ImportError:
            raise ImportError("lightgbm is not installed.  Run: pip install lightgbm")

    if mt == "catboost":
        try:
            from catboost import CatBoostRegressor

            defaults = dict(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                random_seed=42,
                verbose=0,
            )
            defaults.update(params)
            return CatBoostRegressor(**defaults)
        except ImportError:
            raise ImportError("catboost is not installed.  Run: pip install catboost")

    if mt == "random_forest":
        defaults = dict(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        defaults.update(params)
        return RandomForestRegressor(**defaults)

    if mt == "linear_regression":
        # No hyperparameters to speak of; accepts **params for interface
        # consistency but LinearRegression has nothing meaningful to tune here.
        from sklearn.linear_model import LinearRegression

        return LinearRegression(**params)

    if mt == "ridge":
        defaults = dict(alpha=1.0, random_state=42)
        defaults.update(params)
        from sklearn.linear_model import Ridge

        return Ridge(**defaults)

    raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {_MODEL_TYPES}")


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------


class ModelTrainer:
    """Train, evaluate, save and load a price-prediction model.

    Args:
        model_type: One of ``'xgboost'``, ``'random_forest'``, ``'lightgbm'``,
                    ``'catboost'`` (default: ``'xgboost'``).
        config:     Optional dict with keys:
                    - ``params``    — model hyper-parameters
                    - ``test_size`` — train/test split ratio (default: 0.2)
                    - ``cv_folds``  — number of CV folds (default: 5)
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_type = model_type.lower()
        self._config = config or {}
        self._params: Dict[str, Any] = self._config.get("params", {})
        self._test_size: float = float(self._config.get("test_size", 0.2))
        self._cv_folds: int = int(self._config.get("cv_folds", 5))

        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self._train_metrics: Dict[str, float] = {}
        self._trained_at: Optional[str] = None
        self._feature_names: List[str] = []
        self._prediction_interval: Dict[str, Any] = {}

        logger.info("ModelTrainer initialised | model_type=%s", self.model_type)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Train the model on *(X, y)* and return evaluation metrics.

        The data is split 80/20.  ``y`` is expected to be log-transformed
        price (as produced by :class:`~src.features.feature_engineering.FeatureEngineer`).

        Args:
            X: Feature matrix (numeric, no NaNs expected).
            y: Target Series (log-transformed price).
            feature_names: Optional list of feature column names to persist in
                           the artefact for consistent inference.  Defaults to
                           ``list(X.columns)``.

        Returns:
            Dict with keys ``mae``, ``rmse``, ``r2``, ``mape`` on the test set.
        """
        self._feature_names = list(feature_names or X.columns)
        logger.info("Training %s on %d samples, %d features.", self.model_type, len(X), X.shape[1])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self._test_size, random_state=42
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = _build_model(self.model_type, self._params)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate on test set (back-transform from log space)
        y_pred_log = self.model.predict(X_test_scaled)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test.values)

        metrics = self._compute_metrics(y_true, y_pred)
        self._train_metrics = metrics
        self._trained_at = datetime.now().isoformat()
        self._prediction_interval = compute_residual_quantile_interval(y_true, y_pred)

        logger.info(
            "Training done. MAE=%.0f RUB | RMSE=%.0f RUB | R2=%.4f | MAPE=%.2f%%",
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            metrics["mape"],
        )
        return metrics

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Train on a pre-split dataset and return test-set metrics.

        Unlike :meth:`train`, this method does **not** perform an internal
        train/test split — it expects the caller to provide the split.
        This is the recommended approach for reproducible evaluation.

        Args:
            X_train: Training feature matrix.
            y_train: Training target (log-transformed price).
            X_test:  Test feature matrix (never seen during training).
            y_test:  Test target (log-transformed price).
            feature_names: Feature column names to store in the artefact.

        Returns:
            Dict with keys ``mae``, ``rmse``, ``r2``, ``mape``, ``median_ae``
            evaluated on ``(X_test, y_test)``.
        """
        self._feature_names = list(feature_names or X_train.columns)
        logger.info(
            "Fitting %s on %d train / %d test samples, %d features.",
            self.model_type,
            len(X_train),
            len(X_test),
            X_train.shape[1],
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = _build_model(self.model_type, self._params)
        self.model.fit(X_train_scaled, y_train)

        y_pred_log = self.model.predict(X_test_scaled)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test.values)

        metrics = self._compute_metrics(y_true, y_pred)
        self._train_metrics = metrics
        self._trained_at = datetime.now().isoformat()
        self._prediction_interval = compute_residual_quantile_interval(y_true, y_pred)

        logger.info(
            "Fit done. MAE=%.0f RUB | RMSE=%.0f RUB | R2=%.4f | MAPE=%.2f%%",
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            metrics["mape"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """Run *cv*-fold cross-validation and return aggregated scores.

        Args:
            X:  Feature matrix.
            y:  Target Series (log-transformed price).
            cv: Number of folds (default: 5).

        Returns:
            Dict with ``r2_mean``, ``r2_std``, ``neg_mae_mean``, ``neg_mae_std``.
        """
        model = _build_model(self.model_type, self._params)
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        r2_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="r2", n_jobs=-1)
        mae_scores = cross_val_score(
            model, X_scaled, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
        )

        results = {
            "r2_mean": float(r2_scores.mean()),
            "r2_std": float(r2_scores.std()),
            "neg_mae_mean": float(mae_scores.mean()),
            "neg_mae_std": float(mae_scores.std()),
            "cv_folds": cv,
        }
        logger.info(
            "Cross-validation (%d folds): R2=%.4f +/- %.4f | MAE=%.0f +/- %.0f",
            cv,
            results["r2_mean"],
            results["r2_std"],
            abs(results["neg_mae_mean"]),
            results["neg_mae_std"],
        )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        tag: Optional[str] = None,
        set_as_current: bool = True,
    ) -> str:
        """Serialise model and scaler artefacts into *path* directory.

        Filename format: ``{model_type}[_{tag}]_{YYYYMMDD_HHMMSS}.pkl``. Pass
        ``tag="synthetic"`` when the training data was synthetic/demo data so
        this is visible directly in the filename, not just inside the
        artefact metadata.

        Args:
            path: Directory path (created if absent).
            metadata: Extra provenance fields to embed in the artefact, e.g.
                ``data_source``, ``is_synthetic``, ``dataset_path``,
                ``dataset_sha256``, ``dataset_rows``, ``dataset_columns``,
                ``training_start``, ``training_end``, ``split_strategy``,
                ``random_seed``, ``library_versions``, ``feature_schema_version``,
                ``git_commit``. See scripts/run_model_training_real.py for the
                fields actually populated by the training pipeline.
            tag: Optional short filename tag (e.g. ``"synthetic"``).
            set_as_current: If True (default), write/update
                ``{path}/current_model.json`` to point at this artefact, so
                ``Predictor.load()`` selects it explicitly rather than by
                filesystem mtime (see ``Predictor.load()`` docstring — mtime
                selection is still used as a fallback when no manifest is
                present, e.g. for artefacts saved before this feature existed).

        Returns:
            Full path to the saved model file.
        """
        if self.model is None:
            raise RuntimeError("No model trained yet. Call train() first.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_part = f"_{tag}" if tag else ""
        model_filename = f"{self.model_type}{tag_part}_{timestamp}.pkl"
        model_path = save_dir / model_filename

        artefact = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "trained_at": self._trained_at,
            "metrics": self._train_metrics,
            "params": self._params,
            "feature_names": self._feature_names,
            "prediction_interval": self._prediction_interval,
        }
        if metadata:
            artefact.update(metadata)
        joblib.dump(artefact, model_path)
        logger.info("Model saved -> %s", model_path)

        if set_as_current:
            self._write_current_model_manifest(save_dir, model_filename, artefact)

        return str(model_path)

    @staticmethod
    def _write_current_model_manifest(
        save_dir: Path, model_filename: str, artefact: Dict[str, Any]
    ) -> None:
        """Write ``current_model.json`` recording which artefact is "current".

        This exists so artefact selection doesn't rely on filesystem mtime
        (which is fragile — e.g. copying/restoring files can reorder mtimes).
        Only JSON-serialisable metadata is written (not the model/scaler
        objects themselves).
        """
        import hashlib
        import json

        model_path = save_dir / model_filename
        manifest = {
            "filename": model_filename,
            "model_type": artefact.get("model_type"),
            "trained_at": artefact.get("trained_at"),
            "is_synthetic": artefact.get("is_synthetic"),
            "data_source": artefact.get("data_source"),
            "feature_schema_version": artefact.get("feature_schema_version"),
            "metrics": artefact.get("metrics"),
            "artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "set_current_at": datetime.now().isoformat(),
        }
        manifest_path = save_dir / "current_model.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        logger.info("Updated %s -> %s", manifest_path, model_filename)

    def load(self, path: str) -> None:
        """Load a model artefact from *path* (file or directory).

        If *path* is a directory, loads the most recently modified ``.pkl``
        file in that directory.

        Args:
            path: Path to a ``.pkl`` file or a directory containing one.
        """
        load_path = Path(path)
        if load_path.is_dir():
            pkl_files = sorted(load_path.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
            if not pkl_files:
                raise FileNotFoundError(f"No .pkl files found in {load_path}")
            load_path = pkl_files[-1]

        logger.info("Loading model from %s", load_path)
        artefact = joblib.load(load_path)

        self.model = artefact["model"]
        self.scaler = artefact.get("scaler")
        self.model_type = artefact.get("model_type", self.model_type)
        self._trained_at = artefact.get("trained_at")
        self._train_metrics = artefact.get("metrics", {})
        self._feature_names = artefact.get("feature_names", [])
        self._prediction_interval = artefact.get("prediction_interval", {})
        logger.info("Model loaded. Type=%s | Trained at=%s", self.model_type, self._trained_at)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mape = _mape(y_true, y_pred)
        return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

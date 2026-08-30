"""
Model evaluation module for Real Estate Price Prediction.

Provides metric computation, feature importance extraction and
formatted reporting.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """Evaluate a trained model and produce interpretability artefacts.

    Designed to work with any scikit-learn–compatible estimator as well as
    XGBoost / LightGBM / CatBoost models that expose a ``predict`` method.
    """

    def __init__(self) -> None:
        logger.info("ModelEvaluator initialised.")

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        log_target: bool = True,
    ) -> Dict[str, float]:
        """Compute regression metrics for *model* on the held-out test set.

        Args:
            model:      Trained estimator with a ``predict(X)`` method.
            X_test:     Feature matrix for the test split.
            y_test:     True target values.  If *log_target* is ``True``
                        (default), ``y_test`` is assumed to be log-transformed
                        price and will be back-transformed via ``expm1`` before
                        computing metrics.
            log_target: Whether to back-transform predictions/targets from
                        log space (default: ``True``).

        Returns:
            Dict with keys: ``mae``, ``rmse``, ``r2``, ``mape``, ``median_ae``.
        """
        y_pred_raw = model.predict(X_test)

        if log_target:
            y_true = np.expm1(np.asarray(y_test, dtype=float))
            y_pred = np.expm1(np.asarray(y_pred_raw, dtype=float))
        else:
            y_true = np.asarray(y_test, dtype=float)
            y_pred = np.asarray(y_pred_raw, dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mape = _mape(y_true, y_pred)
        median_ae = float(np.median(np.abs(y_true - y_pred)))

        metrics: Dict[str, float] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "median_ae": median_ae,
        }

        logger.info(
            "Evaluation → MAE=%.0f | RMSE=%.0f | R²=%.4f | MAPE=%.2f%% | MedianAE=%.0f",
            mae,
            rmse,
            r2,
            mape,
            median_ae,
        )
        return metrics

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(
        self,
        model,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Extract and rank feature importances from *model*.

        Supports:
        - sklearn estimators with ``feature_importances_`` attribute
        - XGBoost ``XGBRegressor`` (uses ``get_score(importance_type='gain')``)
        - LightGBM ``LGBMRegressor`` (``feature_importances_``)
        - CatBoost ``CatBoostRegressor`` (``get_feature_importance()``)

        Args:
            model:         Trained estimator.
            feature_names: Ordered list of feature column names matching the
                           columns used during training.

        Returns:
            DataFrame with columns ``feature`` and ``importance``, sorted
            descending by importance.
        """
        importances: Optional[np.ndarray] = None

        # Try XGBoost-specific API first (gain-based importance)
        if hasattr(model, "get_booster"):
            try:
                score_dict = model.get_booster().get_score(importance_type="gain")
                # XGBoost may use f0, f1, ... keys when feature names are not set
                importances = np.array(
                    [
                        score_dict.get(f, score_dict.get(f"f{i}", 0.0))
                        for i, f in enumerate(feature_names)
                    ]
                )
                logger.debug("Feature importance extracted via XGBoost get_score (gain).")
            except Exception:
                pass

        # CatBoost
        if importances is None and hasattr(model, "get_feature_importance"):
            try:
                importances = np.array(model.get_feature_importance())
                logger.debug("Feature importance extracted via CatBoost get_feature_importance.")
            except Exception:
                pass

        # sklearn / LightGBM generic
        if importances is None and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            logger.debug("Feature importance extracted via feature_importances_ attribute.")

        if importances is None:
            logger.warning("Model does not expose feature importances.")
            return pd.DataFrame({"feature": feature_names, "importance": 0.0})

        if len(importances) != len(feature_names):
            logger.warning(
                "Feature importance length mismatch (%d vs %d feature names). "
                "Truncating / padding.",
                len(importances),
                len(feature_names),
            )
            min_len = min(len(importances), len(feature_names))
            importances = importances[:min_len]
            feature_names = feature_names[:min_len]

        df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return df

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, metrics: Dict[str, float]) -> None:
        """Print a formatted metrics table to stdout.

        Args:
            metrics: Dict produced by :meth:`evaluate`.
        """
        border = "=" * 52
        print(border)
        print(f"{'MODEL EVALUATION REPORT':^52}")
        print(border)
        metric_labels = {
            "mae": ("MAE", "RUB"),
            "rmse": ("RMSE", "RUB"),
            "r2": ("R2", ""),
            "mape": ("MAPE", "%"),
            "median_ae": ("Median AE", "RUB"),
        }
        for key, (label, unit) in metric_labels.items():
            if key in metrics:
                val = metrics[key]
                if key == "r2":
                    formatted = f"{val:.4f}"
                elif key == "mape":
                    formatted = f"{val:.2f}{unit}"
                else:
                    formatted = f"{val:,.0f} {unit}".strip()
                print(f"  {label:<20} {formatted:>20}")
        print(border)

        # Extra stats if available
        extra_keys = [k for k in metrics if k not in metric_labels]
        if extra_keys:
            print("  Additional metrics:")
            for k in extra_keys:
                print(f"    {k}: {metrics[k]}")
            print(border)

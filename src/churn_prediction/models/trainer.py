"""
Model training, hyper-parameter tuning, and persistence for the Churn
Prediction pipeline.

Supported estimators:
    * Logistic Regression (class-weight balanced)
    * XGBoost           (RandomizedSearchCV, scale_pos_weight)
    * LightGBM          (RandomizedSearchCV, is_unbalance)

Every training run is logged to **MLflow** (experiment *churn-prediction*).

Typical usage::

    from churn_prediction.models.trainer import ModelTrainer

    trainer = ModelTrainer(X_train, y_train, X_test, y_test)
    results = trainer.train_all()
    best_name, best_model, best_metrics = trainer.select_best_model()
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from ..utils.metrics import evaluate_model

# NOTE: `mlflow` and `lightgbm` are training-only dependencies and are imported
# lazily inside the methods that use them. This keeps the serving path (which
# imports this module transitively via the models package) free of heavy
# experiment-tracking and extra-estimator deps. See requirements-api.txt.

if TYPE_CHECKING:  # for type hints only; not imported at runtime
    from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)

# MLflow experiment name (shared across all runs in this module)
_MLFLOW_EXPERIMENT = "churn-prediction"


class ModelTrainer:
    """Train, tune, and evaluate churn-prediction models.

    Parameters
    ----------
    X_train, y_train:
        Training feature matrix and labels.
    X_test, y_test:
        Hold-out test feature matrix and labels.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
        random_state: int = 42,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state

        # Computed once and reused for XGBoost scale_pos_weight
        neg, pos = np.bincount(np.asarray(y_train, dtype=int))
        self._scale_pos_weight: float = neg / pos
        logger.info(
            "Class ratio — negative: %d, positive: %d, "
            "scale_pos_weight: %.3f",
            neg, pos, self._scale_pos_weight,
        )

        # Populated by train_* methods; maps model_name → (model, metrics)
        self._results: Dict[str, Tuple[Any, Dict[str, Any]]] = {}

        # Ensure the MLflow experiment exists (training-only dependency)
        import mlflow
        mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _log_to_mlflow(
        self,
        model_name: str,
        model: Any,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        """Log a single training run to MLflow."""
        import mlflow
        import mlflow.sklearn

        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("model_name", model_name)

            # Parameters ------------------------------------------------
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Scalar metrics --------------------------------------------
            _METRIC_KEYS = [
                "auc_roc", "precision", "recall", "f1",
                "accuracy", "optimal_threshold",
            ]
            for key in _METRIC_KEYS:
                if key in metrics:
                    mlflow.log_metric(key, metrics[key])

            # Model artifact --------------------------------------------
            trusted_types = [
                "collections.OrderedDict",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
                "lightgbm.core.Booster",
            ]
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                skops_trusted_types=trusted_types,
            )

        logger.info("MLflow run logged for '%s'.", model_name)

    # ------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------
    def train_logistic_regression(self) -> Tuple[LogisticRegression, Dict[str, Any]]:
        """Train a balanced Logistic Regression classifier.

        Returns
        -------
        tuple[LogisticRegression, dict]
            Fitted model and evaluation metrics dict.
        """
        logger.info("Training Logistic Regression ...")

        lr = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs",
        )
        lr.fit(self.X_train, self.y_train)

        metrics = evaluate_model(
            lr, self.X_test, self.y_test,
            model_name="Logistic Regression",
        )

        params = lr.get_params()
        self._log_to_mlflow("Logistic Regression", lr, params, metrics)
        self._results["Logistic Regression"] = (lr, metrics)

        return lr, metrics

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    def train_xgboost(
        self,
        n_iter: int = 20,
        cv_folds: int = 5,
    ) -> Tuple[XGBClassifier, Dict[str, Any]]:
        """Train an XGBoost classifier with RandomizedSearchCV.

        Parameters
        ----------
        n_iter:
            Number of random hyper-parameter combinations to sample.
        cv_folds:
            Number of stratified cross-validation folds.

        Returns
        -------
        tuple[XGBClassifier, dict]
            Best estimator and evaluation metrics dict.
        """
        logger.info("Training XGBoost (RandomizedSearchCV, n_iter=%d) ...", n_iter)

        base_xgb = XGBClassifier(
            scale_pos_weight=self._scale_pos_weight,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

        param_distributions: Dict[str, List[Any]] = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5],
        }

        search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv_folds,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(self.X_train, self.y_train)
        best_model: XGBClassifier = search.best_estimator_

        logger.info("XGBoost best params: %s", search.best_params_)

        metrics = evaluate_model(
            best_model, self.X_test, self.y_test,
            model_name="XGBoost",
        )

        self._log_to_mlflow("XGBoost", best_model, search.best_params_, metrics)
        self._results["XGBoost"] = (best_model, metrics)

        return best_model, metrics

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    def train_lightgbm(
        self,
        n_iter: int = 20,
        cv_folds: int = 5,
    ) -> Tuple[LGBMClassifier, Dict[str, Any]]:
        """Train a LightGBM classifier with RandomizedSearchCV.

        Parameters
        ----------
        n_iter:
            Number of random hyper-parameter combinations to sample.
        cv_folds:
            Number of stratified cross-validation folds.

        Returns
        -------
        tuple[LGBMClassifier, dict]
            Best estimator and evaluation metrics dict.
        """
        logger.info("Training LightGBM (RandomizedSearchCV, n_iter=%d) ...", n_iter)

        from lightgbm import LGBMClassifier

        base_lgbm = LGBMClassifier(
            is_unbalance=True,
            random_state=self.random_state,
            verbose=-1,
        )

        param_distributions: Dict[str, List[Any]] = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, -1],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [20, 31, 50],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        }

        search = RandomizedSearchCV(
            estimator=base_lgbm,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv_folds,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(self.X_train, self.y_train)
        best_model: LGBMClassifier = search.best_estimator_

        logger.info("LightGBM best params: %s", search.best_params_)

        metrics = evaluate_model(
            best_model, self.X_test, self.y_test,
            model_name="LightGBM",
        )

        self._log_to_mlflow("LightGBM", best_model, search.best_params_, metrics)
        self._results["LightGBM"] = (best_model, metrics)

        return best_model, metrics

    # ------------------------------------------------------------------
    # Train all
    # ------------------------------------------------------------------
    def train_all(self) -> Dict[str, Dict[str, Any]]:
        """Train all three model types and return their evaluation dicts.

        Returns
        -------
        dict
            ``{model_name: metrics_dict}`` for Logistic Regression,
            XGBoost, and LightGBM.
        """
        logger.info("=== Training all models ===")

        self.train_logistic_regression()
        self.train_xgboost()
        self.train_lightgbm()

        return {name: metrics for name, (_, metrics) in self._results.items()}

    # ------------------------------------------------------------------
    # Select best model
    # ------------------------------------------------------------------
    def select_best_model(self) -> Tuple[str, Any, Dict[str, Any]]:
        """Return the model with the highest AUC-ROC on the test set.

        Returns
        -------
        tuple[str, estimator, dict]
            ``(model_name, fitted_model, metrics_dict)`` for the winner.

        Raises
        ------
        RuntimeError
            If no models have been trained yet.
        """
        if not self._results:
            raise RuntimeError(
                "No models trained yet – call train_all() or an "
                "individual train_*() method first."
            )

        best_name = max(
            self._results,
            key=lambda name: self._results[name][1]["auc_roc"],
        )
        best_model, best_metrics = self._results[best_name]

        logger.info(
            "Best model: %s with AUC-ROC = %.4f",
            best_name, best_metrics["auc_roc"],
        )

        return best_name, best_model, best_metrics

    @property
    def trained_models(self) -> Dict[str, Any]:
        """Return a ``{name: model}`` dict for all trained models."""
        return {name: model for name, (model, _) in self._results.items()}


# =========================================================================
# Model artifact persistence
# =========================================================================

def save_model_artifacts(
    model: Any,
    preprocessor: Any,
    feature_names: List[str],
    metrics: Dict[str, Any],
    output_dir: str = "models/",
) -> Path:
    """Persist the trained model, preprocessor, feature list, and metadata.

    Saved artefacts
    ---------------
    * ``model.joblib``            – serialised estimator
    * ``preprocessor.joblib``     – serialised preprocessing pipeline
    * ``feature_names.json``      – ordered feature name list
    * ``model_metadata.json``     – metrics, training date, dataset hash,
                                    model type

    Parameters
    ----------
    model:
        Fitted estimator.
    preprocessor:
        Fitted sklearn ``Pipeline`` / ``ColumnTransformer``.
    feature_names:
        Ordered list of feature column names.
    metrics:
        Evaluation metrics dict (from ``evaluate_model``).
    output_dir:
        Target directory; created if it does not exist.

    Returns
    -------
    pathlib.Path
        Resolved path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Model
    model_path = out / "model.joblib"
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)

    # 2. Preprocessor
    preprocessor_path = out / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info("Preprocessor saved to %s", preprocessor_path)

    # 3. Feature names
    feature_names_path = out / "feature_names.json"
    with open(feature_names_path, "w", encoding="utf-8") as fh:
        json.dump(feature_names, fh, indent=2)
    logger.info("Feature names saved to %s", feature_names_path)

    # 4. Metadata
    # Build a serialisable copy of the metrics (drop numpy / ndarray types).
    serialisable_metrics: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serialisable_metrics[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serialisable_metrics[key] = value.item()
        else:
            serialisable_metrics[key] = value

    # Compute a deterministic hash of the feature list as a proxy for
    # "which dataset was used".
    dataset_hash = hashlib.sha256(
        json.dumps(feature_names, sort_keys=True).encode()
    ).hexdigest()[:12]

    metadata = {
        "model_type": type(model).__name__,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": dataset_hash,
        "metrics": serialisable_metrics,
    }

    metadata_path = out / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    logger.info("Metadata saved to %s", metadata_path)

    return out.resolve()

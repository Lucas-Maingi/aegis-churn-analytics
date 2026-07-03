"""
Model training and explainability modules for the Churn Prediction pipeline.

Exports:
    ModelTrainer: Trains, tunes, and evaluates LR / XGBoost / LightGBM models.
    save_model_artifacts: Persists model, preprocessor, and metadata to disk.
    ChurnExplainer: SHAP-based model explanation engine.
    generate_prediction_explanation: End-to-end raw-input → explanation pipeline.
"""

from .explainer import ChurnExplainer, generate_prediction_explanation
from .trainer import ModelTrainer, save_model_artifacts

__all__ = [
    "ModelTrainer",
    "save_model_artifacts",
    "ChurnExplainer",
    "generate_prediction_explanation",
]

"""
Churn Prediction — End-to-End Model Training Pipeline
======================================================
This script loads the raw dataset, performs cleaning, executes feature engineering,
splits the data, trains and tunes multiple candidate models (Logistic Regression,
XGBoost, LightGBM), compares their performance, selects the best model,
and persists the model artifacts for API deployment.
"""

import logging
import sys
from sklearn.model_selection import train_test_split

from churn_prediction import config
from churn_prediction.data.loader import load_telco_data, get_feature_target_split
from churn_prediction.data.preprocessor import ChurnPreprocessor
from churn_prediction.models.trainer import ModelTrainer, save_model_artifacts

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def run_training_pipeline() -> None:
    """Execute the end-to-end training and artifact generation pipeline."""
    logger.info("Starting churn prediction training pipeline...")

    # 1. Load and clean data
    logger.info("Loading IBM Telco Customer Churn dataset...")
    df_raw = load_telco_data()
    logger.info("Dataset loaded successfully. Shape: %s", df_raw.shape)

    # 2. Split features and target
    X_raw, y_raw = get_feature_target_split(df_raw)
    logger.info("Features and target split completed.")

    # 3. Stratified Train-Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=config.TEST_SIZE,
        stratify=y_raw,
        random_state=config.RANDOM_STATE,
    )
    logger.info(
        "Train-Test Split completed: Train shape=%s, Test shape=%s",
        X_train_raw.shape,
        X_test_raw.shape,
    )

    # 4. Feature Engineering & Preprocessing
    logger.info("Initializing and fitting preprocessor pipeline...")
    preprocessor = ChurnPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    feature_names = preprocessor.get_feature_names()
    logger.info(
        "Preprocessing completed. Feature count: %d. Preprocessed train shape: %s",
        len(feature_names),
        X_train.shape,
    )

    # 5. Model Training & Comparison
    logger.info("Initializing ModelTrainer and executing candidate training...")
    trainer = ModelTrainer(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=config.RANDOM_STATE,
    )

    # Train all models
    results = trainer.train_all()
    logger.info("Training of candidate models completed.")
    
    # Log results summary
    logger.info("--- Candidate Model Evaluation Summary ---")
    for name, metrics in results.items():
        logger.info(
            "%s: AUC-ROC=%.4f, F1-Score=%.4f, Precision=%.4f, Recall=%.4f, Optimal Threshold=%.3f",
            name,
            metrics["auc_roc"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            metrics["optimal_threshold"],
        )

    # 6. Select Winning Model
    best_name, best_model, best_metrics = trainer.select_best_model()
    logger.info(
        "Winning Model Selected: %s (AUC-ROC = %.4f)",
        best_name,
        best_metrics["auc_roc"],
    )

    # 7. Persist Artifacts
    logger.info("Saving model artifacts to %s...", config.MODELS_DIR)
    save_model_artifacts(
        model=best_model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        metrics=best_metrics,
        output_dir=str(config.MODELS_DIR),
    )
    logger.info("Model training pipeline executed and saved successfully!")

if __name__ == "__main__":
    run_training_pipeline()

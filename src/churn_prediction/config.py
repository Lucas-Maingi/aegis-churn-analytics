"""
Churn Prediction — Configuration Module
========================================
Centralized configuration for the entire project.
All paths, constants, and model parameters are defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# PATHS
# ============================================================================

# Project root (two levels up from this file: src/churn_prediction/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model artifact paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "best_model.joblib")))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", str(MODELS_DIR / "preprocessor.joblib")))
FEATURE_NAMES_PATH = Path(os.getenv("FEATURE_NAMES_PATH", str(MODELS_DIR / "feature_names.json")))
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ============================================================================
# DATASET
# ============================================================================

# IBM Telco Customer Churn dataset URLs (fallback chain)
DATASET_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Telco-Customer-Churn.csv",
]
DATASET_FILENAME = "Telco-Customer-Churn.csv"

# Expected columns in the raw dataset
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

# Target variable
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "PaperlessBilling",
]

MULTI_CLASS_CATEGORICAL_FEATURES = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

# Service columns for num_services feature
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Protection bundle columns
PROTECTION_COLUMNS = ["OnlineSecurity", "TechSupport", "DeviceProtection"]

# Tenure bucket definitions
TENURE_BINS = [0, 12, 24, 48, 72]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]

# ============================================================================
# MODEL TRAINING
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
RANDOMIZED_SEARCH_N_ITER = 20

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")

# ============================================================================
# RISK TIERS
# ============================================================================

# Churn probability thresholds for risk tiers
RISK_THRESHOLDS = {
    "HIGH": 70,    # >= 70% churn probability
    "MEDIUM": 40,  # >= 40% and < 70%
    "LOW": 0,      # < 40%
}

def get_risk_tier(churn_probability: float) -> str:
    """Convert churn probability (0-100) to risk tier string."""
    if churn_probability >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif churn_probability >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    else:
        return "LOW"

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

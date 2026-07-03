"""
Churn Prediction — Data Package
================================
Provides dataset loading, validation, feature engineering, and
preprocessing utilities for the IBM Telco Customer Churn dataset.

Public API
----------
Functions (loader):
    load_telco_data     – Download / cache / clean the raw CSV.
    get_feature_target_split – Separate X and y, dropping customerID.
    validate_input_data – Check that API input has required columns.

Classes (preprocessor):
    ChurnPreprocessor   – Fit/transform wrapper around an sklearn
                          ColumnTransformer pipeline with engineered features.
"""

from .loader import (
    get_feature_target_split,
    load_telco_data,
    validate_input_data,
)
from .preprocessor import ChurnPreprocessor

__all__ = [
    "load_telco_data",
    "get_feature_target_split",
    "validate_input_data",
    "ChurnPreprocessor",
]

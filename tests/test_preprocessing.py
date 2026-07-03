"""
Tests for the Churn Prediction preprocessing pipeline.
======================================================
Covers data loading, cleaning, feature engineering, and the full
ChurnPreprocessor fit → transform → serialize → load round-trip.

Run with::

    pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal synthetic dataset that mirrors the Telco schema
_SYNTHETIC_ROWS = [
    {
        "customerID": "0001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 56.95,
        "TotalCharges": "683.4",
        "Churn": "No",
    },
    {
        "customerID": "0002",
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.10,
        "TotalCharges": "178.2",
        "Churn": "Yes",
    },
    {
        "customerID": "0003",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 0,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 20.25,
        "TotalCharges": " ",  # blank string — the gotcha!
        "Churn": "No",
    },
    {
        "customerID": "0004",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "Yes",
        "tenure": 48,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 105.50,
        "TotalCharges": "5064.0",
        "Churn": "No",
    },
]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Create a small synthetic DataFrame mimicking the raw Telco CSV."""
    return pd.DataFrame(_SYNTHETIC_ROWS)


@pytest.fixture
def cleaned_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned version of the synthetic data, simulating loader output."""
    from src.churn_prediction.data.loader import _clean, _validate_schema

    _validate_schema(raw_df)
    return _clean(raw_df)


@pytest.fixture
def features_target(cleaned_df: pd.DataFrame):
    """Return (X, y) split from the cleaned synthetic data."""
    from src.churn_prediction.data.loader import get_feature_target_split

    return get_feature_target_split(cleaned_df)


# ---------------------------------------------------------------------------
# Tests — Data Loading & Cleaning
# ---------------------------------------------------------------------------


class TestDataCleaning:
    """Validate the loader's cleaning logic."""

    def test_total_charges_is_float(self, cleaned_df: pd.DataFrame):
        """TotalCharges must be float64 after cleaning."""
        assert cleaned_df["TotalCharges"].dtype == np.float64

    def test_blank_total_charges_filled(self, cleaned_df: pd.DataFrame):
        """Blank TotalCharges values (row with tenure=0) should become 0.0."""
        row = cleaned_df.loc[cleaned_df["customerID"] == "0003"]
        assert row["TotalCharges"].iloc[0] == 0.0

    def test_churn_is_binary_int(self, cleaned_df: pd.DataFrame):
        """Churn must be int 0/1 after cleaning."""
        assert cleaned_df["Churn"].dtype in (np.int64, np.int32, int)
        assert set(cleaned_df["Churn"].unique()).issubset({0, 1})

    def test_no_internet_service_collapsed(self, cleaned_df: pd.DataFrame):
        """'No internet service' should be replaced with 'No'."""
        for col in [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]:
            values = cleaned_df[col].unique()
            assert "No internet service" not in values, (
                f"'{col}' still contains 'No internet service'"
            )

    def test_no_phone_service_collapsed(self, cleaned_df: pd.DataFrame):
        """'No phone service' should be replaced with 'No'."""
        assert "No phone service" not in cleaned_df["MultipleLines"].unique()

    def test_customer_id_preserved(self, cleaned_df: pd.DataFrame):
        """customerID should still be present after cleaning (dropped later)."""
        assert "customerID" in cleaned_df.columns

    def test_schema_validation_catches_missing_cols(self):
        """SchemaValidationError must fire on missing columns."""
        from src.churn_prediction.data.loader import SchemaValidationError, _validate_schema

        bad_df = pd.DataFrame({"foo": [1], "bar": [2]})
        with pytest.raises(SchemaValidationError):
            _validate_schema(bad_df)


class TestFeatureTargetSplit:
    """Validate the X/y split function."""

    def test_customer_id_dropped(self, features_target):
        X, y = features_target
        assert "customerID" not in X.columns

    def test_target_not_in_features(self, features_target):
        X, y = features_target
        assert "Churn" not in X.columns

    def test_target_is_series(self, features_target):
        _, y = features_target
        assert isinstance(y, pd.Series)


# ---------------------------------------------------------------------------
# Tests — Feature Engineering
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    """Validate the ChurnPreprocessor's feature engineering."""

    def test_fit_transform_returns_array(self, features_target):
        from src.churn_prediction.data.preprocessor import ChurnPreprocessor

        X, _ = features_target
        pp = ChurnPreprocessor()
        result = pp.fit_transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(X)

    def test_feature_names_available_after_fit(self, features_target):
        from src.churn_prediction.data.preprocessor import ChurnPreprocessor

        X, _ = features_target
        pp = ChurnPreprocessor()
        pp.fit(X)
        names = pp.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert len(names) == pp.transform(X).shape[1]

    def test_transform_before_fit_raises(self, features_target):
        from src.churn_prediction.data.preprocessor import ChurnPreprocessor

        X, _ = features_target
        pp = ChurnPreprocessor()
        with pytest.raises(RuntimeError, match="not been fitted"):
            pp.transform(X)

    def test_engineered_features_created(self, features_target):
        from src.churn_prediction.data.preprocessor import ChurnPreprocessor

        X, _ = features_target
        pp = ChurnPreprocessor()
        pp.fit(X)
        names = pp.get_feature_names()
        # Check that at least some engineered feature names appear
        name_str = " ".join(names)
        assert "avg_monthly_charge" in name_str or "num__avg_monthly_charge" in name_str
        assert "num_services" in name_str or "num__num_services" in name_str

    def test_serialization_round_trip(self, features_target, tmp_path):
        from src.churn_prediction.data.preprocessor import ChurnPreprocessor

        X, _ = features_target
        pp = ChurnPreprocessor()
        pp.fit(X)
        original_output = pp.transform(X)
        original_names = pp.get_feature_names()

        # Save and reload
        save_path = tmp_path / "test_preprocessor.joblib"
        pp.save(save_path)
        loaded = ChurnPreprocessor.load(save_path)

        loaded_output = loaded.transform(X)
        loaded_names = loaded.get_feature_names()

        np.testing.assert_array_almost_equal(original_output, loaded_output)
        assert original_names == loaded_names


# ---------------------------------------------------------------------------
# Tests — Input Validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Validate the API input validation function."""

    def test_valid_input_passes(self, features_target):
        from src.churn_prediction.data.loader import validate_input_data

        X, _ = features_target
        # Should not raise
        errors = validate_input_data(X)
        assert errors == []

    def test_missing_columns_raises(self):
        from src.churn_prediction.data.loader import (
            InputValidationError,
            validate_input_data,
        )

        bad_df = pd.DataFrame({"tenure": [10], "MonthlyCharges": [50.0]})
        with pytest.raises(InputValidationError):
            validate_input_data(bad_df)

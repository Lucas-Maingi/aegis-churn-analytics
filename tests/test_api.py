"""
API Layer Integration Tests
============================
Verifies route execution, request validation, authentication checks,
rate limiting enforcement, and schema validation.
"""

import pytest
from fastapi.testclient import TestClient

from churn_prediction.api.auth import API_KEY
from churn_prediction.api.main import app


@pytest.fixture(name="client", scope="module")
def client_fixture():
    """Fixture to expose a TestClient inside the app lifespan context manager."""
    with TestClient(app) as c:
        yield c


# ── Mock Data ───────────────────────────────────────────────────────────────

MOCK_CUSTOMER_VALID = {
    "customerID": "9999-TEST",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.50,
    "TotalCharges": 966.00,
}

MOCK_CUSTOMER_INVALID_CAT = {
    **MOCK_CUSTOMER_VALID,
    "gender": "InvalidGender",  # Validation error
}

MOCK_CUSTOMER_INVALID_NUM = {
    **MOCK_CUSTOMER_VALID,
    "tenure": -5,  # Must be >= 0
}

MOCK_CUSTOMER_INVALID_TOTAL_CHARGES = {
    **MOCK_CUSTOMER_VALID,
    "TotalCharges": "not-a-number",  # Must be valid float string or float
}

# ── Health Route Tests ───────────────────────────────────────────────────────


def test_health_check(client: TestClient):
    """Verify that the health route returns a 200 status and correct schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert data["model_loaded"] is True
    assert "metadata" in data


# ── Authentication Tests ─────────────────────────────────────────────────────


def test_predict_without_auth(client: TestClient):
    """Ensure that POST /predict fails with 422 if auth header is missing."""
    response = client.post("/predict", json=MOCK_CUSTOMER_VALID)
    # FastAPI returns 422 for missing headers if not handled, or 401.
    # In our verify_api_key implementation, x_api_key = Header(...) raises 422 if missing.
    assert response.status_code == 422


def test_predict_invalid_auth(client: TestClient):
    """Verify that POST /predict returns 401 for an incorrect API key."""
    headers = {"X-API-Key": "wrong_key_1234"}
    response = client.post("/predict", json=MOCK_CUSTOMER_VALID, headers=headers)
    assert response.status_code == 401
    assert "Invalid or missing API key" in response.json()["detail"]


# ── Prediction Endpoint Tests ────────────────────────────────────────────────


def test_predict_single_success(client: TestClient):
    """Verify standard POST /predict execution with a valid customer."""
    headers = {"X-API-Key": API_KEY}
    response = client.post("/predict", json=MOCK_CUSTOMER_VALID, headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert data["customerID"] == "9999-TEST"
    assert 0.0 <= data["churn_probability"] <= 100.0
    assert data["risk_tier"] in ("LOW", "MEDIUM", "HIGH")

    # Check SHAP drivers
    assert len(data["explanations"]) == 3
    for exp in data["explanations"]:
        assert "feature_name" in exp
        assert "shap_value" in exp
        assert "feature_value" in exp
        assert "direction" in exp
        assert "plain_english" in exp
        assert "churn risk" in exp["direction"]


def test_predict_invalid_schema(client: TestClient):
    """Assert that invalid categories yield 422 unprocessable entity."""
    headers = {"X-API-Key": API_KEY}

    # Test categorical validation
    response = client.post("/predict", json=MOCK_CUSTOMER_INVALID_CAT, headers=headers)
    assert response.status_code == 422
    assert "Input validation error" in response.json()["detail"]
    assert "validation_errors" in response.json()

    # Test numerical validation
    response = client.post("/predict", json=MOCK_CUSTOMER_INVALID_NUM, headers=headers)
    assert response.status_code == 422

    # Test string-float parsing validation
    response = client.post(
        "/predict", json=MOCK_CUSTOMER_INVALID_TOTAL_CHARGES, headers=headers
    )
    assert response.status_code == 422


def test_predict_batch_success(client: TestClient):
    """Verify bulk scoring yields correct batch response models."""
    headers = {"X-API-Key": API_KEY}
    batch_payload = {"customers": [MOCK_CUSTOMER_VALID, MOCK_CUSTOMER_VALID]}

    response = client.post("/predict/batch", json=batch_payload, headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2

    # Validate elements
    pred = data["predictions"][0]
    assert pred["customerID"] == "9999-TEST"
    assert 0.0 <= pred["churn_probability"] <= 100.0
    assert len(pred["explanations"]) == 3


# ── Rate Limiting Tests ───────────────────────────────────────────────────────


def test_rate_limiting(client: TestClient):
    """Verify that hitting the rate limit returns a 429 Too Many Requests response."""
    headers = {"X-API-Key": API_KEY}
    
    # Send requests to verify rate limits
    status_codes = []
    # Send 70 rapid requests to force-exceed the 60 requests/minute ceiling
    for _ in range(70):
        res = client.post("/predict", json=MOCK_CUSTOMER_VALID, headers=headers)
        status_codes.append(res.status_code)
        if res.status_code == 429:
            break

    assert 429 in status_codes

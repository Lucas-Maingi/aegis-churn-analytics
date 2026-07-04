"""
API Request and Response Schemas
================================
Pydantic schemas for strict validation of input features and standardized
response structures.
"""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class CustomerInput(BaseModel):
    """Input schema representing the behavioral features of a single customer."""

    customerID: Optional[str] = Field(
        None, description="Optional unique identifier for the customer"
    )
    gender: Literal["Male", "Female"] = Field(
        ..., description="Gender ('Male', 'Female')"
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="Senior citizen indicator (0 or 1)"
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Partner status ('Yes', 'No')"
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Dependents status ('Yes', 'No')"
    )
    tenure: int = Field(
        ..., ge=0, description="Number of months customer has stayed with the company"
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Phone service status ('Yes', 'No')"
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Multiple lines status ('Yes', 'No', 'No phone service')"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Internet service provider ('DSL', 'Fiber optic', 'No')"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Online security status ('Yes', 'No', 'No internet service')"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Online backup status ('Yes', 'No', 'No internet service')"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Device protection ('Yes', 'No', 'No internet service')"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Tech support ('Yes', 'No', 'No internet service')"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Streaming TV ('Yes', 'No', 'No internet service')"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Streaming movies ('Yes', 'No', 'No internet service')"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Contract term ('Month-to-month', 'One year', 'Two year')"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Paperless billing status ('Yes', 'No')"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        ...,
        description="Payment method ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')",
    )
    MonthlyCharges: float = Field(
        ..., ge=0.0, description="Monthly charges amount (must be non-negative)"
    )
    TotalCharges: Union[float, str] = Field(
        ...,
        description="Total charges amount (numeric value or blank space string)",
    )

    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, v: Union[float, str]) -> Union[float, str]:
        """Verify that TotalCharges is either float or a valid float string."""
        if isinstance(v, str):
            stripped = v.strip()
            if stripped == "":
                return " "
            try:
                float(stripped)
            except ValueError:
                raise ValueError(
                    "TotalCharges must be a valid float string or blank spaces"
                )
            return stripped
        return v

    def to_pandas_dict(self) -> dict:
        """Convert pydantic model to a raw pandas dictionary format.

        Keeps the structure identical to the CSV columns so that data cleaning and
        preprocessing pipeline functions work seamlessly.
        """
        data = self.model_dump()
        # Ensure customerID is passed as string if present, else empty string
        if data.get("customerID") is None:
            data["customerID"] = ""
        return data


class CustomerBatchInput(BaseModel):
    """Wrapper schema for batch customer prediction inputs."""

    customers: List[CustomerInput] = Field(
        ..., description="List of customer record items to predict"
    )


class ExplanationItem(BaseModel):
    """Structured representation of a single behavioral feature driver."""

    feature_name: str = Field(..., description="The name of the engineered/raw feature")
    shap_value: float = Field(..., description="The calculated SHAP contribution score")
    feature_value: Union[float, int, str] = Field(
        ..., description="The raw/engineered input value for this feature"
    )
    direction: str = Field(
        ..., description="Whether this driver increases or decreases churn risk"
    )
    plain_english: str = Field(
        ..., description="Human-friendly explanation message of this risk signal"
    )


class PredictionResponse(BaseModel):
    """Individual prediction response schema."""

    customerID: Optional[str] = Field(
        None, description="The customer ID matching the request item"
    )
    churn_probability: float = Field(
        ..., ge=0.0, le=100.0, description="The churn risk probability score (0-100)"
    )
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Calculated risk tier based on probability score"
    )
    explanations: List[ExplanationItem] = Field(
        ..., description="Top 3 behavioral signals driving this prediction score"
    )


class BatchPredictionResponse(BaseModel):
    """Wrapper schema for batch prediction responses."""

    predictions: List[PredictionResponse] = Field(
        ..., description="Ordered list of prediction items matching the batch inputs"
    )

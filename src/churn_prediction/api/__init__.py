"""
Churn Prediction — API Package
===============================
REST API layer exposing endpoints to predict churn risk and explain predictions
using Pydantic validation, memory-based rate limiting, and API key authentication.
"""

from churn_prediction.api.main import app

__all__ = ["app"]

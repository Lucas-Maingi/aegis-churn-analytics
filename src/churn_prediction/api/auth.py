"""
Authentication Provider
=======================
API key security dependency to protect prediction endpoints.
"""

import os

from fastapi import Header, HTTPException, status

# Load API key from environment, default to mock key for local development
API_KEY = os.getenv("API_KEY", "test_api_key_1234")


def verify_api_key(x_api_key: str = Header(..., description="API key for authentication")) -> str:
    """FastAPI security dependency to validate the incoming API key header.

    Parameters
    ----------
    x_api_key : str
        The header value of 'X-API-Key'.

    Returns
    -------
    str
        The validated API key.

    Raises
    ------
    HTTPException
        If the API key header is missing or incorrect.
    """
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Please check your credentials.",
        )
    return x_api_key

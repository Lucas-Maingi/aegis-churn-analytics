"""
CSV Column Mapping & Value Normalization
========================================
Translates an arbitrary customer CSV (from an ISP/telecom billing export)
into the model's Telco-shaped feature schema.

The upload flow sends a ``mapping`` dict of ``{model_field: csv_column}``.
Unmapped fields fall back to conservative defaults, and raw cell values
are normalized to the categorical vocabulary the model was trained on
(e.g. "fiber" → "Fiber optic", "1 year" → "One year").
"""

from typing import Any, Dict, Optional

# Model fields a row must be able to produce (directly or via defaults)
REQUIRED_FIELDS = ["external_id", "tenure", "MonthlyCharges"]

# Extra, non-model columns we also capture for the dashboard/outreach
CONTACT_FIELDS = ["name", "email"]

MODEL_FIELDS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# Conservative defaults for fields the uploader could not map.
# TotalCharges is derived (tenure * MonthlyCharges) when absent.
FIELD_DEFAULTS: Dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "No",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

# Allowed vocabulary per categorical field (must match training data)
FIELD_VOCABULARY: Dict[str, list] = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

# Common real-world synonyms → canonical training vocabulary
_VALUE_SYNONYMS: Dict[str, Dict[str, str]] = {
    "Contract": {
        "monthly": "Month-to-month", "month to month": "Month-to-month",
        "month-to-month": "Month-to-month", "m2m": "Month-to-month",
        "prepaid": "Month-to-month", "pay as you go": "Month-to-month",
        "1 year": "One year", "one year": "One year", "annual": "One year",
        "yearly": "One year", "12 months": "One year",
        "2 year": "Two year", "two year": "Two year", "biennial": "Two year",
        "24 months": "Two year",
    },
    "InternetService": {
        "fiber": "Fiber optic", "fibre": "Fiber optic",
        "fiber optic": "Fiber optic", "ftth": "Fiber optic",
        "fixed wireless": "DSL", "wireless": "DSL", "adsl": "DSL",
        "dsl": "DSL", "copper": "DSL",
        "none": "No", "no": "No", "n/a": "No",
    },
    "PaymentMethod": {
        "mpesa": "Electronic check", "m-pesa": "Electronic check",
        "mobile money": "Electronic check", "airtel money": "Electronic check",
        "paypal": "Electronic check", "electronic check": "Electronic check",
        "cash": "Mailed check", "check": "Mailed check", "cheque": "Mailed check",
        "mailed check": "Mailed check",
        "bank transfer": "Bank transfer (automatic)",
        "bank transfer (automatic)": "Bank transfer (automatic)",
        "direct debit": "Bank transfer (automatic)", "ach": "Bank transfer (automatic)",
        "card": "Credit card (automatic)", "credit card": "Credit card (automatic)",
        "credit card (automatic)": "Credit card (automatic)",
        "debit card": "Credit card (automatic)", "visa": "Credit card (automatic)",
        "mastercard": "Credit card (automatic)",
    },
    "gender": {
        "m": "Male", "male": "Male", "f": "Female", "female": "Female",
    },
}

_YES_VALUES = {"yes", "y", "true", "1", "1.0", "active", "enabled"}
_NO_VALUES = {"no", "n", "false", "0", "0.0", "inactive", "disabled", "none", ""}


def normalize_value(field: str, raw: Any) -> Optional[Any]:
    """Normalize a raw CSV cell into the model's vocabulary for *field*.

    Returns None when the value cannot be interpreted (caller applies the
    field default).
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() in ("nan", "n/a", "na", "null"):
        return None

    if field == "SeniorCitizen":
        return 1 if text.lower() in _YES_VALUES else 0

    if field in ("tenure",):
        try:
            return max(0, int(float(text)))
        except ValueError:
            return None

    if field in ("MonthlyCharges", "TotalCharges"):
        try:
            return max(0.0, float(text.replace(",", "").replace("$", "")))
        except ValueError:
            return None

    vocab = FIELD_VOCABULARY.get(field)
    if vocab is None:
        return text

    # Exact vocabulary match (case-insensitive)
    for allowed in vocab:
        if text.lower() == allowed.lower():
            return allowed

    # Known synonyms
    synonyms = _VALUE_SYNONYMS.get(field, {})
    if text.lower() in synonyms:
        return synonyms[text.lower()]

    # Generic Yes/No coercion for binary-ish fields
    if "Yes" in vocab and text.lower() in _YES_VALUES:
        return "Yes"
    if "No" in vocab and text.lower() in _NO_VALUES:
        return "No"

    return None


def map_row(row: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """Convert one raw CSV row into the model feature schema.

    Parameters
    ----------
    row:
        Raw CSV row as ``{column_name: cell_value}``.
    mapping:
        ``{model_field: csv_column}`` chosen by the uploader. May also map
        ``external_id``, ``name``, and ``email``.

    Returns
    -------
    dict with keys: ``external_id``, ``name``, ``email``, ``features``
    (the 19 model fields, normalized and defaulted).
    """
    def source(field: str) -> Any:
        col = mapping.get(field)
        return row.get(col) if col else None

    features: Dict[str, Any] = {}
    for field in MODEL_FIELDS:
        value = normalize_value(field, source(field))
        if value is None:
            value = FIELD_DEFAULTS.get(field)
        features[field] = value

    # Derive TotalCharges when not mapped/parsable
    if features.get("TotalCharges") is None:
        features["TotalCharges"] = round(
            float(features["tenure"] or 0) * float(features["MonthlyCharges"] or 0.0), 2
        )
    if features.get("tenure") is None:
        features["tenure"] = 0
    if features.get("MonthlyCharges") is None:
        features["MonthlyCharges"] = 0.0

    # The training loader collapses "No internet service" / "No phone
    # service" to "No" before fitting the encoder, so stored features must
    # use the collapsed form or they one-hot-encode as all zeros.
    if features["InternetService"] == "No":
        for dep in ("OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"):
            features[dep] = "No"
    if features["PhoneService"] == "No":
        features["MultipleLines"] = "No"
    for field in ("MultipleLines", "OnlineSecurity", "OnlineBackup",
                  "DeviceProtection", "TechSupport", "StreamingTV",
                  "StreamingMovies"):
        if features[field] in ("No internet service", "No phone service"):
            features[field] = "No"

    external_id = source("external_id")
    return {
        "external_id": str(external_id).strip() if external_id is not None else "",
        "name": str(source("name") or "").strip(),
        "email": str(source("email") or "").strip(),
        "features": features,
    }


def suggest_mapping(csv_columns: list) -> Dict[str, str]:
    """Auto-suggest a ``{model_field: csv_column}`` mapping from headers.

    Matches on lowercased, de-underscored header names against common
    aliases so the uploader only has to fix the leftovers.
    """
    aliases: Dict[str, list] = {
        "external_id": ["customerid", "customer id", "id", "subscriber id",
                        "account id", "account number", "customer_id", "user id"],
        "name": ["name", "customer name", "full name", "subscriber name"],
        "email": ["email", "e-mail", "email address", "customer email"],
        "gender": ["gender", "sex"],
        "SeniorCitizen": ["seniorcitizen", "senior citizen", "senior"],
        "Partner": ["partner", "has partner", "married"],
        "Dependents": ["dependents", "has dependents", "children"],
        "tenure": ["tenure", "months active", "tenure months", "months",
                   "subscription length", "account age"],
        "PhoneService": ["phoneservice", "phone service", "phone", "voice"],
        "MultipleLines": ["multiplelines", "multiple lines", "lines"],
        "InternetService": ["internetservice", "internet service", "internet",
                            "connection type", "service type", "plan type"],
        "OnlineSecurity": ["onlinesecurity", "online security", "security",
                           "security addon"],
        "OnlineBackup": ["onlinebackup", "online backup", "backup"],
        "DeviceProtection": ["deviceprotection", "device protection", "insurance"],
        "TechSupport": ["techsupport", "tech support", "support plan",
                        "tech support addon", "support addon"],
        "StreamingTV": ["streamingtv", "streaming tv", "tv", "iptv", "tv addon"],
        "StreamingMovies": ["streamingmovies", "streaming movies", "movies", "vod"],
        "Contract": ["contract", "contract type", "plan", "billing cycle",
                     "subscription type"],
        "PaperlessBilling": ["paperlessbilling", "paperless billing", "paperless",
                             "e-billing"],
        "PaymentMethod": ["paymentmethod", "payment method", "payment",
                          "payment type"],
        "MonthlyCharges": ["monthlycharges", "monthly charges", "monthly charge",
                           "monthly fee", "monthly bill", "mrr", "monthly cost",
                           "plan price", "amount"],
        "TotalCharges": ["totalcharges", "total charges", "total billed",
                         "lifetime value", "total revenue", "total paid"],
    }

    def canon(s: str) -> str:
        return s.strip().lower().replace("_", " ").replace("-", " ")

    canon_cols = {canon(c): c for c in csv_columns}
    suggestion: Dict[str, str] = {}
    for field, names in aliases.items():
        for alias in names:
            hit = canon_cols.get(canon(alias))
            if hit is not None:
                suggestion[field] = hit
                break
    return suggestion

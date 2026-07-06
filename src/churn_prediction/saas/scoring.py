"""
Batch Scoring Service
=====================
Scores stored customers through the loaded model artifacts (preprocessor →
XGBoost → SHAP) in one vectorized pass and persists results back onto the
Customer rows.
"""

import logging
from datetime import datetime, timezone
from typing import Any, List

import numpy as np
import pandas as pd

from churn_prediction.config import SERVICE_COLUMNS, get_risk_tier
from churn_prediction.models.explainer import ChurnExplainer
from churn_prediction.saas.models import Customer

logger = logging.getLogger(__name__)

_CONTRACT_CODE = {"Month-to-month": 0, "One year": 1, "Two year": 2}


def _raw_feature_value(feature_name: str, features: dict, transformed_value):
    """Resolve the human-readable raw value behind a transformed feature.

    The model works on scaled/one-hot values, but explanation templates
    like "customer for {value} months" must show the original numbers.
    """
    clean = feature_name
    for prefix in ("num__", "bin__", "cat__", "remainder__"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break

    if clean in features:
        return features[clean]

    tenure = float(features.get("tenure") or 0)
    monthly = float(features.get("MonthlyCharges") or 0.0)
    total = float(features.get("TotalCharges") or 0.0)

    # Engineered features — mirror data/preprocessor.py definitions
    if clean == "avg_monthly_charge":
        return round(total / max(tenure, 1.0), 2)
    if clean == "contract_charge_interaction":
        return round(_CONTRACT_CODE.get(features.get("Contract"), 0) * monthly, 2)
    if clean == "num_services":
        count = 0
        for col in SERVICE_COLUMNS:
            val = features.get(col, "No")
            count += int(val != "No") if col == "InternetService" else int(val == "Yes")
        return count
    if clean == "has_protection_bundle":
        protected = all(
            features.get(col) == "Yes"
            for col in ("OnlineSecurity", "TechSupport", "DeviceProtection")
        )
        return "Yes" if protected else "No"
    if clean.startswith("tenure_bucket"):
        if tenure <= 12:
            return "0-12"
        if tenure <= 24:
            return "13-24"
        if tenure <= 48:
            return "25-48"
        return "49-72"

    return transformed_value


def score_customers(
    customers: List[Customer],
    model: Any,
    preprocessor: Any,
    feature_names: List[str],
    top_n: int = 3,
) -> int:
    """Score a list of Customer rows in place. Returns the number scored.

    Each customer's ``features`` dict (Telco-shaped raw fields) is run
    through the fitted preprocessor and model; probability, risk tier, and
    the top-N SHAP drivers are written back onto the ORM objects (caller
    commits the session).
    """
    if not customers:
        return 0

    raw_rows = []
    for cust in customers:
        row = dict(cust.features)
        row["customerID"] = cust.external_id
        raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    X = preprocessor.transform(raw_df)
    probs = model.predict_proba(X)[:, 1]

    explainer = ChurnExplainer(model, feature_names)
    shap_values = explainer.compute_shap_values(X)

    now = datetime.now(timezone.utc)
    for i, cust in enumerate(customers):
        prob_pct = round(float(probs[i]) * 100.0, 2)
        cust.churn_probability = prob_pct
        cust.risk_tier = get_risk_tier(prob_pct)
        cust.scored_at = now

        row_shap = shap_values[i]
        row_features = X[i]
        top_indices = np.argsort(np.abs(row_shap))[::-1][:top_n]

        explanations = []
        for idx in top_indices:
            feat = feature_names[idx]
            sv = float(row_shap[idx])
            fv = row_features[idx]
            raw_value = _raw_feature_value(feat, cust.features, fv)
            direction = "increases" if sv > 0 else "decreases"
            explanations.append({
                "feature_name": feat,
                "shap_value": sv,
                "feature_value": raw_value.item() if hasattr(raw_value, "item") else raw_value,
                "direction": f"{direction} churn risk",
                "plain_english": explainer._render_explanation(feat, raw_value),
            })
        cust.explanations = explanations

    logger.info("Scored %d customers.", len(customers))
    return len(customers)

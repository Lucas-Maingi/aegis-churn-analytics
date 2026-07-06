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

from churn_prediction.config import get_risk_tier
from churn_prediction.models.explainer import ChurnExplainer
from churn_prediction.saas.models import Customer

logger = logging.getLogger(__name__)


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
            direction = "increases" if sv > 0 else "decreases"
            explanations.append({
                "feature_name": feat,
                "shap_value": sv,
                "feature_value": fv.item() if hasattr(fv, "item") else fv,
                "direction": f"{direction} churn risk",
                "plain_english": explainer._render_explanation(feat, fv),
            })
        cust.explanations = explanations

    logger.info("Scored %d customers.", len(customers))
    return len(customers)

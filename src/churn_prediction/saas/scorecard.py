"""
Model Scorecard
===============
Measures how the *active* model actually performed against ground-truth
outcomes the operator recorded — the feedback half of the loop. Works with
even a handful of resolved outcomes (reported alongside the sample size so
the operator can judge how much to trust it).
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from churn_prediction.config import get_risk_tier
from churn_prediction.saas.models import Customer
from churn_prediction.saas.tenant_model import get_active_model, get_promoted_tenant_model

logger = logging.getLogger(__name__)

# Operational decision threshold: a customer is "flagged to act on" when the
# active model puts them at >= 50% churn probability (i.e. HIGH or upper-MEDIUM).
ACT_THRESHOLD = 50.0


def labeled_customers(db: Session, org_id: int) -> List[Customer]:
    """Return the org's customers that have a recorded ground-truth outcome."""
    return (
        db.query(Customer)
        .filter(
            Customer.org_id == org_id,
            Customer.actual_outcome.in_(["churned", "retained"]),
        )
        .all()
    )


def build_labeled_matrix(
    customers: List[Customer], preprocessor: Any
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Transform labeled customers into (X, y). y=1 means churned.

    Returns None if there are no labeled customers.
    """
    if not customers:
        return None
    raw_rows = []
    labels = []
    for c in customers:
        row = dict(c.features)
        row["customerID"] = c.external_id
        raw_rows.append(row)
        labels.append(1 if c.actual_outcome == "churned" else 0)
    X = preprocessor.transform(pd.DataFrame(raw_rows))
    y = np.asarray(labels, dtype=int)
    return X, y


def compute_scorecard(
    db: Session, org_id: int, base_model: Any, preprocessor: Any
) -> dict:
    """Compute active-model performance on the org's labeled customers."""
    active_model, source = get_active_model(db, org_id, base_model)
    customers = labeled_customers(db, org_id)

    total = db.query(Customer).filter(Customer.org_id == org_id).count()
    n_labeled = len(customers)

    # When a tenant model is active, its promotion was decided on a held-out
    # split — that comparison is the honest, un-leaked evidence of improvement
    # (the live scorecard below scores all labeled rows, some of which the
    # tenant model trained on, so its AUC would be optimistic).
    promoted = get_promoted_tenant_model(db, org_id)
    validated = (
        {
            "base_auc": promoted.base_auc,
            "tenant_auc": promoted.tenant_auc,
            "n_eval": promoted.n_eval,
        }
        if promoted is not None
        else None
    )

    scorecard = {
        "active_model": source,
        "n_customers": total,
        "n_outcomes": n_labeled,
        "n_churned": sum(1 for c in customers if c.actual_outcome == "churned"),
        "n_retained": sum(1 for c in customers if c.actual_outcome == "retained"),
        # metrics below are null until there are enough labeled outcomes
        "accuracy": None,
        "high_risk_precision": None,
        "recall": None,
        "auc": None,
        "confusion": None,
        "validated_improvement": validated,
    }

    if n_labeled == 0:
        return scorecard

    matrix = build_labeled_matrix(customers, preprocessor)
    X, y = matrix
    proba = active_model.predict_proba(X)[:, 1] * 100.0
    predicted_positive = proba >= ACT_THRESHOLD

    # Confusion at the action threshold (positive = "model said act")
    tp = int(np.sum(predicted_positive & (y == 1)))
    fp = int(np.sum(predicted_positive & (y == 0)))
    fn = int(np.sum(~predicted_positive & (y == 1)))
    tn = int(np.sum(~predicted_positive & (y == 0)))

    scorecard["accuracy"] = round((tp + tn) / n_labeled, 3)
    scorecard["high_risk_precision"] = round(tp / (tp + fp), 3) if (tp + fp) else None
    scorecard["recall"] = round(tp / (tp + fn), 3) if (tp + fn) else None
    scorecard["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    # Live AUC is only honest when the active model did not train on these
    # rows — i.e. the base model (trained on Telco, never on this tenant).
    # With a tenant model active, the held-out `validated_improvement` above
    # is the trustworthy figure instead.
    if source == "base" and scorecard["n_churned"] > 0 and scorecard["n_retained"] > 0:
        from sklearn.metrics import roc_auc_score

        scorecard["auc"] = round(float(roc_auc_score(y, proba)), 3)

    # Per-tier actual churn rate — the most intuitive operator view
    tier_stats: dict = {"HIGH": [0, 0], "MEDIUM": [0, 0], "LOW": [0, 0]}
    for c in customers:
        tier = get_risk_tier(c.churn_probability) if c.churn_probability is not None else "LOW"
        tier_stats[tier][0] += 1 if c.actual_outcome == "churned" else 0
        tier_stats[tier][1] += 1
    scorecard["tier_actual_churn"] = {
        tier: {
            "churned": churned,
            "total": count,
            "rate": round(churned / count, 3) if count else None,
        }
        for tier, (churned, count) in tier_stats.items()
    }

    return scorecard

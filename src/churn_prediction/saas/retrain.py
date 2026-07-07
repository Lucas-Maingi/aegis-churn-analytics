"""
Per-Tenant Retraining
=====================
Trains an organization-specific churn model on the outcomes that org has
actually recorded, evaluates it against the base model on a held-out split
of the *same* tenant data, and promotes it only if it genuinely wins.

Uses XGBoost directly (not models.trainer.ModelTrainer, which eagerly imports
MLflow — a training-only dependency absent from the lean serving image).

The minimum-outcome gate is deliberately conservative: a model trained on a
few dozen labels overfits, so retraining is refused until the org has enough
resolved outcomes with both churned and retained examples present.
"""

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session
from xgboost import XGBClassifier

from churn_prediction.config import RANDOM_STATE
from churn_prediction.saas.models import Customer, TenantModel
from churn_prediction.saas.scorecard import build_labeled_matrix, labeled_customers
from churn_prediction.saas.scoring import score_customers
from churn_prediction.saas.tenant_model import dump_model

logger = logging.getLogger(__name__)

# Gate: retraining is refused below this many resolved outcomes. Real
# deployments would set this into the hundreds; it is low enough here that a
# seeded demo org can exercise the pipeline.
MIN_OUTCOMES_FOR_RETRAIN = 40

# A tenant model must beat the base model's held-out AUC by at least this
# margin to be promoted — avoids churning the active model on noise.
PROMOTE_MARGIN = 0.01


def can_retrain(db: Session, org_id: int) -> tuple[bool, str]:
    """Report whether the org has enough labeled data to retrain."""
    customers = labeled_customers(db, org_id)
    n = len(customers)
    n_churned = sum(1 for c in customers if c.actual_outcome == "churned")
    n_retained = n - n_churned

    if n < MIN_OUTCOMES_FOR_RETRAIN:
        return False, (
            f"Need at least {MIN_OUTCOMES_FOR_RETRAIN} recorded outcomes to "
            f"retrain safely — you have {n}."
        )
    if n_churned < 2 or n_retained < 2:
        return False, (
            "Need both churned and retained examples before retraining "
            f"(have {n_churned} churned, {n_retained} retained)."
        )
    return True, "Ready to retrain."


def retrain_tenant_model(
    db: Session,
    org_id: int,
    base_model: Any,
    preprocessor: Any,
    feature_names: list,
) -> dict:
    """Train, evaluate, and conditionally promote a tenant-specific model.

    Returns a result dict describing what happened (trained/promoted, the
    base vs. tenant AUC, and sample sizes).
    """
    ok, reason = can_retrain(db, org_id)
    if not ok:
        return {"trained": False, "promoted": False, "detail": reason}

    customers = labeled_customers(db, org_id)
    X, y = build_labeled_matrix(customers, preprocessor)

    # Stratified hold-out so both models are judged on the same unseen tenant rows
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    neg, pos = int(np.sum(y_train == 0)), int(np.sum(y_train == 1))
    scale_pos_weight = (neg / pos) if pos else 1.0

    tenant_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    tenant_model.fit(X_train, y_train)

    base_auc = float(roc_auc_score(y_eval, base_model.predict_proba(X_eval)[:, 1]))
    tenant_auc = float(roc_auc_score(y_eval, tenant_model.predict_proba(X_eval)[:, 1]))
    promoted = tenant_auc >= base_auc + PROMOTE_MARGIN

    # Any previously promoted model is demoted; only one active model per org.
    if promoted:
        for row in (
            db.query(TenantModel)
            .filter(TenantModel.org_id == org_id, TenantModel.promoted.is_(True))
            .all()
        ):
            row.promoted = False

    record = TenantModel(
        org_id=org_id,
        model_blob=dump_model(tenant_model),
        n_train=len(y_train),
        n_eval=len(y_eval),
        base_auc=round(base_auc, 4),
        tenant_auc=round(tenant_auc, 4),
        promoted=promoted,
    )
    db.add(record)

    # When promoted, re-score all of the org's customers so the dashboard
    # immediately reflects the new model.
    rescored = 0
    if promoted:
        all_customers = db.query(Customer).filter(Customer.org_id == org_id).all()
        rescored = score_customers(all_customers, tenant_model, preprocessor, feature_names)

    db.commit()

    detail = (
        f"Tenant model promoted — held-out AUC {tenant_auc:.3f} beats the base "
        f"model's {base_auc:.3f}. Re-scored {rescored} customers."
        if promoted
        else (
            f"Tenant model trained but not promoted — held-out AUC "
            f"{tenant_auc:.3f} did not beat the base model's {base_auc:.3f}. "
            "Keeping the base model."
        )
    )
    logger.info("Retrain for org %d: %s", org_id, detail)

    return {
        "trained": True,
        "promoted": promoted,
        "base_auc": round(base_auc, 3),
        "tenant_auc": round(tenant_auc, 3),
        "n_train": len(y_train),
        "n_eval": len(y_eval),
        "rescored": rescored,
        "detail": detail,
    }

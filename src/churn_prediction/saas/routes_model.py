"""
Feedback-Loop Routes
====================
Record ground-truth outcomes, view how the active model actually performed,
and trigger a gated per-tenant retrain.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from churn_prediction.saas.db import get_db
from churn_prediction.saas.deps import get_current_user
from churn_prediction.saas.models import Customer, User
from churn_prediction.saas.retrain import (
    MIN_OUTCOMES_FOR_RETRAIN,
    can_retrain,
    retrain_tenant_model,
)
from churn_prediction.saas.scorecard import compute_scorecard
from churn_prediction.saas.schemas import (
    RecordOutcomeRequest,
    RetrainResponse,
    ScorecardResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/model", tags=["model"])


@router.post("/outcome", status_code=status.HTTP_200_OK)
def record_outcome(
    payload: RecordOutcomeRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Record whether a customer actually churned or was retained."""
    cust = (
        db.query(Customer)
        .filter(Customer.id == payload.customer_id, Customer.org_id == user.org_id)
        .first()
    )
    if cust is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found."
        )

    cust.actual_outcome = payload.outcome
    cust.outcome_recorded_at = datetime.now(timezone.utc)
    db.commit()
    return {"customer_id": cust.id, "actual_outcome": cust.actual_outcome}


@router.get("/scorecard", response_model=ScorecardResponse)
def scorecard(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """How the active model performed against recorded outcomes."""
    card = compute_scorecard(
        db, user.org_id, request.app.state.model, request.app.state.preprocessor
    )
    ready, hint = can_retrain(db, user.org_id)
    card["can_retrain"] = ready
    card["retrain_hint"] = hint
    card["min_outcomes_for_retrain"] = MIN_OUTCOMES_FOR_RETRAIN
    return card


@router.post("/retrain", response_model=RetrainResponse)
def retrain(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Train a tenant-specific model and promote it if it beats the base."""
    return retrain_tenant_model(
        db,
        user.org_id,
        request.app.state.model,
        request.app.state.preprocessor,
        request.app.state.feature_names,
    )

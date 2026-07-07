"""
Tenant Model Resolution
=======================
Helpers for serializing per-tenant models to/from database blobs and
resolving which model is active for an organization (a promoted tenant
model if one exists, otherwise the shared base model).
"""

import io
import logging
from typing import Any, Optional, Tuple

import joblib
from sqlalchemy.orm import Session

from churn_prediction.saas.models import TenantModel

logger = logging.getLogger(__name__)


def dump_model(model: Any) -> bytes:
    """Serialize a fitted estimator to raw bytes for DB storage."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


def load_model(blob: bytes) -> Any:
    """Deserialize an estimator from a stored blob."""
    return joblib.load(io.BytesIO(blob))


def get_promoted_tenant_model(db: Session, org_id: int) -> Optional[TenantModel]:
    """Return the org's active (promoted) tenant model row, if any."""
    return (
        db.query(TenantModel)
        .filter(TenantModel.org_id == org_id, TenantModel.promoted.is_(True))
        .order_by(TenantModel.created_at.desc())
        .first()
    )


def get_active_model(db: Session, org_id: int, base_model: Any) -> Tuple[Any, str]:
    """Resolve the scoring model for an org.

    Returns ``(model, source)`` where source is ``"tenant"`` when a promoted
    tenant model is used, else ``"base"``.
    """
    row = get_promoted_tenant_model(db, org_id)
    if row is not None:
        try:
            return load_model(row.model_blob), "tenant"
        except Exception:
            logger.exception(
                "Failed to load promoted tenant model %d; falling back to base.",
                row.id,
            )
    return base_model, "base"

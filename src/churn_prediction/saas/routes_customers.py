"""
Customer Routes
===============
CSV upload (with column mapping), automatic batch scoring, and the ranked
customer list that powers the dashboard.
"""

import csv
import io
import json
import logging
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from sqlalchemy import func
from sqlalchemy.orm import Session

from churn_prediction.saas.db import get_db
from churn_prediction.saas.deps import get_current_user
from churn_prediction.saas.mapping import map_row, suggest_mapping
from churn_prediction.saas.models import Customer, User
from churn_prediction.saas.schemas import (
    CustomerDetail,
    CustomerListResponse,
    CustomerSummary,
    TierCounts,
    UploadResponse,
)
from churn_prediction.saas.scoring import score_customers

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/customers", tags=["customers"])

MAX_UPLOAD_ROWS = 20_000


def _read_csv(file_bytes: bytes) -> tuple[list, list]:
    """Decode CSV bytes into (columns, rows-as-dicts)."""
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV file is empty or has no header row.",
        )
    rows = list(reader)
    if len(rows) > MAX_UPLOAD_ROWS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"CSV exceeds the {MAX_UPLOAD_ROWS}-row upload limit.",
        )
    return list(reader.fieldnames), rows


@router.post("/upload/preview")
async def preview_upload(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """Inspect a CSV and auto-suggest a column mapping for the uploader UI."""
    columns, rows = _read_csv(await file.read())
    return {
        "columns": columns,
        "row_count": len(rows),
        "sample_rows": rows[:5],
        "suggested_mapping": suggest_mapping(columns),
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_customers(
    request: Request,
    file: UploadFile = File(...),
    mapping: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Import a customer CSV, map it to the model schema, and score everyone.

    ``mapping`` is a JSON object of ``{model_field: csv_column}``; it must
    map at least ``external_id``. Existing customers (same external_id) are
    updated rather than duplicated.
    """
    try:
        mapping_dict = json.loads(mapping)
        if not isinstance(mapping_dict, dict):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'mapping' must be a JSON object of {model_field: csv_column}.",
        )
    if not mapping_dict.get("external_id"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Mapping must include 'external_id' (the customer ID column).",
        )

    _, rows = _read_csv(await file.read())

    imported, updated, skipped = 0, 0, 0
    errors: list[str] = []
    touched: list[Customer] = []

    existing_by_ext = {
        c.external_id: c
        for c in db.query(Customer).filter(Customer.org_id == user.org_id).all()
    }

    for i, row in enumerate(rows):
        mapped = map_row(row, mapping_dict)
        if not mapped["external_id"]:
            skipped += 1
            if len(errors) < 10:
                errors.append(f"Row {i + 2}: missing customer ID — skipped.")
            continue

        feats = mapped["features"]
        cust = existing_by_ext.get(mapped["external_id"])
        if cust is None:
            cust = Customer(org_id=user.org_id, external_id=mapped["external_id"])
            db.add(cust)
            existing_by_ext[mapped["external_id"]] = cust
            imported += 1
        else:
            updated += 1

        cust.name = mapped["name"] or cust.name
        cust.email = mapped["email"] or cust.email
        cust.features = feats
        cust.tenure = int(feats.get("tenure") or 0)
        cust.monthly_charges = float(feats.get("MonthlyCharges") or 0.0)
        cust.contract = str(feats.get("Contract") or "")
        touched.append(cust)

    scored = score_customers(
        touched,
        model=request.app.state.model,
        preprocessor=request.app.state.preprocessor,
        feature_names=request.app.state.feature_names,
    )
    db.commit()

    logger.info(
        "Upload for org %d: %d imported, %d updated, %d scored, %d skipped.",
        user.org_id, imported, updated, scored, skipped,
    )
    return UploadResponse(
        imported=imported, updated=updated, scored=scored,
        skipped=skipped, errors=errors,
    )


@router.get("", response_model=CustomerListResponse)
def list_customers(
    risk_tier: Optional[str] = None,
    search: Optional[str] = None,
    page: int = 1,
    page_size: int = 25,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Ranked at-risk customer list with tier filtering and search."""
    page = max(1, page)
    page_size = min(max(1, page_size), 100)

    base = db.query(Customer).filter(Customer.org_id == user.org_id)

    # Tier counts + revenue at risk are computed over the whole org,
    # not just the current filter/page.
    tier_rows = (
        db.query(Customer.risk_tier, func.count(Customer.id))
        .filter(Customer.org_id == user.org_id)
        .group_by(Customer.risk_tier)
        .all()
    )
    counts = TierCounts(**{tier: n for tier, n in tier_rows if tier})
    revenue_at_risk = (
        db.query(func.coalesce(func.sum(Customer.monthly_charges), 0.0))
        .filter(Customer.org_id == user.org_id, Customer.risk_tier == "HIGH")
        .scalar()
    )

    query = base
    if risk_tier:
        query = query.filter(Customer.risk_tier == risk_tier.upper())
    if search:
        like = f"%{search}%"
        query = query.filter(
            (Customer.external_id.ilike(like))
            | (Customer.name.ilike(like))
            | (Customer.email.ilike(like))
        )

    total = query.count()
    items = (
        query.order_by(Customer.churn_probability.desc().nullslast())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return CustomerListResponse(
        items=[CustomerSummary.model_validate(c, from_attributes=True) for c in items],
        total=total,
        page=page,
        page_size=page_size,
        tier_counts=counts,
        revenue_at_risk=round(float(revenue_at_risk or 0.0), 2),
    )


@router.get("/{customer_id}", response_model=CustomerDetail)
def get_customer(
    customer_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Full customer detail: features, score, and plain-English drivers."""
    cust = (
        db.query(Customer)
        .filter(Customer.id == customer_id, Customer.org_id == user.org_id)
        .first()
    )
    if cust is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found."
        )
    return CustomerDetail.model_validate(cust, from_attributes=True)

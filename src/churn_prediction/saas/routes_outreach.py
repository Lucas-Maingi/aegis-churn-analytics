"""
Outreach Routes
===============
Offer templates, one-click send to an at-risk customer, and send history.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from churn_prediction.saas.db import get_db
from churn_prediction.saas.deps import get_current_user
from churn_prediction.saas.models import Customer, Organization, OutreachMessage, User
from churn_prediction.saas.outreach import get_template, render, send_email, TEMPLATES
from churn_prediction.saas.schemas import (
    OutreachMessageOut,
    OutreachTemplate,
    SendOutreachRequest,
    SendOutreachResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/outreach", tags=["outreach"])


@router.get("/templates", response_model=list[OutreachTemplate])
def list_templates(user: User = Depends(get_current_user)):
    """The available one-click retention templates."""
    return [OutreachTemplate(**tpl) for tpl in TEMPLATES]


@router.post("/send", response_model=SendOutreachResponse)
def send_outreach(
    payload: SendOutreachRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Render a template for one customer and send (or simulate) the email."""
    cust = (
        db.query(Customer)
        .filter(Customer.id == payload.customer_id, Customer.org_id == user.org_id)
        .first()
    )
    if cust is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found."
        )

    template = get_template(payload.template_key)
    if template is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown template '{payload.template_key}'.",
        )

    org = db.get(Organization, user.org_id)
    company = org.name if org else ""
    subject = payload.subject or render(
        template["subject"], name=cust.name, company=company, tenure=cust.tenure
    )
    body = payload.body or render(
        template["body"], name=cust.name, company=company, tenure=cust.tenure
    )

    if cust.email:
        send_status = send_email(cust.email, subject, body)
    else:
        # No address on file — record the intent so the team can follow up
        # through another channel.
        send_status = "simulated"

    message = OutreachMessage(
        org_id=user.org_id,
        customer_id=cust.id,
        template_key=payload.template_key,
        subject=subject,
        body=body,
        status=send_status,
    )
    db.add(message)
    db.commit()
    db.refresh(message)

    detail = {
        "sent": f"Email sent to {cust.email}.",
        "simulated": (
            "Message recorded (simulated send — configure RESEND_API_KEY "
            "and customer emails to deliver for real)."
        ),
        "failed": "Email provider rejected the message; it was logged as failed.",
    }[send_status]

    return SendOutreachResponse(
        message=_to_out(message, cust),
        detail=detail,
    )


@router.get("/history", response_model=list[OutreachMessageOut])
def outreach_history(
    customer_id: int | None = None,
    limit: int = 50,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Recent outreach messages for the organization (optionally one customer)."""
    query = (
        db.query(OutreachMessage, Customer)
        .join(Customer, OutreachMessage.customer_id == Customer.id)
        .filter(OutreachMessage.org_id == user.org_id)
    )
    if customer_id is not None:
        query = query.filter(OutreachMessage.customer_id == customer_id)

    rows = (
        query.order_by(OutreachMessage.created_at.desc())
        .limit(min(max(1, limit), 200))
        .all()
    )
    return [_to_out(msg, cust) for msg, cust in rows]


def _to_out(msg: OutreachMessage, cust: Customer) -> OutreachMessageOut:
    return OutreachMessageOut(
        id=msg.id,
        customer_id=msg.customer_id,
        customer_external_id=cust.external_id,
        customer_name=cust.name,
        template_key=msg.template_key,
        channel=msg.channel,
        subject=msg.subject,
        status=msg.status,
        created_at=msg.created_at,
    )

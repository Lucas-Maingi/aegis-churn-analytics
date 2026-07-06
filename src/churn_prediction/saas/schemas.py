"""
SaaS API Schemas
================
Pydantic request/response models for auth, customer management, and outreach.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


# ── Auth ─────────────────────────────────────────────────────────────────────


class SignupRequest(BaseModel):
    organization_name: str = Field(..., min_length=2, max_length=255)
    industry: str = Field("telecom", max_length=100)
    full_name: str = Field("", max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    organization_name: str
    email: str
    full_name: str


# ── Customers ────────────────────────────────────────────────────────────────


class CustomerSummary(BaseModel):
    id: int
    external_id: str
    name: str
    email: str
    tenure: int
    monthly_charges: float
    contract: str
    churn_probability: Optional[float] = None
    risk_tier: Optional[str] = None


class CustomerDetail(CustomerSummary):
    features: dict
    explanations: Optional[list] = None
    scored_at: Optional[datetime] = None


class TierCounts(BaseModel):
    HIGH: int = 0
    MEDIUM: int = 0
    LOW: int = 0


class CustomerListResponse(BaseModel):
    items: List[CustomerSummary]
    total: int
    page: int
    page_size: int
    tier_counts: TierCounts
    revenue_at_risk: float = Field(
        0.0, description="Sum of monthly charges across HIGH-risk customers"
    )


class UploadResponse(BaseModel):
    imported: int
    updated: int
    scored: int
    skipped: int
    errors: List[str] = []


# ── Outreach ─────────────────────────────────────────────────────────────────


class OutreachTemplate(BaseModel):
    key: str
    label: str
    description: str
    subject: str
    body: str


class SendOutreachRequest(BaseModel):
    customer_id: int
    template_key: str
    subject: Optional[str] = None
    body: Optional[str] = None


class OutreachMessageOut(BaseModel):
    id: int
    customer_id: int
    customer_external_id: str
    customer_name: str
    template_key: str
    channel: str
    subject: str
    status: str
    created_at: datetime


class SendOutreachResponse(BaseModel):
    message: OutreachMessageOut
    detail: str

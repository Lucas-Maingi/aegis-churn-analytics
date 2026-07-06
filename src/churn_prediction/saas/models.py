"""
SaaS ORM Models
===============
Multi-tenant entities: organizations (the businesses using Aegis), their
users, the customers they upload for scoring, and outreach messages sent
to at-risk customers.
"""

import secrets
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from churn_prediction.saas.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_api_key() -> str:
    return f"aeg_{secrets.token_urlsafe(32)}"


class Organization(Base):
    """A business (ISP, MVNO, telecom operator) using the platform."""

    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    industry: Mapped[str] = mapped_column(String(100), default="telecom")
    api_key: Mapped[str] = mapped_column(String(64), unique=True, default=_new_api_key)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    users: Mapped[list["User"]] = relationship(back_populates="organization")
    customers: Mapped[list["Customer"]] = relationship(back_populates="organization")


class User(Base):
    """A dashboard login belonging to an organization."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    organization: Mapped["Organization"] = relationship(back_populates="users")


class Customer(Base):
    """An end-customer of an organization, with their latest churn score.

    ``features`` holds the customer's attributes mapped to the model's
    Telco-shaped schema; ``explanations`` holds the top SHAP drivers from
    the most recent scoring pass.
    """

    __tablename__ = "customers"
    __table_args__ = (
        UniqueConstraint("org_id", "external_id", name="uq_customer_org_external"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Optional contact details for outreach
    name: Mapped[str] = mapped_column(String(255), default="")
    email: Mapped[str] = mapped_column(String(255), default="")

    # Model-schema features (raw, pre-preprocessing)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Denormalized fields for fast filtering/sorting
    tenure: Mapped[int] = mapped_column(Integer, default=0)
    monthly_charges: Mapped[float] = mapped_column(Float, default=0.0)
    contract: Mapped[str] = mapped_column(String(50), default="")

    # Latest scoring results
    churn_probability: Mapped[float] = mapped_column(Float, nullable=True)
    risk_tier: Mapped[str] = mapped_column(String(10), nullable=True)
    explanations: Mapped[list] = mapped_column(JSON, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    organization: Mapped["Organization"] = relationship(back_populates="customers")
    outreach_messages: Mapped[list["OutreachMessage"]] = relationship(
        back_populates="customer"
    )


class OutreachMessage(Base):
    """A retention message sent (or simulated) to a customer."""

    __tablename__ = "outreach_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False)

    template_key: Mapped[str] = mapped_column(String(50), nullable=False)
    channel: Mapped[str] = mapped_column(String(20), default="email")
    subject: Mapped[str] = mapped_column(String(500), default="")
    body: Mapped[str] = mapped_column(Text, default="")

    # 'sent' via a real provider, 'simulated' when no provider is configured,
    # 'failed' when the provider rejected the request.
    status: Mapped[str] = mapped_column(String(20), default="simulated")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    customer: Mapped["Customer"] = relationship(back_populates="outreach_messages")

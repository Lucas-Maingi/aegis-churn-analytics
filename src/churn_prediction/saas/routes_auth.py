"""
Auth Routes
===========
Organization signup (creates the org + owner user) and login, both
returning a JWT for the dashboard session.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from churn_prediction.saas.db import get_db
from churn_prediction.saas.deps import get_current_user
from churn_prediction.saas.models import Organization, User
from churn_prediction.saas.schemas import AuthResponse, LoginRequest, SignupRequest
from churn_prediction.saas.security import (
    create_access_token,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    """Create an organization and its first user, returning a session token."""
    existing = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    org = Organization(name=payload.organization_name, industry=payload.industry)
    db.add(org)
    db.flush()  # assign org.id

    user = User(
        org_id=org.id,
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
    )
    db.add(user)
    db.commit()

    logger.info("New organization signed up: %s (org_id=%d)", org.name, org.id)
    return AuthResponse(
        access_token=create_access_token(user.id, org.id, user.email),
        organization_name=org.name,
        email=user.email,
        full_name=user.full_name,
    )


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate an existing user and return a session token."""
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    org = db.get(Organization, user.org_id)
    return AuthResponse(
        access_token=create_access_token(user.id, user.org_id, user.email),
        organization_name=org.name if org else "",
        email=user.email,
        full_name=user.full_name,
    )


@router.get("/me", response_model=AuthResponse)
def me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return the current session's identity (token is echoed back empty)."""
    org = db.get(Organization, user.org_id)
    return AuthResponse(
        access_token="",
        organization_name=org.name if org else "",
        email=user.email,
        full_name=user.full_name,
    )

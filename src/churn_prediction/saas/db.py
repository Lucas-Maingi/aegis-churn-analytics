"""
Database Engine & Session
=========================
SQLAlchemy setup for the multi-tenant SaaS layer.

Uses ``DATABASE_URL`` when provided (e.g. a Supabase Postgres connection
string in production) and falls back to a local SQLite file for
zero-configuration development.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from churn_prediction import config

DEFAULT_SQLITE_URL = f"sqlite:///{config.PROJECT_ROOT / 'aegis.db'}"

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)

# SQLite needs check_same_thread=False to be shared across FastAPI workers.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all SaaS ORM models."""


def init_db() -> None:
    """Create all tables that do not yet exist."""
    # Import models so they register against Base before create_all.
    from churn_prediction.saas import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency yielding a scoped database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

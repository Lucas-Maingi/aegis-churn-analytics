"""
Demo Seeding
============
Populates an empty database with a demo organization and the sample ISP
customer CSV, fully scored — so the hosted demo shows a working dashboard
without requiring visitors to sign up and upload data first.

Activated by setting ``SEED_DEMO=1`` (used by the Hugging Face Space image);
never runs if any organization already exists.
"""

import csv
import logging
from datetime import datetime, timezone

from churn_prediction import config
from churn_prediction.saas.db import SessionLocal
from churn_prediction.saas.mapping import map_row
from churn_prediction.saas.models import Customer, Organization, User
from churn_prediction.saas.scoring import score_customers
from churn_prediction.saas.security import hash_password

logger = logging.getLogger(__name__)

DEMO_ORG_NAME = "Savannah Fiber (Demo)"
DEMO_EMAIL = "demo@aegis.app"
DEMO_PASSWORD = "aegis-demo-2026"

_SAMPLE_CSV = config.PROJECT_ROOT / "docs" / "sample_customers.csv"

_SAMPLE_MAPPING = {
    "external_id": "subscriber_id",
    "name": "customer_name",
    "email": "email_address",
    "tenure": "months_active",
    "MonthlyCharges": "monthly_fee",
    "InternetService": "plan_type",
    "Contract": "contract_type",
    "PaymentMethod": "payment_method",
    "PaperlessBilling": "paperless_billing",
    "TechSupport": "tech_support_addon",
}


def _seed_outcomes(customers) -> None:
    """Assign realistic ground-truth outcomes to most demo customers.

    Outcomes are drawn from each customer's scored probability so the
    scorecard reflects a real (imperfect) model. A deliberate tenant-specific
    twist is added — in this ISP, short-tenure DSL customers churn more than
    the Telco-trained base model expects — so a per-tenant retrain has a real
    signal to learn and can legitimately beat the base model. ~1 in 4
    customers is left unresolved to show that state in the UI.
    """
    import random

    rng = random.Random(2026)
    now = datetime.now(timezone.utc)

    for cust in customers:
        if rng.random() < 0.25:
            continue  # leave unresolved

        p = (cust.churn_probability or 0.0) / 100.0
        # Tenant twist the base model under-weights: short-tenure DSL churns more.
        feats = cust.features
        if feats.get("InternetService") == "DSL" and cust.tenure <= 12:
            p = min(1.0, p + 0.35)

        cust.actual_outcome = "churned" if rng.random() < p else "retained"
        cust.outcome_recorded_at = now


def seed_demo_org(model, preprocessor, feature_names) -> bool:
    """Create the demo org with scored sample customers. Returns True if seeded."""
    if not _SAMPLE_CSV.exists():
        logger.warning("Demo seed skipped: %s not found.", _SAMPLE_CSV)
        return False

    db = SessionLocal()
    try:
        if db.query(Organization).count() > 0:
            return False

        org = Organization(name=DEMO_ORG_NAME, industry="ISP")
        db.add(org)
        db.flush()
        db.add(
            User(
                org_id=org.id,
                email=DEMO_EMAIL,
                password_hash=hash_password(DEMO_PASSWORD),
                full_name="Demo Operator",
            )
        )

        with open(_SAMPLE_CSV, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        customers = []
        for row in rows:
            mapped = map_row(row, _SAMPLE_MAPPING)
            if not mapped["external_id"]:
                continue
            feats = mapped["features"]
            customers.append(
                Customer(
                    org_id=org.id,
                    external_id=mapped["external_id"],
                    name=mapped["name"],
                    email=mapped["email"],
                    features=feats,
                    tenure=int(feats.get("tenure") or 0),
                    monthly_charges=float(feats.get("MonthlyCharges") or 0.0),
                    contract=str(feats.get("Contract") or ""),
                )
            )
        db.add_all(customers)

        score_customers(customers, model, preprocessor, feature_names)
        _seed_outcomes(customers)
        db.commit()
        logger.info(
            "Seeded demo org '%s' with %d scored customers (login: %s).",
            DEMO_ORG_NAME, len(customers), DEMO_EMAIL,
        )
        return True
    except Exception:
        db.rollback()
        logger.exception("Demo seeding failed; continuing without demo data.")
        return False
    finally:
        db.close()

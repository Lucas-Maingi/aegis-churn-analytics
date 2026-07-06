"""
Retention Outreach
==================
Offer templates and email delivery. Sends through Resend when
``RESEND_API_KEY`` is configured; otherwise records the message as
'simulated' so the full workflow is testable without an email account.
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
OUTREACH_FROM = os.getenv("OUTREACH_FROM_EMAIL", "retention@aegis-analytics.app")

TEMPLATES = [
    {
        "key": "discount_offer",
        "label": "Loyalty discount",
        "description": "A time-limited discount on their current plan — the "
                       "strongest lever for price-driven churn risk.",
        "subject": "A thank-you from {company}: 20% off your next 3 months",
        "body": (
            "Hi {name},\n\n"
            "You've been with {company} for {tenure} months, and we want to "
            "make sure you're getting the best value from your plan.\n\n"
            "As a thank-you, here's 20% off your next 3 months — no action "
            "needed, it will be applied to your next bill.\n\n"
            "If there's anything about your service we can improve, just reply "
            "to this email.\n\n"
            "— The {company} team"
        ),
    },
    {
        "key": "contract_upgrade",
        "label": "Annual plan offer",
        "description": "Move a month-to-month customer onto a discounted "
                       "annual contract — contract length is the #1 churn driver.",
        "subject": "{name}, lock in a better rate with an annual plan",
        "body": (
            "Hi {name},\n\n"
            "Customers on our annual plan save up to 25% compared to paying "
            "month-to-month — and your rate stays locked for the full year.\n\n"
            "Switching takes one click and your service continues unchanged.\n\n"
            "Want us to set it up? Just reply 'yes' to this email.\n\n"
            "— The {company} team"
        ),
    },
    {
        "key": "check_in",
        "label": "Personal check-in",
        "description": "A human, no-offer check-in from the team — best for "
                       "customers whose risk is driven by support or service issues.",
        "subject": "How is your service, {name}?",
        "body": (
            "Hi {name},\n\n"
            "Just checking in — we noticed you might not be getting the most "
            "out of your service with {company}, and we'd love to fix that.\n\n"
            "Is everything working the way you expect? Anything slow, "
            "unreliable, or confusing? Reply to this email and a real person "
            "on our team will take it from there.\n\n"
            "— The {company} team"
        ),
    },
]


def get_template(key: str) -> Optional[dict]:
    for tpl in TEMPLATES:
        if tpl["key"] == key:
            return tpl
    return None


def render(text: str, *, name: str, company: str, tenure: int) -> str:
    """Fill template placeholders with customer/org values."""
    return text.format(
        name=name or "there",
        company=company or "our team",
        tenure=tenure,
    )


def send_email(to_email: str, subject: str, body: str) -> str:
    """Deliver an email. Returns the resulting status string.

    'sent' on successful Resend delivery, 'simulated' when no provider is
    configured, 'failed' when the provider rejects the request.
    """
    if not RESEND_API_KEY:
        logger.info("RESEND_API_KEY not set — outreach to %s recorded as simulated.", to_email)
        return "simulated"

    try:
        response = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={
                "from": OUTREACH_FROM,
                "to": [to_email],
                "subject": subject,
                "text": body,
            },
            timeout=15.0,
        )
        if response.status_code in (200, 201):
            return "sent"
        logger.error("Resend rejected email to %s: %s %s", to_email,
                     response.status_code, response.text)
        return "failed"
    except httpx.HTTPError as exc:
        logger.error("Resend request failed for %s: %s", to_email, exc)
        return "failed"

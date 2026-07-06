"""
Authentication Primitives
=========================
Password hashing (PBKDF2-SHA256, stdlib only) and JWT issuance/validation
for dashboard sessions.
"""

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt

JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-insecure-secret-change-in-production")
JWT_ALGORITHM = "HS256"
TOKEN_TTL_DAYS = 7

_PBKDF2_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    """Hash a password with PBKDF2-SHA256 and a random per-user salt."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
    return "pbkdf2${}${}${}".format(
        _PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode(),
        base64.b64encode(dk).decode(),
    )


def verify_password(password: str, stored: str) -> bool:
    """Check a plaintext password against a stored PBKDF2 hash."""
    try:
        _, iterations, salt_b64, hash_b64 = stored.split("$")
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, int(iterations))
        return hmac.compare_digest(dk, expected)
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: int, org_id: int, email: str) -> str:
    """Issue a signed JWT for a dashboard session."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "org": org_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(days=TOKEN_TTL_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT; raises jwt.PyJWTError on failure."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

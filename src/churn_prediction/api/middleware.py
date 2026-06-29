"""
API Middleware & Rate Limiting
==============================
Implements a memory-based sliding-window rate limiter and global exception handlers
to protect endpoints and standardize error responses.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """A sliding-window rate limiter stored in application memory.

    Has zero external dependencies, making it ideal for free-tier deployments.
    """

    def __init__(self, requests_limit: int = 100, window_seconds: int = 60):
        self.limit = requests_limit
        self.window_seconds = window_seconds
        self._history: Dict[str, List[float]] = defaultdict(list)

    def check_limit(self, key: str) -> bool:
        """Verify if a client key (e.g. IP) has exceeded the rate limit.

        Returns True if the request is allowed (not limited), False if blocked.
        """
        now = time.time()
        # Retrieve and prune expired request timestamps
        timestamps = self._history[key]
        valid_timestamps = [t for t in timestamps if now - t < self.window_seconds]
        self._history[key] = valid_timestamps

        if len(valid_timestamps) >= self.limit:
            return False

        # Register this request
        self._history[key].append(now)
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Global ASGI middleware enforcing rate limits on clients.

    Limits requests based on the client IP address. Excludes health and docs routes.
    """

    def __init__(
        self,
        app,
        requests_limit: int = 60,
        window_seconds: int = 60,
    ):
        super().__init__(app)
        self.limiter = InMemoryRateLimiter(
            requests_limit=requests_limit, window_seconds=window_seconds
        )

        # Routes that are excluded from rate limiting constraints
        self.excluded_prefixes = ("/health", "/docs", "/redoc", "/openapi.json")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Bypass rate limiter for docs or health checks
        if any(path.startswith(prefix) for prefix in self.excluded_prefixes):
            return await call_next(request)

        # Get client IP address
        client_ip = request.client.host if request.client else "unknown"

        if not self.limiter.check_limit(client_ip):
            logger.warning(
                "Rate limit exceeded for client IP '%s' on path '%s'",
                client_ip,
                path,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": (
                        f"Rate limit exceeded. Maximum allowed: {self.limiter.limit} "
                        f"requests per {self.limiter.window_seconds} seconds."
                    )
                },
            )

        return await call_next(request)


# ── Exception Handlers ───────────────────────────────────────────────────────


async def custom_validation_exception_handler(request: Request, exc):
    """Standardizes Pydantic input validation failures with structured details."""
    logger.warning("Request schema validation failed for path '%s'", request.url.path)
    # Extract only clean messages without Pydantic traceback overhead
    errors = []
    for err in exc.errors():
        field_path = " -> ".join(str(loc) for loc in err.get("loc", []))
        errors.append(f"{field_path}: {err.get('msg', 'invalid value')}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Input validation error. Check feature parameters.",
            "validation_errors": errors,
        },
    )


async def global_exception_handler(request: Request, exc: Exception):
    """Fallback handler to intercept unhandled crashes and mask server tracebacks."""
    logger.error(
        "Unhandled exception crashed request '%s %s': %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred. Please contact api support."
        },
    )

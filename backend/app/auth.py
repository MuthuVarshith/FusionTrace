"""Optional API-key authentication.

Set ``FUSIONTRACE_API_KEYS`` to a comma-separated list of keys to turn it on.
When it is unset or empty, authentication is off and local use needs nothing.

When on, every ``/api/*`` request must send ``X-API-Key: <key>``. Each scan
stores an owner id derived from the key (a truncated SHA-256 digest, never the
key itself), and scan listing, reports, heatmaps and feedback are scoped to the
caller's owner id. Scans created while authentication was off have no owner
and are not visible to key holders.

The variable is read on every request, so tests can toggle it with an
environment patch; parsing a short string per request is negligible.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass

from fastapi import HTTPException, Request

API_KEY_HEADER = "X-API-Key"
_OWNER_PREFIX = b"fusiontrace-owner:"


@dataclass(frozen=True)
class Caller:
    owner: str | None  # owner id when auth is on, None when auth is off
    rate_key: str  # bucket for rate limiting: owner id, or client IP when auth is off


def configured_keys() -> list[str]:
    raw = os.getenv("FUSIONTRACE_API_KEYS", "")
    return [key.strip() for key in raw.split(",") if key.strip()]


def auth_enabled() -> bool:
    return bool(configured_keys())


def owner_id(api_key: str) -> str:
    """Stable, non-reversible owner id for a key. The key itself is never stored."""
    return hashlib.sha256(_OWNER_PREFIX + api_key.encode("utf-8")).hexdigest()[:32]


def _matches(candidate: str, keys: list[str]) -> bool:
    supplied = candidate.encode("utf-8")
    # Compare against every key so timing does not reveal which one matched.
    matched = False
    for key in keys:
        matched |= hmac.compare_digest(supplied, key.encode("utf-8"))
    return matched


def require_caller(request: Request) -> Caller:
    """FastAPI dependency for ``/api/*`` routes."""
    keys = configured_keys()
    if not keys:
        host = request.client.host if request.client else "unknown"
        return Caller(owner=None, rate_key=f"ip:{host}")
    supplied = request.headers.get(API_KEY_HEADER, "")
    if not supplied or not _matches(supplied, keys):
        raise HTTPException(status_code=401, detail="A valid API key is required.", headers={"WWW-Authenticate": API_KEY_HEADER})
    owner = owner_id(supplied)
    return Caller(owner=owner, rate_key=f"key:{owner}")

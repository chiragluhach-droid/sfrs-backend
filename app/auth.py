"""Minimal admin auth: one shared password -> signed, expiring bearer token.

- ADMIN_PASSWORD  set it to enable auth (required when deployed on Render)
- SECRET_KEY      signs tokens; if unset a random key is used (tokens reset on restart)
- TOKEN_HOURS     token lifetime, default 12
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

from fastapi import Request
from fastapi.responses import JSONResponse

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
TOKEN_HOURS = float(os.getenv("TOKEN_HOURS", "12"))
AUTH_ENABLED = bool(ADMIN_PASSWORD)

PUBLIC_PATHS = {"/api/health", "/api/login", "/docs", "/openapi.json", "/"}

if not AUTH_ENABLED and (os.getenv("RENDER") or os.getenv("RAILWAY_ENVIRONMENT")):
    raise RuntimeError("ADMIN_PASSWORD must be set when deployed. Refusing to start an open API.")


def _sign(payload: bytes) -> str:
    return hmac.new(SECRET_KEY.encode(), payload, hashlib.sha256).hexdigest()


def issue_token() -> tuple[str, int]:
    exp = int(time.time() + TOKEN_HOURS * 3600)
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode())
    return f"{payload.decode()}.{_sign(payload)}", exp


def verify_token(token: str) -> bool:
    try:
        payload, sig = token.split(".", 1)
        if not hmac.compare_digest(_sign(payload.encode()), sig):
            return False
        return json.loads(base64.urlsafe_b64decode(payload))["exp"] > time.time()
    except Exception:
        return False


def check_password(password: str) -> bool:
    return AUTH_ENABLED and hmac.compare_digest(password.encode(), ADMIN_PASSWORD.encode())


async def auth_middleware(request: Request, call_next):
    if (
        not AUTH_ENABLED
        or request.method == "OPTIONS"
        or request.url.path in PUBLIC_PATHS
        or not request.url.path.startswith("/api/")
    ):
        return await call_next(request)
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer ") and verify_token(header[7:].strip()):
        return await call_next(request)
    return JSONResponse({"detail": "Not authenticated"}, status_code=401)

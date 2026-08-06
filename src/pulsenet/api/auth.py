"""
JWT authentication and role-based access control.
"""

from __future__ import annotations

import json
import os
import time
import uuid as _uuid_mod

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from pulsenet.logger import get_logger

log = get_logger(__name__)

security = HTTPBearer(auto_error=False)

_JWT_SECRET = os.environ.get("PULSENET_JWT_SECRET")
if not _JWT_SECRET or len(_JWT_SECRET) < 32:
    if os.environ.get("PULSENET_ENV") == "testing":
        _JWT_SECRET = "pulsenet-test-secret-not-for-production"  # noqa: S105
    else:
        log.critical("PULSENET_JWT_SECRET is required")
        raise RuntimeError("PULSENET_JWT_SECRET must be set.")

_JWT_ALGORITHM = "HS256"
_JWT_EXPIRY_MIN = 60

# Issuer and audience claims are required for strict JWT validation.
# Set PULSENET_JWT_ISSUER and PULSENET_JWT_AUDIENCE in the deployment environment.
# Defaults are safe for development but MUST be overridden in production.
_JWT_ISSUER: str = os.environ.get("PULSENET_JWT_ISSUER", "pulsenet-api")
_JWT_AUDIENCE: str = os.environ.get("PULSENET_JWT_AUDIENCE", "pulsenet-clients")

# Role -> allowed endpoints
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"predict", "train", "health", "audit", "verify"},
    "engineer": {"predict", "train", "health"},
    "operator": {"predict", "health"},
}


def _hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    try:
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    except Exception as e:
        log.error(f"Password verification error: {e}")
        return False


def _load_users() -> dict:
    """Load users from PULSENET_USERS JSON env var or secure defaults."""
    users_json = os.environ.get("PULSENET_USERS")
    if users_json:
        try:
            return json.loads(users_json)
        except Exception as e:
            log.error(f"Failed to parse PULSENET_USERS: {e}")
            raise RuntimeError("Invalid PULSENET_USERS JSON configuration") from e

    # If no users JSON provided, but an admin password is set,
    # create a single fallback admin
    admin_pw = os.environ.get("PULSENET_ADMIN_PASSWORD")
    if admin_pw:
        return {"admin": {"hashed_password": _hash_password(admin_pw), "role": "admin"}}

    # Final fallback: Require explicit configuration
    log.error("No users configured! Authentication will fail.")
    return {}


USER_DB: dict[str, dict] = _load_users()


def create_token(username: str, role: str) -> tuple[str, int]:
    """Create a JWT token. Returns (token, expiry_minutes).

    The token includes the following claims for strict validation:
    - ``sub``: subject (username)
    - ``role``: application-specific role claim
    - ``iat``: issued-at timestamp
    - ``exp``: expiry timestamp
    - ``nbf``: not-before timestamp (same as ``iat`` — token is valid immediately)
    - ``iss``: issuer (``PULSENET_JWT_ISSUER`` env var, default ``pulsenet-api``)
    - ``aud``: audience (``PULSENET_JWT_AUDIENCE`` env var, default ``pulsenet-clients``)
    - ``jti``: unique token ID (UUID4) for anti-replay tracking
    """
    now = int(time.time())
    payload = {
        "sub": username,
        "role": role,
        "iat": now,
        "nbf": now,  # token is valid immediately; set future value to delay activation
        "exp": now + _JWT_EXPIRY_MIN * 60,
        "iss": _JWT_ISSUER,
        "aud": _JWT_AUDIENCE,
        "jti": str(_uuid_mod.uuid4()),  # unique token ID — enables per-token revocation
    }
    # _JWT_SECRET is guaranteed to be a string here due to fallback above
    token = jwt.encode(payload, str(_JWT_SECRET), algorithm=_JWT_ALGORITHM)
    return token, _JWT_EXPIRY_MIN


def verify_token(token: str) -> dict:
    """Decode and verify a JWT token.

    Validates all security-relevant claims:
    - ``exp``: must not be expired (validated by python-jose automatically)
    - ``nbf``: must not be used before the not-before time (validated by python-jose)
    - ``iss``: must match ``_JWT_ISSUER``
    - ``aud``: must match ``_JWT_AUDIENCE``

    Raises
    ------
    fastapi.HTTPException
        HTTP 401 on any validation failure.  The exception detail includes the
        specific reason so that callers can return structured error responses.
        Bare ``except Exception`` is intentionally avoided — only ``JWTError``
        and its subclasses are caught here; unexpected exceptions propagate.
    """
    try:
        payload = jwt.decode(
            token,
            str(_JWT_SECRET),
            algorithms=[_JWT_ALGORITHM],
            audience=_JWT_AUDIENCE,
            issuer=_JWT_ISSUER,
            options={
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iss": True,
                "verify_aud": True,
            },
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        ) from e


def authenticate_user(username: str, password: str) -> dict | None:
    """Validate credentials. Returns user dict or None."""
    user = USER_DB.get(username)
    if user:
        # Check both naming conventions (new 'hashed_password' vs old 'password_hash')
        hash_val = user.get("hashed_password") or user.get("password_hash")
        if hash_val and _verify_password(password, hash_val):
            return {"username": username, "role": user["role"]}
    return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """FastAPI dependency to extract current user from JWT."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    payload = verify_token(credentials.credentials)
    return {"username": payload["sub"], "role": payload["role"]}


def require_role(allowed_roles: set[str]):
    """FastAPI dependency factory for role-based access control."""

    async def check_role(user: dict = Depends(get_current_user)):
        if user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{user['role']}' not authorized. "
                    f"Required: {allowed_roles}"
                ),
            )
        return user

    return check_role


def require_permission(permission: str):
    """Check if user's role has a specific permission."""

    async def check_perm(user: dict = Depends(get_current_user)):
        user_perms = ROLE_PERMISSIONS.get(user["role"], set())
        if permission not in user_perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Permission '{permission}' not granted "
                    f"to role '{user['role']}'"
                ),
            )
        return user

    return check_perm

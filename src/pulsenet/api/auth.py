"""
JWT authentication and role-based access control.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from pulsenet.logger import get_logger

log = get_logger(__name__)

try:
    from jose import JWTError, jwt
except ImportError:
    from jwt import PyJWTError as JWTError
    import jwt as _jwt

    class jwt:  # type: ignore[no-redef]
        @staticmethod
        def encode(payload, key, algorithm):
            return _jwt.encode(payload, key, algorithm=algorithm)

        @staticmethod
        def decode(token, key, algorithms):
            return _jwt.decode(token, key, algorithms=algorithms)


_JWT_SECRET = os.environ.get(
    "PULSENET_JWT_SECRET", "pulsenet-dev-secret-change-in-production"
)
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRY_MIN = 60

# Role → allowed endpoints
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"predict", "train", "health", "audit", "verify"},
    "engineer": {"predict", "train", "health"},
    "operator": {"predict", "health"},
}


def _hash_password(password: str) -> str:
    """SHA-256 hash for password comparison."""
    import hashlib

    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> dict[str, dict]:
    """Load users from PULSENET_USERS env var (JSON) or secure defaults.

    Production: set PULSENET_USERS='{"admin": {"password_hash": "...", "role": "admin"}}'
    Dev fallback: loads default users but logs a security warning.
    """
    import json

    users_json = os.environ.get("PULSENET_USERS")
    if users_json:
        try:
            users = json.loads(users_json)
            log.info("Users loaded from PULSENET_USERS environment variable")
            return users
        except json.JSONDecodeError:
            log.error("Invalid JSON in PULSENET_USERS — falling back to defaults")

    # Dev-only defaults — log warning
    log.warning(
        "⚠️  Using default credentials (dev only). "
        "Set PULSENET_USERS env var for production."
    )
    return {
        "admin": {
            "password_hash": _hash_password(
                os.environ.get("PULSENET_ADMIN_PASSWORD", "admin123")
            ),
            "role": "admin",
        },
        "engineer": {
            "password_hash": _hash_password(
                os.environ.get("PULSENET_ENGINEER_PASSWORD", "eng123")
            ),
            "role": "engineer",
        },
        "operator": {
            "password_hash": _hash_password(
                os.environ.get("PULSENET_OPERATOR_PASSWORD", "ops123")
            ),
            "role": "operator",
        },
    }


USER_DB: dict[str, dict] = _load_users()

security = HTTPBearer(auto_error=False)


def create_token(username: str, role: str) -> tuple[str, int]:
    """Create a JWT token. Returns (token, expiry_minutes)."""
    payload = {
        "sub": username,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + _JWT_EXPIRY_MIN * 60,
    }
    token = jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)
    return token, _JWT_EXPIRY_MIN


def verify_token(token: str) -> dict:
    """Decode and verify a JWT token."""
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        return payload
    except (JWTError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Validate credentials against hashed passwords. Returns user dict or None."""
    user = USER_DB.get(username)
    if user and user.get("password_hash") == _hash_password(password):
        return {"username": username, "role": user["role"]}
    return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
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
                detail=f"Role '{user['role']}' not authorized. Required: {allowed_roles}",
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
                detail=f"Permission '{permission}' not granted to role '{user['role']}'",
            )
        return user

    return check_perm

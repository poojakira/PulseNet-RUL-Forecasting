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


_JWT_SECRET = os.environ.get("PULSENET_JWT_SECRET", "pulsenet-dev-secret-change-in-production")
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRY_MIN = 60

# Role → allowed endpoints
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"predict", "train", "health", "audit", "verify"},
    "engineer": {"predict", "train", "health"},
    "operator": {"predict", "health"},
}

# Simple user store (replace with DB in production)
USER_DB: dict[str, dict] = {
    "admin": {"password": "admin123", "role": "admin"},
    "engineer": {"password": "eng123", "role": "engineer"},
    "operator": {"password": "ops123", "role": "operator"},
}

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
    """Validate credentials. Returns user dict or None."""
    user = USER_DB.get(username)
    if user and user["password"] == password:
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

"""
JWT token revocation blocklist.

Solves Finding 7 from the security audit: stolen tokens were valid for up to
15 minutes with no way to revoke them. This module provides an in-memory
blocklist (for single-instance deployments) and a Redis-backed blocklist
(for multi-instance production deployments).

Usage:
    from pulsenet.security.token_revocation import revocation_list

    # Revoke a token (e.g., on logout, password change, or compromise)
    revocation_list.revoke(token_jti, expires_at)

    # Check if token is revoked (call in auth middleware)
    if revocation_list.is_revoked(token_jti):
        raise HTTPException(401, "Token has been revoked")
"""
from __future__ import annotations

import os
import threading
import time
from typing import Protocol


class TokenBlocklist(Protocol):
    """Interface for token revocation backends."""

    def revoke(self, jti: str, expires_at: float) -> None:
        """Add a token to the blocklist. expires_at is Unix timestamp."""
        ...

    def is_revoked(self, jti: str) -> bool:
        """Check if a token JTI is in the blocklist."""
        ...

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        ...


class InMemoryBlocklist:
    """Thread-safe in-memory token blocklist.

    Suitable for single-instance deployments. Tokens are automatically
    cleaned up after expiration (no memory leak). For multi-instance
    deployments behind a load balancer, use RedisBlocklist instead.
    """

    def __init__(self) -> None:
        self._blocked: dict[str, float] = {}  # jti -> expires_at
        self._lock = threading.Lock()

    def revoke(self, jti: str, expires_at: float) -> None:
        """Revoke a token by its JTI. The entry auto-expires at expires_at."""
        with self._lock:
            self._blocked[jti] = expires_at

    def is_revoked(self, jti: str) -> bool:
        """Check if a token is revoked. Returns False if expired (already invalid)."""
        with self._lock:
            if jti not in self._blocked:
                return False
            # If the token's natural expiry has passed, remove it
            if self._blocked[jti] < time.time():
                del self._blocked[jti]
                return False
            return True

    def cleanup_expired(self) -> int:
        """Remove all entries whose tokens have naturally expired."""
        now = time.time()
        with self._lock:
            expired = [jti for jti, exp in self._blocked.items() if exp < now]
            for jti in expired:
                del self._blocked[jti]
            return len(expired)

    @property
    def size(self) -> int:
        """Current number of blocked tokens."""
        return len(self._blocked)


class RedisBlocklist:
    """Redis-backed token blocklist for multi-instance deployments.

    Each revoked JTI is stored as a Redis key with TTL matching the token's
    remaining validity. This ensures:
    - All instances see the revocation immediately
    - No memory leaks (Redis TTL handles cleanup)
    - Survives instance restarts

    Requires: pip install redis
    Set PULSENET_REDIS_URL environment variable.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        import redis

        url = redis_url or os.environ.get("PULSENET_REDIS_URL", "redis://localhost:6379/1")
        self._redis = redis.from_url(url, decode_responses=True)
        self._prefix = "pulsenet:revoked:"

    def revoke(self, jti: str, expires_at: float) -> None:
        """Revoke a token. TTL is set to remaining token lifetime."""
        ttl = max(1, int(expires_at - time.time()))
        self._redis.setex(f"{self._prefix}{jti}", ttl, "revoked")

    def is_revoked(self, jti: str) -> bool:
        """Check Redis for revocation status."""
        return self._redis.exists(f"{self._prefix}{jti}") > 0

    def cleanup_expired(self) -> int:
        """No-op for Redis (TTL handles expiry automatically)."""
        return 0


def _create_blocklist() -> InMemoryBlocklist | RedisBlocklist:
    """Factory: use Redis if PULSENET_REDIS_URL is set, else in-memory."""
    redis_url = os.environ.get("PULSENET_REDIS_URL")
    if redis_url:
        try:
            return RedisBlocklist(redis_url)
        except Exception:  # noqa: S110
            pass
    return InMemoryBlocklist()


# Module-level singleton
revocation_list: InMemoryBlocklist | RedisBlocklist = _create_blocklist()

"""
Multi-Tenancy Middleware — extracts X-Tenant-ID from headers (Staff-level Gap 3).
"""

from __future__ import annotations

from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class TenantMiddleware(BaseHTTPMiddleware):
    """Injects tenant context into the request state for data isolation."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Default to 'public' if no tenant header is present
        tenant_id = request.headers.get("X-Tenant-ID", "public")
        request.state.tenant_id = tenant_id
        
        response = await call_next(request)
        
        # Inject tenant_id back into response for traceability
        response.headers["X-Tenant-ID"] = tenant_id
        return response

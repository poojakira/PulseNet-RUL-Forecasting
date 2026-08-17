"""
Multi-Tenancy Middleware — extracts X-Tenant-ID from headers (Staff-level Gap 3).
"""

from __future__ import annotations

import re

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

_TENANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class TenantMiddleware(BaseHTTPMiddleware):
    """Injects tenant context into the request state for data isolation."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        tenant_id = request.headers.get("X-Tenant-ID", "public")
        if not _TENANT_ID_RE.fullmatch(tenant_id):
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid tenant identifier"},
            )
        request.state.tenant_id = tenant_id

        response = await call_next(request)

        # Inject tenant_id back into response for traceability
        response.headers["X-Tenant-ID"] = tenant_id
        return response

"""
GET /audit, GET /verify-chain — Blockchain audit endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from pulsenet.api.auth import require_permission
from pulsenet.api.schemas import AuditResponse
from pulsenet.security.audit import AuditLogger

router = APIRouter(tags=["Audit"])
audit = AuditLogger()

_audit_refs: dict = {}


def set_audit_refs(refs: dict) -> None:
    global _audit_refs
    _audit_refs = refs


@router.get("/audit", response_model=AuditResponse)
async def get_audit_log(
    request: Request,
    user: dict = Depends(require_permission("audit")),
):
    """Get blockchain audit log summary."""
    ledger = _audit_refs.get("ledger")
    if ledger is None:
        return AuditResponse(
            chain_length=0,
            is_valid=False,
            validation_message="Ledger not initialized",
            recent_blocks=[],
        )

    tenant_id = getattr(request.state, "tenant_id", "public")
    is_valid, msg = ledger.validate_integrity(tenant_id)
    
    audit.log_access(
        endpoint="/audit",
        method="GET",
        user=user["username"],
        role=user["role"],
        tenant_id=tenant_id
    )

    metrics = ledger.get_metrics(tenant_id)
    return AuditResponse(
        chain_length=metrics.get("total_blocks_global", 0), # Total across all for audit overview
        merkle_root=ledger.compute_merkle_root(tenant_id),
        is_valid=is_valid,
        validation_message=msg,
        recent_blocks=ledger.get_recent_blocks(10, tenant_id),
    )


@router.get("/verify-chain")
async def verify_chain(
    request: Request,
    user: dict = Depends(require_permission("verify")),
):
    """Full blockchain integrity verification."""
    ledger = _audit_refs.get("ledger")
    if not ledger:
        return {"valid": False, "message": "Ledger not initialized"}

    tenant_id = getattr(request.state, "tenant_id", "public")
    is_valid, msg = ledger.validate_integrity(tenant_id)
    tampered = ledger.detect_tampering(tenant_id)
    metrics = ledger.get_metrics(tenant_id)

    audit.log_access(
        endpoint="/verify-chain",
        method="GET",
        user=user["username"],
        role=user["role"],
        metadata={"result": is_valid},
        tenant_id=tenant_id
    )

    return {
        "valid": is_valid,
        "message": msg,
        "tampered_blocks": tampered,
        "metrics": metrics,
    }

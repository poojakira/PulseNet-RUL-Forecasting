# Incident Response Playbook

## Scope

Use this playbook when telemetry, model artifacts, dependencies, signing keys, audit logs, or prediction behavior may be compromised.

## Triage

1. Freeze model promotion and rollback actions.
2. Preserve audit logs, CI logs, model hashes, data-lineage evidence, and deployment metadata.
3. Identify affected tenants, model versions, data windows, and API endpoints.
4. Compare current artifact hashes against the expected registry hashes.

## Containment

1. Revoke exposed API tokens or signing keys.
2. Disable suspicious model versions and route traffic to the last verified artifact.
3. Block suspicious clients or tenants at the API gateway.
4. Stop ingestion from telemetry sources with failed integrity checks.

## Eradication

1. Remove poisoned telemetry or compromised artifacts from promotion paths.
2. Patch vulnerable dependencies and rerun `pip-audit`, Bandit, tests, and verification scripts.
3. Rotate secrets and regenerate signing material outside git.
4. Rebuild model artifacts from verified data only.

## Recovery

1. Re-enable traffic gradually with drift, error-rate, and prediction-distribution monitoring.
2. Verify hash-chain continuity or document the exact break and restart point.
3. Publish an internal incident summary with root cause, blast radius, and follow-up controls.

## Lessons Learned

Track corrective actions for auth, RBAC, tenant isolation, artifact signing, dependency management, and monitoring gaps.

# 2026 ML Security Audit

Generated from local repository evidence. This file is intentionally conservative: it reports files, controls, and gaps that can be inspected in this repository.

## Scope

- Topic: Predictive maintenance ML serving
- Standards map: OWASP A03/A05, NIST AI RMF Manage
- Data provenance: NASA C-MAPSS predictive-maintenance data path where available.
- Git state at generation: dirty
- Test files detected: 13
- GitHub workflows detected: 1

## Evidence Found

- jwt
- rate limit
- audit
- artifact integrity
- prometheus

## Open Gaps

- No configured evidence-map gaps detected.

## Recruiter Signal

This repository should be evaluated by running its checked-in tests and CI/security gates. Do not cite benchmark numbers or production readiness unless the repo contains the command, artifact, and current passing validation needed to reproduce the claim.

## Rebuild

Run from the profile repository:

```bash
python tools/build_security_dashboard.py
```

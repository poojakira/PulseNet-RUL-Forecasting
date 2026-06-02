# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.1.x   | ✅ Current Release |
| < 2.1.0 | ❌ Legacy          |

Only the `main` branch and tagged releases >= 2.1.0 are supported for security fixes.

## Reporting a Vulnerability

If you discover a security vulnerability, please do NOT open a public issue. Use GitHub's [Private Vulnerability Reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability).

## Security Implementation

### 1. Authentication & RBAC
We use JWT (JSON Web Tokens) with RS256 or HS256 signing. Roles are strictly enforced:
- `ROLE_ADMIN`: Full access to audit logs and system verification.
- `ROLE_OPERATOR`: Can trigger training and model promotion.
- `ROLE_CONSUMER`: Prediction-only access.

### 2. Tenant Isolation
The `TenantMiddleware` ensures that every request is tagged with an `X-Tenant-ID`. Data access layers are restricted to the tenant context, preventing cross-tenant leakage in multi-tenant deployments.

### 3. Audit Logging
Every sensitive action (prediction, model change, login) is recorded in a hash-chained ledger. Each entry contains the hash of the previous entry, making the log immutable and verifiable via `verify.py`.

### 4. Data Lineage & Integrity
We verify the SHA-256 hashes of the NASA C-MAPSS datasets before any ingestion or training. This prevents supply-chain attacks targeting the training data.

### 5. Dependency Management
- All dependencies are pinned in `requirements.txt`.
- CI/CD runs `pip-audit` to detect known CVEs in the dependency tree.
- Static analysis with `bandit` is performed on every PR.

## Known Limitations

- **Safety-Criticality**: This project is a portfolio-grade demonstration. It has NOT been certified for use in flight-control or life-critical systems.
- **Local Secrets**: In development, `secret.key` is used for signing. In production, this must be replaced by a KMS (Key Management Service) or Vault.
- **Inference Latency**: Security middleware adds minimal overhead (~5-10ms), which may not be suitable for ultra-low-latency (<1ms) hard real-time requirements.

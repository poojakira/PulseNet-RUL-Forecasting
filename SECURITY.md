# Security Policy

## Status

This repository is archived and has no supported deployment or production
version. Historical source remains public for review, but no security maintenance
or response-time SLA is promised.

## Reporting

Use GitHub's private vulnerability reporting feature if it is enabled for this
repository. Otherwise, open a GitHub issue containing no exploit details or
sensitive data and request a private contact channel.

Do not test against systems you do not own or have explicit permission to test.
Do not deploy attacks, use credentials, or connect this code to equipment.

## Known dependency risk

A `pip-audit 2.10.1` scan of the Python 3.12 locked runtime graph on 2026-08-11
reported known advisories in the following packages:

- `ecdsa 0.19.2`
- `starlette 0.37.2`
- `mlflow 2.14.1`
- `protobuf 4.25.9`
- `pyarrow 15.0.2`
- `python-dotenv 1.0.1`
- `python-jose 3.3.0`
- `python-multipart 0.0.6`
- `torch 2.3.0`
- `transformers 4.40.0`

Some advisories had patched versions; others did not list one in the audit
output. None is waived or represented as remediated. The dependency stack is
retained only to preserve historical reproducibility and must not be treated as
safe for deployment.

## Safe review boundary

Prefer static source review. If execution is necessary, use a disposable,
unprivileged environment with no secrets, no production data, no mounted home
directory, and outbound networking disabled after dependencies are obtained.
Destroy the environment after use.

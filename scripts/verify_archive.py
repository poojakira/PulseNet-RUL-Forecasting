"""Fail when the archived repository regains unsupported product surfaces."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_TEXT = {
    "README.md": "not production-ready",
    "ARCHIVE.md": "**Decision:** ARCHIVE",
    "PORTFOLIO.md": "ARCHIVED",
}

FORBIDDEN_PATHS = (
    "BENCHMARK_RESULTS.md",
    "Dockerfile",
    "coverage.xml",
    "docker-compose.yml",
    "dashboard/index.html",
    "deploy/k8s/deployment.yaml",
    "deploy/terraform/main.tf",
    "docs/evidence/deep_rul_fd001.json",
    "docs/evidence/validation_metrics.svg",
    "docs/evidence/validation_results.json",
    "evidence_policy.json",
    "ledger_public.json",
    "provenance.json",
    "sarif_output.json",
    "sbom.json",
)

FORBIDDEN_DEPENDENCY_TEXT = (
    "git+https://github.com/poojakira/attack-v19-core",
    '"mitreattack-python',
    '"stix2',
)


def main() -> int:
    failures: list[str] = []

    for relative, marker in REQUIRED_TEXT.items():
        path = ROOT / relative
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            failures.append(f"{relative} must contain {marker!r}")

    for relative in FORBIDDEN_PATHS:
        if (ROOT / relative).exists():
            failures.append(f"unsupported deployment surface restored: {relative}")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for marker in FORBIDDEN_DEPENDENCY_TEXT:
        if marker in pyproject:
            failures.append(f"removed dependency restored: {marker}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1

    print("Archive boundaries verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Evidence policy

This repository follows a **no fabricated metrics** rule for documentation and hiring materials.

## Rules

1. **README files do not contain performance, accuracy, latency, or security efficacy numbers** unless they are copied from a committed artifact file produced by a script in the same commit.
2. **All measurements** must be reproducible with documented commands (for example `python verify.py`, `python scripts/run_validation.py`, `make eval`).
3. **Randomness** uses explicit seeds in verification scripts; do not change seeds without updating committed artifacts.
4. **Illustrative API examples** must be labeled as structural examples, not recorded benchmark runs.
5. **CI** must run the same verification commands that contributors are told to run locally, or a documented subset with equivalent checks.

## Where metrics live

| Type | Location |
|------|----------|
| Verification output | `artifacts/`, `docs/evidence/`, or `results/` JSON/MD/SVG from scripts |
| Threat boundaries | `docs/THREAT_MODEL.md` |
| Data lineage | `docs/DATA_LINEAGE.md` or README data policy section |

If a metric is not in those paths after running the documented command, do not cite it.

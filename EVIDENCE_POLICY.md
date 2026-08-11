# Evidence policy

This archived repository follows a **no fabricated metrics** rule.

## Rules

1. **README files do not contain performance, accuracy, latency, or security efficacy numbers** unless they are copied from a committed artifact file produced by a script in the same commit.
2. **All measurements** must be reproducible with documented commands (for example `python verify.py`, `python scripts/run_validation.py`, `make eval`).
3. **Randomness** uses explicit seeds in verification scripts; do not publish a
   result without recording the seed and full run context.
4. **Illustrative API examples** must be labeled as structural examples, not recorded benchmark runs.
5. **Archive status** does not make historical results valid. Regenerate and
   independently review evidence before citing any value.

## Where metrics live

| Type | Location |
|------|----------|
| Verification output | Local path outside the repository until independently reviewed |
| Archive boundaries | `ARCHIVE.md` and `SECURITY_AUDIT.md` |
| Data lineage | `docs/DATA_LINEAGE.md` or README data policy section |

If a metric is not in those paths after running the documented command, do not cite it.

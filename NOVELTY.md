# Novelty, Data & Production-Readiness — honest, skeptical assessment

## Genuinely useful here

- **RUL forecasting wrapped in real serving-security controls**: JWT/RBAC,
  tenant isolation, a tamper-evident hash-chained audit ledger (now optionally
  HMAC-keyed), an FDIA/anomaly detector, and inference-time adversarial
  telemetry. Coupling prognostics with these controls is uncommon and useful.
- **Dataset choice is honest**: NASA C-MAPSS (FD001–FD004) is the industry
  standard for RUL. This is the right benchmark.

## Corrected over-claims (be skeptical)

- **"Blockchain" / "FAA-Spec Blockchain" was misleading.** The ledger is a
  SHA-256 **hash chain in local JSON files** — tamper-*evident*, not
  tamper-*proof*. There is no distributed consensus, no PoW/PoS/BFT, and an
  attacker with filesystem write access can rewrite it unless the HMAC key
  (`PULSENET_LEDGER_HMAC_KEY`) is set and/or the tip is anchored externally.
  Docstrings and messages now say "hash-chained audit ledger"; symbol names are
  retained for compatibility but a future rename of the `blockchain.py` module /
  `BlockchainConfig` is recommended. Do not market this as a blockchain or as
  meeting an "FAA specification" — no such certification exists here.
- **Adversarial telemetry is expensive and model-contract-specific.**
  Finite-difference gradients cost O(features) forward passes per row. A torch
  autograd fast-path now exists, and the previously silent `model.score()`
  assumption (which PyTorch modules do not satisfy) now raises a clear error.
  The guard is still a *sampled* heuristic, not a robustness proof.

## Data / evaluation discipline required (the real risks)

| Item | Why it matters |
|------|----------------|
| **Temporal-only split** | Random train/test split leaks future→past and is the #1 RUL evaluation mistake. Validate chronologically. |
| Sensor-noise robustness set | Inject realistic dropout/drift/bias into C-MAPSS to test robustness. |
| Real maintenance logs | Ground-truth failure/maintenance events beyond the simulated benchmark. |
| Cross-fleet transfer | Train turbofan, test a different engine type, to measure generalization. |

## Known gaps

- Ledger has archival (`archive_chain`) but no automatic rotation/checkpoint;
  true rotation needs external tip anchoring.
- No external anchoring wired by default (QLDB/S3 Object-Lock/transparency log).
- Adversarial telemetry gives detection signals, not certified robustness.

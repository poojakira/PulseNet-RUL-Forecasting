"""PulseNet verification on official NASA C-MAPSS FD001 data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import bcrypt  # noqa: E402


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


os.environ.setdefault("PULSENET_ENV", "testing")
os.environ.setdefault("PULSENET_JWT_SECRET", "verify-script-dev-key-not-for-prod")
os.environ.setdefault(
    "PULSENET_USERS",
    json.dumps(
        {
            "admin": {
                "hashed_password": _hash_password("admin123"),
                "role": "admin",
            },
            "operator": {
                "hashed_password": _hash_password("ops123"),
                "role": "operator",
            },
        }
    ),
)

print("=" * 60)
print("PULSENET OFFICIAL DATA VERIFICATION")
print("=" * 60)

print("\n[1/5] Loading official NASA C-MAPSS FD001 archive...")
from pulsenet.models.isolation_forest import IsolationForestModel  # noqa: E402
from pulsenet.pipeline.official_cmapss import load_official_fd001  # noqa: E402
from pulsenet.pipeline.preprocessing import (  # noqa: E402
    compute_rolling_features,
    create_labels,
    get_feature_columns,
    normalize,
)

fd001 = load_official_fd001(
    "data/official", max_train_rows=4000, max_test_rows=None, download=False
)
print(f"  Source: {fd001.source_url}")
print(f"  Landing page: {fd001.landing_page}")
print(f"  Archive SHA256: {fd001.archive_sha256}")
print(f"  Train rows used: {len(fd001.train)}")
print(f"  Test rows used: {len(fd001.test)}")

print("\n[2/5] Training Isolation Forest on official sensor features...")
train_df = compute_rolling_features(fd001.train.copy())
test_df = compute_rolling_features(fd001.test.copy())
train_df, test_df, _ = normalize(train_df, test_df)
feature_cols = get_feature_columns(train_df)
x_test = test_df[feature_cols].to_numpy()
y_test = create_labels(test_df, fd001.rul, failure_threshold=125)

healthy_train = train_df[train_df["time_in_cycles"] <= 40][feature_cols].to_numpy()
t0 = time.perf_counter()
model = IsolationForestModel(n_estimators=50, contamination=0.12)
model.train(healthy_train)
train_seconds = time.perf_counter() - t0
metrics = model.evaluate(x_test, y_test)
print(f"  Features: {len(feature_cols)}")
print(f"  Train seconds: {train_seconds:.3f}")
print(f"  F1: {metrics['f1']:.3f}")
print(f"  Precision: {metrics['precision']:.3f}")
print(f"  Recall: {metrics['recall']:.3f}")

print("\n[3/5] Testing JWT auth and tenant header traceability...")
from fastapi.testclient import TestClient  # noqa: E402

from pulsenet.api.app import create_app  # noqa: E402
from pulsenet.api.auth import create_token, verify_token  # noqa: E402

token, expiry = create_token("operator", "operator")
payload = verify_token(token)
client = TestClient(create_app())
resp = client.get("/health", headers={"X-Tenant-ID": "official-nasa"})
assert resp.status_code == 200
assert resp.headers["X-Tenant-ID"] == "official-nasa"
print(f"  Token subject: {payload['sub']}")
print(f"  Token role: {payload['role']}")
print(f"  Expires in: {expiry}min")

print("\n[4/5] Testing hash-chain audit ledger...")
from pulsenet.security.audit import AuditLogger  # noqa: E402

ledger = AuditLogger()
ledger.log_access(
    "/predict",
    "POST",
    user="operator",
    role="operator",
    metadata={"model": "isolation_forest", "dataset": "NASA C-MAPSS FD001"},
    tenant_id="official-nasa",
)
valid, corrupt_count = ledger.verify_integrity()
print(f"  Chain integrity: {'VALID' if valid else 'BROKEN'}")
print(f"  Corrupt entries: {corrupt_count}")

print("\n[5/5] Running focused test suite...")
result = subprocess.run(  # noqa: S603
    [sys.executable, "-m", "pytest", "tests/", "-q", "-x", "--tb=no"],
    capture_output=True,
    text=True,
    timeout=240,
    check=False,
)
if result.returncode != 0:
    print(result.stdout)
    print(result.stderr)
    raise SystemExit(result.returncode)

last_line = [line for line in result.stdout.strip().split("\n") if line][-1]
print(f"  {last_line}")

results = {
    "dataset": {
        "name": "NASA C-MAPSS FD001",
        "source_url": fd001.source_url,
        "landing_page": fd001.landing_page,
        "archive_sha256": fd001.archive_sha256,
        "train_rows": int(len(fd001.train)),
        "test_rows": int(len(fd001.test)),
    },
    "model": {
        "name": "isolation_forest",
        "features": int(len(feature_cols)),
        "train_seconds": round(float(train_seconds), 4),
        "f1": round(float(metrics["f1"]), 4),
        "precision": round(float(metrics["precision"]), 4),
        "recall": round(float(metrics["recall"]), 4),
        "roc_auc": round(float(metrics["roc_auc"]), 4),
    },
}
Path("results").mkdir(exist_ok=True)
Path("results/verification_results.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)

"""Generate local model artifacts required by the FastAPI service."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from pulsenet.config import cfg  # noqa: E402
from pulsenet.models.isolation_forest import IsolationForestModel  # noqa: E402
from pulsenet.pipeline.feature_registry import FeatureRegistry  # noqa: E402
from pulsenet.pipeline.official_cmapss import load_official_fd001  # noqa: E402

ARTIFACT_MANIFEST_KEY_ENV = "PULSENET_ARTIFACT_MANIFEST_KEY"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_signature_payload(payload: dict[str, object]) -> bytes:
    signed_payload = {
        "schema_version": payload.get("schema_version"),
        "artifacts": payload.get("artifacts"),
    }
    return json.dumps(signed_payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _manifest_key_bytes() -> bytes:
    key = os.environ.get(ARTIFACT_MANIFEST_KEY_ENV, "")
    if not key:
        raise RuntimeError(f"{ARTIFACT_MANIFEST_KEY_ENV} must be set")
    return key.encode("utf-8")


def _sign_manifest(payload: dict[str, object]) -> str:
    return hmac.new(
        _manifest_key_bytes(), _artifact_signature_payload(payload), hashlib.sha256
    ).hexdigest()


def main() -> int:
    fd001 = load_official_fd001("data/official", download=False)
    registry = FeatureRegistry(rolling_window=cfg.data.rolling_window)
    train_df = registry.process_offline(fd001.train.copy())
    registry.process_offline(fd001.test.copy())
    scaler = registry.fit_scaler(train_df)
    feature_cols = registry.feature_cols
    train_df[feature_cols] = train_df[feature_cols].astype(float)
    train_df.loc[:, feature_cols] = scaler.transform(train_df[feature_cols])

    healthy = train_df[train_df["time_in_cycles"] <= cfg.data.healthy_cycle_limit]
    x_train = np.asarray(healthy[feature_cols])

    model = IsolationForestModel(
        n_estimators=cfg.models.isolation_forest.n_estimators,
        contamination=cfg.models.isolation_forest.contamination,
        max_samples=cfg.models.isolation_forest.max_samples,
        random_state=cfg.models.isolation_forest.random_state,
    )
    start = time.perf_counter()
    model.train(x_train)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "isolation_forest.joblib"
    scaler_path = models_dir / "scaler.joblib"
    registry_path = models_dir / "feature_registry.joblib"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(registry.save_config(), registry_path)

    manifest_path = models_dir / "api_artifacts.sha256.json"
    artifacts = {
        path.as_posix(): {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in (model_path, scaler_path, registry_path)
    }
    payload: dict[str, object] = {"schema_version": 1, "artifacts": artifacts}
    payload["signature"] = {
        "algorithm": "hmac-sha256",
        "value": _sign_manifest(payload),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "feature_registry_path": str(registry_path),
        "artifact_manifest_path": str(manifest_path),
        "manifest_signature_algorithm": "hmac-sha256",
        "features": len(feature_cols),
        "training_rows": int(len(x_train)),
        "train_seconds": round(time.perf_counter() - start, 3),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Config loader — reads config.yaml with environment variable overrides.

Usage:
    from pulsenet.config import cfg
    print(cfg.models.isolation_forest.n_estimators)
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # PulseNet-main/
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to SimpleNamespace for attribute access."""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _dict_to_namespace(v)
        elif isinstance(v, list):
            d[k] = [_dict_to_namespace(i) if isinstance(i, dict) else i for i in v]
    return SimpleNamespace(**d)


def _apply_env_overrides(d: dict, prefix: str = "PULSENET") -> dict:
    """Override config values from environment variables.

    Convention: PULSENET_SECTION_KEY  (upper-case, underscored).
    Only leaf values are overridden.
    """
    for key, value in d.items():
        env_key = f"{prefix}_{key}".upper()
        if isinstance(value, dict):
            _apply_env_overrides(value, env_key)
        else:
            env_val = os.environ.get(env_key)
            if env_val is not None:
                # Coerce to original type
                if isinstance(value, bool):
                    d[key] = env_val.lower() in ("1", "true", "yes")
                elif isinstance(value, int):
                    d[key] = int(env_val)
                elif isinstance(value, float):
                    d[key] = float(env_val)
                else:
                    d[key] = env_val
    return d


def load_config(path: Path | str | None = None) -> SimpleNamespace:
    """Load and return configuration as a nested namespace."""
    path = Path(path) if path else _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    raw = _apply_env_overrides(raw)
    return _dict_to_namespace(raw)


# Module-level singleton — import anywhere as `from pulsenet.config import cfg`
try:
    cfg = load_config()
except FileNotFoundError:
    cfg = _dict_to_namespace({})  # graceful fallback for tests

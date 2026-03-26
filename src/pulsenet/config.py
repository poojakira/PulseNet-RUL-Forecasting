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

from pydantic import BaseModel, ValidationError


_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # PulseNet-main/
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


class PulseNetConfigSchema(BaseModel):
    """Strict validation schema for config.yaml"""

    models: dict
    api: dict
    streaming: dict


def _dict_to_namespace(d: dict | list) -> SimpleNamespace | list:
    """Recursively convert dict to SimpleNamespace for dot-notation access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(v) for v in d]
    return d


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


def load_config(config_path: str = "config.yaml") -> SimpleNamespace:
    """Load, validate, and parse YAML config, with env var overrides."""

    # 1. Start with hardcoded defaults
    config_dict = {
        "models": {
            "active_model": "isolation_forest",
            "batch_size": 256,
            "threshold": 0.5,
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "rate_limit": 100,
        },
        "streaming": {
            "queue_maxsize": 10000,
            "flush_interval_ms": 100,
            "enable_backpressure": True,
        },
    }

    # 2. Override with YAML if present
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            yaml_content = yaml.safe_load(f) or {}

            # Deep update
            for section, values in yaml_content.items():
                if section in config_dict and isinstance(values, dict):
                    config_dict[section].update(values)
                else:
                    config_dict[section] = values

    # 3. Validate schema
    try:
        PulseNetConfigSchema(**config_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid config.yaml format: {e}")

    # 4. Override with Environment Variables
    config_dict = _apply_env_overrides(config_dict)

    return _dict_to_namespace(config_dict)


# Module-level singleton — import anywhere as `from pulsenet.config import cfg`
try:
    cfg = load_config()
except FileNotFoundError:
    cfg = _dict_to_namespace({})  # graceful fallback for tests

# pyright: reportGeneralTypeIssues=false
"""
Config loader — reads config.yaml with environment variable overrides.

Usage:
    from pulsenet.config import cfg
    print(cfg.models.active_model)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Load .env if present
load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


class ModelConfig(BaseModel):
    active_model: str = "isolation_forest"
    batch_size: int = 256
    threshold: float = 0.5


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    rate_limit: int = 100


class StreamingConfig(BaseModel):
    queue_maxsize: int = 10000
    flush_interval_ms: int = 100
    enable_backpressure: bool = True


class DataConfig(BaseModel):
    train_file: str = "train_FD001.txt"
    test_file: str = "test_FD001.txt"
    rul_file: str = "RUL_FD001.txt"
    rolling_window: int = 5
    healthy_cycle_limit: int = 50
    failure_rul_threshold: int = 30


class PulseNetConfigSchema(BaseModel):
    """Strict validation schema for PulseNet configuration."""

    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)


def _apply_env_overrides(d: dict[str, Any], prefix: str = "PULSENET") -> dict[str, Any]:
    """Override config values from environment variables."""
    for key, value in d.items():
        env_key = f"{prefix}_{key}".upper()
        if isinstance(value, dict):
            _apply_env_overrides(value, env_key)
        else:
            env_val = os.environ.get(env_key)
            if env_val is not None:
                if isinstance(value, bool):
                    d[key] = env_val.lower() in ("1", "true", "yes")
                elif isinstance(value, int):
                    d[key] = int(env_val)
                elif isinstance(value, float):
                    d[key] = float(env_val)
                else:
                    d[key] = env_val
    return d


def load_config(config_path: str = "config.yaml") -> PulseNetConfigSchema:
    """Load, validate, and parse YAML config, with env var overrides."""
    # 1. Start with hardcoded defaults
    config_dict: dict[str, Any] = {
        "models": ModelConfig().model_dump(),
        "api": ApiConfig().model_dump(),
        "streaming": StreamingConfig().model_dump(),
    }

    # 2. Override with YAML if present
    path = Path(config_path)
    if path.exists():
        try:
            with open(path) as f:
                yaml_content = yaml.safe_load(f) or {}

                # Deep update
                for section, values in yaml_content.items():
                    if section in config_dict and isinstance(values, dict):
                        config_dict[section].update(values)
                    else:
                        config_dict[section] = values
        except Exception as e:
            print(f"Warning: Failed to load config from {path}: {e}")

    # 3. Override with Environment Variables
    config_dict = _apply_env_overrides(config_dict)

    # 4. Validate schema
    try:
        return PulseNetConfigSchema(**config_dict)
    except ValidationError as e:
        # Fallback to defaults on validation error, but log it
        print(f"Warning: Invalid configuration detected, using defaults. Error: {e}")
        return PulseNetConfigSchema()


# Module-level singleton
cfg: PulseNetConfigSchema = load_config(str(_CONFIG_PATH))

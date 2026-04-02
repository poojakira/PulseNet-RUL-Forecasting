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


class SystemConfig(BaseModel):
    name: str = "PulseNet"
    version: str = "2.1.0"
    log_level: str = "INFO"
    log_format: str = "json"
    data_dir: str = "./data"
    output_dir: str = "./outputs"


class IsolationForestConfig(BaseModel):
    n_estimators: int = 200
    contamination: float = 0.05
    max_samples: float = 0.8
    random_state: int = 42


class LstmConfig(BaseModel):
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    sequence_length: int = 30
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 64


class TransformerConfig(BaseModel):
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    sequence_length: int = 30
    learning_rate: float = 0.0001
    epochs: int = 50
    batch_size: int = 64


class ThresholdConfig(BaseModel):
    method: str = "youden"
    percentile_value: int = 95


class ModelConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    active_model: str = "isolation_forest"
    model_dir: str = "./models"
    isolation_forest: IsolationForestConfig = Field(default_factory=IsolationForestConfig)
    lstm: LstmConfig = Field(default_factory=LstmConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)


class SecurityConfig(BaseModel):
    encryption_algorithm: str = "AES-256-Fernet"
    key_env_variable: str = "PULSENET_ENCRYPTION_KEY"
    key_file: str = "secret.key"
    key_rotation_days: int = 30
    jwt_secret_env: str = "PULSENET_JWT_SECRET"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    # simplified representation of roles
    roles: dict[str, list[str]] = Field(default_factory=dict)


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 4
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    rate_limit: int = 100


class StreamingConfig(BaseModel):
    queue_max_size: int = 1000
    batch_size: int = 32
    producer_delay_ms: int = 30
    consumer_timeout_ms: int = 5000
    backpressure_threshold: float = 0.8


class DataConfig(BaseModel):
    train_file: str = "train_FD001.txt"
    test_file: str = "test_FD001.txt"
    rul_file: str = "RUL_FD001.txt"
    drop_sensors: list[str] = Field(default_factory=list)
    drop_settings: list[str] = Field(default_factory=list)
    rolling_window: int = 5
    healthy_cycle_limit: int = 50
    failure_rul_threshold: int = 30


class BlockchainConfig(BaseModel):
    chain_file: str = "blackbox_ledger.json"
    hash_algorithm: str = "sha256"
    enable_merkle: bool = True


class MlopsConfig(BaseModel):
    experiment_name: str = "PulseNet_PredictiveMaintenance"
    tracking_uri: str = "mlruns"
    drift_threshold: float = 0.1
    auto_retrain: bool = True
    monitor_interval_sec: int = 3600


class BenchmarkConfig(BaseModel):
    inference_target_ms: int = 50
    packet_loss_rates: list[float] = Field(default_factory=lambda: [0.10, 0.20, 0.30])
    network_trials: int = 1000
    warmup_iterations: int = 10
    benchmark_iterations: int = 100


class DashboardConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8501
    refresh_interval_sec: int = 5


class PulseNetConfigSchema(BaseModel):
    """Strict validation schema for PulseNet configuration."""
    model_config = {"protected_namespaces": ()}

    system: SystemConfig = Field(default_factory=SystemConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    blockchain: BlockchainConfig = Field(default_factory=BlockchainConfig)
    mlops: MlopsConfig = Field(default_factory=MlopsConfig)
    benchmarks: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


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

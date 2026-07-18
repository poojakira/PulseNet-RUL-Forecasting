# PulseNet Configuration with Production Validation
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Fail fast in production if required env vars not set
if os.environ.get("PULSENET_ENV") == "production":
    required_vars = [
        "PULSENET_JWT_SECRET",
        "PULSENET_ENCRYPTION_KEY",
        "PULSENET_ADMIN_PASSWORD_HASH",
    ]
    for var in required_vars:
        if not os.environ.get(var):
            raise RuntimeError(
                f"Required environment variable {var} is not set. "
                f"See .env.example for required variables."
            )
    
    # Validate CORS
    cors = os.environ.get("CORS_ORIGINS", "")
    if "*" in cors:
        raise RuntimeError(
            "Wildcard CORS '*' not allowed in production. "
            "Set CORS_ORIGINS to specific domains."
        )

@dataclass
class DataConfig:
    """Data pipeline configuration."""
    raw_data_dir: Path = field(default_factory=lambda: Path("./data/raw"))
    processed_data_dir: Path = field(default_factory=lambda: Path("./data/processed"))
    rolling_window: int = 30
    contamination: float = 0.05
    n_sensors: int = 21

@dataclass
class ModelConfig:
    """Model training configuration."""
    n_estimators: int = 200
    max_samples: float = 0.8
    max_features: float = 1.0
    bootstrap: bool = False
    n_jobs: int = -1
    random_state: int = 42

@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=list)
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

@dataclass
class SecurityConfig:
    """Security configuration."""
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiry_min: int = 60
    encryption_key: str = ""
    admin_password_hash: str = ""
    users_json: str = ""

@dataclass
class SystemConfig:
    """System configuration."""
    debug: bool = False
    log_level: str = "INFO"
    ddp_backend: str = "gloo"
    num_gpus: int = 0

@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        cfg = cls()

        # API config
        cfg.api.host = os.environ.get("HOST", "0.0.0.0")
        cfg.api.port = int(os.environ.get("PORT", "8000"))
        cors_env = os.environ.get("CORS_ORIGINS", "")
        if cors_env:
            cfg.api.cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
        cfg.api.rate_limit_requests = int(os.environ.get("RATE_LIMIT_REQUESTS", "100"))
        cfg.api.rate_limit_window = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

        # Security config
        cfg.security.jwt_secret = os.environ.get("PULSENET_JWT_SECRET", "")
        cfg.security.encryption_key = os.environ.get("PULSENET_ENCRYPTION_KEY", "")
        cfg.security.admin_password_hash = os.environ.get("PULSENET_ADMIN_PASSWORD_HASH", "")
        cfg.security.users_json = os.environ.get("PULSENET_USERS", "")

        # System config
        cfg.system.debug = os.environ.get("PULSENET_ENV") != "production"
        cfg.system.log_level = os.environ.get("PULSENET_LOG_LEVEL", "INFO")
        cfg.system.ddp_backend = os.environ.get("PULSENET_DDP_BACKEND", "gloo")
        cfg.system.num_gpus = int(os.environ.get("NUM_GPUS", "0"))

        # Data config
        cfg.data.rolling_window = int(os.environ.get("DETECTOR_WINDOW_SIZE", "30"))
        cfg.data.contamination = float(os.environ.get("DETECTOR_CONTAMINATION", "0.05"))

        return cfg


# Global config instance
cfg = Config.from_env()
"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    # Application
    app_name: str = Field(default="Network Anomaly Detection", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    secret_key: str = Field(default="dev-secret-key-change-in-production", alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Database
    mongodb_url: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URL")
    database_name: str = Field(default="network_anomaly_db", alias="DATABASE_NAME")

    # Paths
    model_path: Path = Field(default=Path("./models"), alias="MODEL_PATH")
    data_path: Path = Field(default=Path("./data"), alias="DATA_PATH")
    log_dir: Path = Field(default=Path("./logs"), alias="LOG_DIR")

    # API
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        alias="CORS_ORIGINS"
    )

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Detection parameters
    anomaly_threshold: float = Field(default=0.5, alias="ANOMALY_THRESHOLD")
    isolation_contamination: float = Field(default=0.1, alias="ISOLATION_CONTAMINATION")
    batch_size: int = Field(default=64, alias="BATCH_SIZE")

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    model_config = {"env_file": ".env", "populate_by_name": True}


settings = Settings()

"""
Configuration management for OpenReasoning.
"""

import os
import platform
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import Field, validator
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    huggingface_api_key: Optional[str] = Field(None, env="HUGGINGFACE_API_KEY")

    # Database Settings
    mongodb_uri: str = Field("mongodb://localhost:27017", env="MONGODB_URI")
    database_name: str = Field("openreasoning", env="DATABASE_NAME")

    # Application Settings
    log_level: str = Field("INFO", env="LOG_LEVEL")
    environment: str = Field("development", env="ENVIRONMENT")

    # Model Settings
    default_model_provider: str = Field("openai", env="DEFAULT_MODEL_PROVIDER")
    default_model: str = Field("gpt-4o", env="DEFAULT_MODEL")
    default_embedding_model: str = Field(
        "text-embedding-3-large", env="DEFAULT_EMBEDDING_MODEL"
    )

    # Compute Settings
    use_mlx: bool = Field(False, env="USE_MLX")
    mlx_num_cores: int = Field(10, env="MLX_NUM_CORES")
    mlx_memory_limit: str = Field("24GB", env="MLX_MEMORY_LIMIT")

    # System Information
    is_apple_silicon: bool = Field(
        default_factory=lambda: platform.machine() == "arm64"
        and platform.system() == "Darwin"
    )
    is_m3_chip: bool = Field(
        default_factory=lambda: platform.machine() == "arm64"
        and platform.system() == "Darwin"
        and os.popen("sysctl -n machdep.cpu.brand_string").read().strip().find("M3")
        >= 0
    )

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = ["development", "testing", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()

    @validator("default_model_provider")
    def validate_model_provider(cls, v):
        """Validate model provider."""
        valid_providers = ["openai", "anthropic", "huggingface", "local"]
        if v.lower() not in valid_providers:
            raise ValueError(f"Model provider must be one of {valid_providers}")
        return v.lower()

    class Config:
        """Config class for Settings."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create settings instance
settings = Settings()

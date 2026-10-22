from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from app import __version__

SCHEMA_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Prefer the local .env over ambient variables.

        Developer shells often export DATABASE_URL for another project. An
        explicit .env in this repository should win; containers have no .env
        file and still receive everything through environment variables.
        """
        return (init_settings, dotenv_settings, env_settings, file_secret_settings)

    app_name: str = "chat-memory"
    app_version: str = __version__
    log_level: str = "INFO"

    database_url: str
    db_schema: str = "rag_demo"
    db_pool_min: int = Field(default=1, ge=1, le=20)
    db_pool_max: int = Field(default=10, ge=1, le=50)
    hnsw_ef_search: int = Field(default=100, ge=1, le=1000)

    source_database_url: str | None = None

    ocr_cache_dir: str = "data/cache/ocr"
    ocr_provider: Literal["tesseract", "openrouter"] = "tesseract"
    ocr_model: str = "google/gemini-2.5-flash"
    embedding_model: str = "minishlab/potion-multilingual-128M"
    embedding_cache_dir: str = "data/cache/fastembed"
    embedding_threads: int = Field(default=3, ge=1, le=64)
    embedding_dim: int = Field(default=256, ge=1)

    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str = "https://github.com/Nezhirn/chat-memory"
    openrouter_app_name: str = "chat-memory"

    openrouter_api_key: str | None = None
    agent_model: str = "google/gemini-2.5-flash"
    judge_model: str = "openai/gpt-4.1-mini"

    telegram_bot_token: str | None = None
    telegram_allowlist: str = ""
    telegram_proxy: str | None = None
    public_base_url: str | None = None
    admin_token: str | None = None

    @field_validator("db_schema")
    @classmethod
    def validate_schema(cls, value: str) -> str:
        if not SCHEMA_PATTERN.match(value):
            raise ValueError("DB_SCHEMA must match ^[a-z_][a-z0-9_]{0,62}$")
        return value

    @model_validator(mode="after")
    def validate_pool(self) -> Settings:
        if self.db_pool_min > self.db_pool_max:
            raise ValueError("DB_POOL_MIN must not exceed DB_POOL_MAX")
        return self

    @property
    def telegram_allowlist_ids(self) -> set[int]:
        raw = re.split(r"[,\s]+", self.telegram_allowlist.strip())
        return {int(item) for item in raw if item}


@lru_cache
def get_settings() -> Settings:
    return Settings()

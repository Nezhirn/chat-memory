from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TableCounts(BaseModel):
    sources: int = 0
    items: int = 0
    chunks: int = 0


class DatabaseHealth(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: Literal["ok", "error"]
    database: str | None = None
    user: str | None = None
    pg_version: str | None = None
    pgvector: str | None = None
    schema_name: str | None = Field(default=None, alias="schema")
    schema_exists: bool = False
    tables: TableCounts = Field(default_factory=TableCounts)
    migrations: int = 0
    latency_ms: float = 0.0
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    service: str
    version: str
    db: DatabaseHealth

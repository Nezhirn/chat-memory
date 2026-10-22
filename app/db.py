from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from psycopg import AsyncConnection, sql
from psycopg_pool import AsyncConnectionPool

from app.config import Settings

logger = logging.getLogger(__name__)


class Database:
    """Async PostgreSQL access built around a psycopg connection pool."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.schema = settings.db_schema
        self.pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            min_size=settings.db_pool_min,
            max_size=settings.db_pool_max,
            open=False,
            kwargs={"autocommit": True, "application_name": settings.app_name},
            configure=self._configure,
        )

    async def _configure(self, conn: AsyncConnection) -> None:
        await conn.execute(
            "select set_config('hnsw.ef_search', %s, false)",
            (str(self.settings.hnsw_ef_search),),
        )

    async def open(self) -> None:
        await self.pool.open(wait=True, timeout=10)

    async def close(self) -> None:
        await self.pool.close()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncConnection]:
        async with self.pool.connection() as conn:
            yield conn

    async def ensure_ready(self) -> None:
        """Fail fast when the database is missing pgvector or the schema."""
        async with self.connection() as conn:
            version = await self._fetchone(
                conn, "select extversion from pg_extension where extname = 'vector'"
            )
            if version is None:
                raise RuntimeError(
                    "pgvector extension is not installed; run CREATE EXTENSION vector; "
                    "as a superuser in the chat_memory database"
                )

            exists = await self._fetchone(
                conn,
                "select exists (select 1 from pg_namespace where nspname = %s)",
                (self.schema,),
            )
            if not exists or not exists[0]:
                raise RuntimeError(f"schema {self.schema!r} is missing; run python -m app.migrate")

    async def health(self) -> dict[str, Any]:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "status": "ok",
            "schema": self.schema,
            "schema_exists": False,
            "tables": {"sources": 0, "items": 0, "chunks": 0},
            "migrations": 0,
        }
        try:
            async with self.connection() as conn:
                row = await self._fetchone(
                    conn,
                    "select current_database(), current_user, split_part(version(), ' ', 2)",
                )
                payload.update(database=row[0], user=row[1], pg_version=row[2])

                ext = await self._fetchone(
                    conn, "select extversion from pg_extension where extname = 'vector'"
                )
                payload["pgvector"] = ext[0] if ext else None

                exists = await self._fetchone(
                    conn,
                    "select exists (select 1 from pg_namespace where nspname = %s)",
                    (self.schema,),
                )
                payload["schema_exists"] = bool(exists and exists[0])

                if payload["schema_exists"]:
                    payload["tables"] = await self._table_counts(conn)
                    payload["migrations"] = await self._migration_count(conn)

                if not payload["pgvector"]:
                    payload["status"] = "error"
                    payload["error"] = "pgvector extension is missing"
                elif not payload["schema_exists"]:
                    payload["status"] = "error"
                    payload["error"] = f"schema {self.schema!r} is missing; run migrations"
        except Exception as exc:
            logger.exception("database health check failed")
            payload["status"] = "error"
            payload["error"] = f"{type(exc).__name__}: {exc}"

        payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return payload

    async def _table_counts(self, conn: AsyncConnection) -> dict[str, int]:
        counts = {"sources": 0, "items": 0, "chunks": 0}
        for table in counts:
            reg = await self._fetchone(conn, "select to_regclass(%s)", (f"{self.schema}.{table}",))
            if reg is None or reg[0] is None:
                continue
            row = await self._fetchone(
                conn,
                sql.SQL("select count(*) from {}").format(sql.Identifier(self.schema, table)),
            )
            if row is not None:
                counts[table] = int(row[0])
        return counts

    async def _migration_count(self, conn: AsyncConnection) -> int:
        reg = await self._fetchone(
            conn, "select to_regclass(%s)", (f"{self.schema}.schema_migrations",)
        )
        if reg is None or reg[0] is None:
            return 0
        row = await self._fetchone(
            conn,
            sql.SQL("select count(*) from {}").format(
                sql.Identifier(self.schema, "schema_migrations")
            ),
        )
        return int(row[0]) if row is not None else 0

    @staticmethod
    async def _fetchone(
        conn: AsyncConnection,
        query: str | sql.Composable,
        params: tuple[Any, ...] | None = None,
    ) -> tuple[Any, ...] | None:
        result = await conn.execute(query, params)
        return await result.fetchone()

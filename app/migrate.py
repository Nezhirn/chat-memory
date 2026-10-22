from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg
from psycopg import sql

from app.config import SCHEMA_PATTERN, get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "db" / "migrations"
SCHEMA_TOKEN = "{{schema}}"
RESIZE_MIGRATION = "002_embedding_dim.sql"
LOCK_KEY = "chat-memory:migrations"


def quote_schema(schema: str) -> str:
    if not SCHEMA_PATTERN.match(schema):
        raise ValueError(f"invalid schema name: {schema!r}")
    return f'"{schema}"'


def render_migration(text: str, schema: str, embedding_dim: int = 1024) -> str:
    """Substitute the configured schema and vector dimension placeholders."""
    return text.replace(SCHEMA_TOKEN, quote_schema(schema)).replace(
        "{{embedding_dim}}", str(int(embedding_dim))
    )


def migration_files(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    return sorted(directory.glob("*.sql"))


def apply_migrations(
    conn: psycopg.Connection,
    schema: str,
    directory: Path = MIGRATIONS_DIR,
    *,
    embedding_dim: int = 1024,
) -> list[str]:
    """Apply pending migrations and return their filenames.

    DDL and the bookkeeping row for each file are applied in the same
    transaction, so a failed migration leaves the schema unchanged.
    """
    files = migration_files(directory)
    applied_now: list[str] = []
    migrations_table = sql.Identifier(schema, "schema_migrations")

    with conn.transaction():
        conn.execute("select pg_advisory_xact_lock(hashtext(%s))", (LOCK_KEY,))
        conn.execute(sql.SQL("create schema if not exists {}").format(sql.Identifier(schema)))
        conn.execute(
            sql.SQL(
                "create table if not exists {} ("
                "version text primary key, "
                "applied_at timestamptz not null default now()"
                ")"
            ).format(migrations_table)
        )
        applied = {
            row[0]
            for row in conn.execute(sql.SQL("select version from {}").format(migrations_table))
        }
        for path in files:
            if path.name in applied:
                continue
            conn.execute(render_migration(path.read_text(encoding="utf-8"), schema, embedding_dim))
            conn.execute(
                sql.SQL("insert into {} (version) values (%s)").format(migrations_table),
                (path.name,),
            )
            applied_now.append(path.name)

    return applied_now


def current_embedding_dim(conn: psycopg.Connection, schema: str) -> int | None:
    """Return the dimension of chunks.embedding, or None if the table is missing."""
    row = conn.execute(
        "select atttypmod from pg_attribute "
        "where attrelid = to_regclass(%s) and attname = 'embedding' and not attisdropped",
        (f"{quote_schema(schema)}.chunks",),
    ).fetchone()
    # pgvector stores the declared dimension as the type modifier; -1 means unbounded.
    return int(row[0]) if row is not None and row[0] > 0 else None


def resize_embeddings(
    conn: psycopg.Connection,
    schema: str,
    embedding_dim: int,
    directory: Path = MIGRATIONS_DIR,
) -> None:
    """Change chunks.embedding to the given dimension.

    Reuses the resize migration, so every stored vector is set to NULL and the
    HNSW index is rebuilt; run the embed command afterwards.
    """
    text = (directory / RESIZE_MIGRATION).read_text(encoding="utf-8")
    with conn.transaction():
        conn.execute("select pg_advisory_xact_lock(hashtext(%s))", (LOCK_KEY,))
        conn.execute(render_migration(text, schema, embedding_dim))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.migrate",
        description="Apply pending SQL migrations and check the embedding dimension.",
    )
    parser.add_argument(
        "--resize-embeddings",
        action="store_true",
        help="change chunks.embedding to EMBEDDING_DIM; drops all stored vectors",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    target_dim = settings.embedding_dim
    with psycopg.connect(settings.database_url) as conn:
        extension = conn.execute(
            "select extversion from pg_extension where extname = 'vector'"
        ).fetchone()
        if extension is None:
            print(
                "pgvector extension is missing; run as a superuser: CREATE EXTENSION vector;",
                file=sys.stderr,
            )
            return 1
        applied = apply_migrations(conn, settings.db_schema, embedding_dim=target_dim)
        if applied:
            print("applied:", ", ".join(applied))
        else:
            print("schema is up to date")

        current_dim = current_embedding_dim(conn, settings.db_schema)
        if current_dim is not None and current_dim != target_dim:
            if not args.resize_embeddings:
                print(
                    f"chunks.embedding is vector({current_dim}) but EMBEDDING_DIM={target_dim}; "
                    "rerun with --resize-embeddings (drops stored vectors), "
                    "then run: python -m app.ingest embed",
                    file=sys.stderr,
                )
                return 1
            resize_embeddings(conn, settings.db_schema, target_dim)
            print(
                f"resized chunks.embedding: vector({current_dim}) -> vector({target_dim}); "
                "run: python -m app.ingest embed"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

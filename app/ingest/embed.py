from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import psycopg
from psycopg import sql

from app.config import get_settings
from app.retrieval.embeddings import Embedder, vector_literal

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 256


def embed_chunks(
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    rebuild: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    settings = get_settings()
    where = "" if rebuild else "where embedding is null"
    limit_sql = "" if limit is None else "limit %s"
    params: tuple = () if limit is None else (limit,)
    query = sql.SQL(f"select id, body from {{chunks}} {where} order by id {limit_sql}").format(
        chunks=sql.Identifier(settings.db_schema, "chunks")
    )

    with psycopg.connect(settings.database_url) as conn:
        rows = conn.execute(query, params).fetchall()
        # Close the implicit read transaction so each batch below commits on its
        # own instead of becoming a savepoint inside one long transaction.
        conn.commit()
        if dry_run:
            print(f"chunks to embed: {len(rows)}")
            return {"chunks": len(rows)}
        if not rows:
            print("all chunks are already embedded")
            return {"chunks": 0}

        embedder = Embedder(settings)
        logger.info(
            "embedding %s chunks with %s (%s-d)",
            len(rows),
            settings.embedding_model,
            embedder.dim,
        )

        started = time.perf_counter()
        update = sql.SQL("update {chunks} set embedding = %s::vector where id = %s").format(
            chunks=sql.Identifier(settings.db_schema, "chunks")
        )
        done = 0
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            vectors = embedder.embed_passages(
                [row[1] for row in batch],
                batch_size=batch_size,
            )
            with conn.transaction(), conn.cursor() as cursor:
                cursor.executemany(
                    update,
                    [
                        (vector_literal(vector), row[0])
                        for vector, row in zip(vectors, batch, strict=True)
                    ],
                )
            done += len(batch)
            logger.info("embedded %s/%s chunks", done, len(rows))

        elapsed = time.perf_counter() - started
        stats: dict[str, Any] = {
            "chunks": done,
            "model": settings.embedding_model,
            "dim": embedder.dim,
            "elapsed_s": round(elapsed, 2),
            "chunks_per_s": round(done / elapsed, 2) if elapsed else 0,
        }
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return {"chunks": done}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest embed",
        description="Embed chunks that have no vector yet (or rebuild all).",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="max chunks in this run")
    parser.add_argument("--rebuild", action="store_true", help="re-embed every chunk")
    parser.add_argument("--dry-run", action="store_true", help="count without calling the model")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    embed_chunks(
        batch_size=args.batch_size,
        limit=args.limit,
        rebuild=args.rebuild,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

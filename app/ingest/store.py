from __future__ import annotations

from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from app.ingest.chunker import content_hash, token_count


def upsert_source(
    conn: psycopg.Connection,
    schema: str,
    *,
    kind: str,
    external_id: str,
    title: str,
    url: str | None,
    meta: dict[str, Any] | None = None,
) -> tuple[int, bool]:
    query = sql.SQL(
        """
        insert into {sources} (kind, external_id, title, url, meta)
        values (%s, %s, %s, %s, %s)
        on conflict (kind, external_id) do update set
            title = excluded.title,
            url = excluded.url,
            meta = excluded.meta,
            updated_at = now()
        returning id, (xmax = 0) as inserted
        """
    ).format(sources=sql.Identifier(schema, "sources"))
    row = conn.execute(
        query,
        (kind, external_id, title, url, Jsonb(meta or {})),
    ).fetchone()
    return int(row[0]), bool(row[1])


def upsert_item(
    conn: psycopg.Connection,
    schema: str,
    *,
    source_id: int,
    external_id: str,
    body: str,
    channel: str | None = None,
    message_id: int | None = None,
    thread_id: int | None = None,
    reply_to_id: int | None = None,
    author: str | None = None,
    published_at: Any = None,
    hashtags: list[str] | None = None,
    views: int | None = None,
    url: str | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[int, bool]:
    body = body.strip()
    query = sql.SQL(
        """
        insert into {items} as i (
            source_id, external_id, channel, message_id, thread_id, reply_to_id,
            author, published_at, body, hashtags, views, url, content_hash, meta
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        on conflict (source_id, external_id) do update set
            channel = excluded.channel,
            message_id = excluded.message_id,
            thread_id = excluded.thread_id,
            reply_to_id = excluded.reply_to_id,
            author = excluded.author,
            published_at = excluded.published_at,
            body = case
                when length(excluded.body) >= length(i.body) then excluded.body
                else i.body
            end,
            hashtags = excluded.hashtags,
            views = excluded.views,
            url = excluded.url,
            content_hash = case
                when length(excluded.body) >= length(i.body) then excluded.content_hash
                else i.content_hash
            end,
            meta = excluded.meta,
            updated_at = now()
        returning id, (xmax = 0) as inserted
        """
    ).format(items=sql.Identifier(schema, "items"))
    row = conn.execute(
        query,
        (
            source_id,
            external_id,
            channel,
            message_id,
            thread_id,
            reply_to_id,
            author,
            published_at,
            body,
            hashtags or [],
            views,
            url,
            content_hash(body),
            Jsonb(meta or {}),
        ),
    ).fetchone()
    return int(row[0]), bool(row[1])


def replace_chunks(
    conn: psycopg.Connection,
    schema: str,
    item_id: int,
    chunks: list[tuple[str | None, str]],
    meta: dict[str, Any] | None = None,
) -> int:
    """Replace all chunks of an item; stale embeddings are invalidated."""
    query = sql.SQL(
        """
        insert into {chunks} as c (item_id, ord, heading, body, token_count, content_hash, meta)
        values (%s, %s, %s, %s, %s, %s, %s)
        on conflict (item_id, ord) do update set
            heading = excluded.heading,
            body = excluded.body,
            token_count = excluded.token_count,
            content_hash = excluded.content_hash,
            meta = excluded.meta,
            embedding = case
                when c.content_hash is distinct from excluded.content_hash then null
                else c.embedding
            end
        """
    ).format(chunks=sql.Identifier(schema, "chunks"))
    # Drop empty chunks first so `ord` stays contiguous and the tail delete is exact.
    chunks = [(heading, body.strip()) for heading, body in chunks if body.strip()]
    for ord_, (heading, body) in enumerate(chunks):
        conn.execute(
            query,
            (
                item_id,
                ord_,
                heading,
                body,
                token_count(body),
                content_hash(body),
                Jsonb(meta or {}),
            ),
        )
    conn.execute(
        sql.SQL("delete from {chunks} where item_id = %s and ord >= %s").format(
            chunks=sql.Identifier(schema, "chunks")
        ),
        (item_id, len(chunks)),
    )
    return len(chunks)


def start_run(conn: psycopg.Connection, schema: str, source: str) -> int:
    row = conn.execute(
        sql.SQL(
            "insert into {runs} (source, status, stats) values (%s, 'running', %s) returning id"
        ).format(runs=sql.Identifier(schema, "ingest_runs")),
        (source, Jsonb({})),
    ).fetchone()
    return int(row[0])


def finish_run(
    conn: psycopg.Connection,
    schema: str,
    run_id: int,
    *,
    status: str,
    stats: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        sql.SQL(
            "update {runs} set status = %s, stats = %s, error = %s, finished_at = now() "
            "where id = %s"
        ).format(runs=sql.Identifier(schema, "ingest_runs")),
        (status, Jsonb(stats or {}), error, run_id),
    )

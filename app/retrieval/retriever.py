from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import psycopg
from psycopg import sql

from app.db import Database


@dataclass(frozen=True)
class HybridHit:
    chunk_id: int
    item_id: int
    channel: str | None
    message_id: int | None
    published_at: datetime | None
    url: str | None
    source_kind: str
    source_title: str
    body: str
    score: float


def rrf_scores(
    fts_ids: list[int],
    dense_ids: list[int],
    *,
    k: int = 60,
) -> dict[int, float]:
    """Reciprocal Rank Fusion, used by tests as the reference formula."""
    scores: dict[int, float] = {}
    for rank, chunk_id in enumerate(fts_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for rank, chunk_id in enumerate(dense_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def _filter_clause(
    *,
    channel: str | None = None,
    source_kind: str | None = None,
    date_from: date | datetime | None = None,
    date_to: date | datetime | None = None,
    hashtags: list[str] | None = None,
) -> tuple[sql.Composable, dict[str, Any]]:
    parts: list[sql.Composable] = []
    params: dict[str, Any] = {}
    if channel:
        parts.append(sql.SQL("i.channel = %(channel)s"))
        params["channel"] = channel
    if source_kind:
        parts.append(sql.SQL("s.kind = %(source_kind)s"))
        params["source_kind"] = source_kind
    if date_from:
        parts.append(sql.SQL("i.published_at >= %(date_from)s"))
        params["date_from"] = date_from
    if date_to:
        parts.append(sql.SQL("i.published_at < (%(date_to)s::date + interval '1 day')"))
        params["date_to"] = date_to
    if hashtags:
        parts.append(sql.SQL("i.hashtags && %(hashtags)s::text[]"))
        params["hashtags"] = hashtags

    if not parts:
        return sql.SQL(""), params
    return sql.SQL(" and ") + sql.SQL(" and ").join(parts), params


def build_hybrid_query(
    schema: str,
    *,
    channel: str | None = None,
    source_kind: str | None = None,
    date_from: date | datetime | None = None,
    date_to: date | datetime | None = None,
    hashtags: list[str] | None = None,
) -> tuple[sql.Composed, dict[str, Any]]:
    filters, params = _filter_clause(
        channel=channel,
        source_kind=source_kind,
        date_from=date_from,
        date_to=date_to,
        hashtags=hashtags,
    )
    query = sql.SQL(
        """
        with q as (
            select websearch_to_tsquery('russian', %(query)s) as query
        ),
        fts as (
            select
                c.id as chunk_id,
                row_number() over (order by ts_rank_cd(c.tsv, q.query) desc) as rank
            from {chunks} c
            join {items} i on i.id = c.item_id
            join {sources} s on s.id = i.source_id
            cross join q
            where c.tsv @@ q.query {filters}
            order by ts_rank_cd(c.tsv, q.query) desc
            limit %(candidates)s
        ),
        dense as (
            select
                c.id as chunk_id,
                row_number() over (order by c.embedding <=> %(vector)s::vector) as rank
            from {chunks} c
            join {items} i on i.id = c.item_id
            join {sources} s on s.id = i.source_id
            where c.embedding is not null {filters}
            order by c.embedding <=> %(vector)s::vector
            limit %(candidates)s
        ),
        fused as (
            select
                coalesce(f.chunk_id, d.chunk_id) as chunk_id,
                coalesce(1.0 / (%(rrf_k)s::numeric + f.rank), 0)
                + coalesce(1.0 / (%(rrf_k)s::numeric + d.rank), 0) as score
            from fts f
            full outer join dense d on f.chunk_id = d.chunk_id
        )
        select
            c.id,
            c.item_id,
            i.channel,
            i.message_id,
            i.published_at,
            i.url,
            s.kind,
            s.title,
            c.body,
            fused.score
        from fused
        join {chunks} c on c.id = fused.chunk_id
        join {items} i on i.id = c.item_id
        join {sources} s on s.id = i.source_id
        order by fused.score desc, i.published_at desc nulls last
        limit %(limit)s
        """
    ).format(
        chunks=sql.Identifier(schema, "chunks"),
        items=sql.Identifier(schema, "items"),
        sources=sql.Identifier(schema, "sources"),
        filters=filters,
    )
    return query, params


def search_hybrid(
    conn: psycopg.Connection,
    schema: str,
    query: str,
    query_vector: str,
    *,
    limit: int = 10,
    candidates: int = 50,
    rrf_k: int = 60,
    channel: str | None = None,
    source_kind: str | None = None,
    date_from: date | datetime | None = None,
    date_to: date | datetime | None = None,
    hashtags: list[str] | None = None,
) -> list[HybridHit]:
    statement, filter_params = build_hybrid_query(
        schema,
        channel=channel,
        source_kind=source_kind,
        date_from=date_from,
        date_to=date_to,
        hashtags=hashtags,
    )
    params = {
        **filter_params,
        "query": query,
        "vector": query_vector,
        "candidates": candidates,
        "rrf_k": rrf_k,
        "limit": limit,
    }
    rows = conn.execute(statement, params).fetchall()
    return [HybridHit(*row[:-1], float(row[-1])) for row in rows]


async def search_hybrid_async(
    db: Database,
    query: str,
    query_vector: str,
    *,
    limit: int = 10,
    candidates: int = 50,
    rrf_k: int = 60,
    channel: str | None = None,
    source_kind: str | None = None,
    date_from: date | datetime | None = None,
    date_to: date | datetime | None = None,
    hashtags: list[str] | None = None,
) -> list[HybridHit]:
    statement, filter_params = build_hybrid_query(
        db.schema,
        channel=channel,
        source_kind=source_kind,
        date_from=date_from,
        date_to=date_to,
        hashtags=hashtags,
    )
    params = {
        **filter_params,
        "query": query,
        "vector": query_vector,
        "candidates": candidates,
        "rrf_k": rrf_k,
        "limit": limit,
    }
    async with db.connection() as conn:
        result = await conn.execute(statement, params)
        rows = await result.fetchall()
    return [HybridHit(*row[:-1], float(row[-1])) for row in rows]

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import psycopg
from psycopg import sql

from app.config import get_settings
from app.db import Database


@dataclass(frozen=True)
class Hit:
    channel: str | None
    message_id: int | None
    published_at: datetime | None
    url: str | None
    rank: float
    headline: str
    source_title: str | None = None

    @property
    def label(self) -> str:
        """Telegram posts are cited as channel/message_id, documents by title."""
        if self.channel:
            return f"{self.channel}/{self.message_id}"
        return self.source_title or "—"


def build_search_query(schema: str) -> sql.Composed:
    return sql.SQL(
        """
        with q as (
            select websearch_to_tsquery('russian', %(query)s) as query
        )
        select
            i.channel,
            i.message_id,
            i.published_at,
            i.url,
            ts_rank_cd(c.tsv, q.query) as rank,
            ts_headline(
                'russian',
                c.body,
                q.query,
                'MaxWords=32, MinWords=8, StartSel=[, StopSel=]'
            ) as headline,
            s.title as source_title
        from {chunks} c
        join {items} i on i.id = c.item_id
        join {sources} s on s.id = i.source_id
        cross join q
        where c.tsv @@ q.query
          and (%(channel)s::text is null or i.channel = %(channel)s)
        order by rank desc, i.published_at desc nulls last
        limit %(limit)s
        """
    ).format(
        chunks=sql.Identifier(schema, "chunks"),
        items=sql.Identifier(schema, "items"),
        sources=sql.Identifier(schema, "sources"),
    )


def search_chunks(
    conn: psycopg.Connection,
    schema: str,
    query: str,
    limit: int = 10,
    channel: str | None = None,
) -> list[Hit]:
    rows = conn.execute(
        build_search_query(schema),
        {"query": query, "channel": channel, "limit": limit},
    ).fetchall()
    return [Hit(*row) for row in rows]


async def search_chunks_async(
    db: Database,
    query: str,
    limit: int = 10,
    channel: str | None = None,
) -> list[Hit]:
    async with db.connection() as conn:
        result = await conn.execute(
            build_search_query(db.schema),
            {"query": query, "channel": channel, "limit": limit},
        )
        rows = await result.fetchall()
    return [Hit(*row) for row in rows]


def format_hit(index: int, hit: Hit) -> str:
    date = hit.published_at.strftime("%Y-%m-%d") if hit.published_at else "—"
    lines = [
        f"{index}. {hit.label}  rank={hit.rank:.6f}  {date}",
        f"   {hit.url or '—'}",
        f"   {hit.headline}",
    ]
    return "\n".join(lines)


def format_hybrid_hit(index: int, hit: Any) -> str:
    date = hit.published_at.strftime("%Y-%m-%d") if hit.published_at else "—"
    label = f"{hit.channel}/{hit.message_id}" if hit.channel else hit.source_title
    snippet = " ".join(hit.body.split())[:320]
    lines = [
        f"{index}. {label}  score={hit.score:.6f}  {date}",
        f"   {hit.url or '—'}",
        f"   {snippet}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.search",
        description="Full-text search over ingested chunks (PostgreSQL FTS).",
    )
    parser.add_argument("query", help="search query; websearch syntax is supported")
    parser.add_argument("--limit", type=int, default=10, help="number of hits")
    parser.add_argument("--mode", choices=("hybrid", "fts"), default="hybrid")
    parser.add_argument("--channel", default=None, help="filter by Telegram channel")
    parser.add_argument(
        "--source-kind",
        choices=("telegram", "pdf", "web"),
        default=None,
        help="filter by source kind",
    )
    parser.add_argument("--date-from", default=None, help="published_at >= YYYY-MM-DD")
    parser.add_argument("--date-to", default=None, help="published_at <= YYYY-MM-DD")
    parser.add_argument("--hashtag", action="append", dest="hashtags", default=None)
    parser.add_argument("--candidates", type=int, default=50, help="candidates per branch")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF k")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)

    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.candidates < 1:
        parser.error("--candidates must be positive")

    def parse_date(value: str | None) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            parser.error(f"invalid date: {value!r}")

    date_from = parse_date(args.date_from)
    date_to = parse_date(args.date_to)
    hashtags = [tag.lower().lstrip("#") for tag in args.hashtags] if args.hashtags else None
    settings = get_settings()
    mode = args.mode
    with psycopg.connect(settings.database_url) as conn:
        if mode == "hybrid":
            from app.retrieval.embeddings import Embedder, vector_literal
            from app.retrieval.retriever import search_hybrid

            embedder = Embedder(settings)
            query_vector = vector_literal(embedder.embed_query(args.query))
            hits = search_hybrid(
                conn,
                settings.db_schema,
                args.query,
                query_vector,
                limit=args.limit,
                candidates=args.candidates,
                rrf_k=args.rrf_k,
                channel=args.channel,
                source_kind=args.source_kind,
                date_from=date_from,
                date_to=date_to,
                hashtags=hashtags,
            )
        else:
            hits = search_chunks(
                conn,
                schema=settings.db_schema,
                query=args.query,
                limit=args.limit,
                channel=args.channel,
            )

    if args.json:
        payload: list[dict[str, Any]] = []
        for hit in hits:
            item = asdict(hit)
            item["published_at"] = hit.published_at.isoformat() if hit.published_at else None
            payload.append(item)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not hits:
        print("Ничего не найдено. Проверь, что корпус загружен и запрос не слишком узкий.")
        return 0

    for index, hit in enumerate(hits, start=1):
        if mode == "hybrid":
            print(format_hybrid_hit(index, hit))
        else:
            print(format_hit(index, hit))
        if index != len(hits):
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

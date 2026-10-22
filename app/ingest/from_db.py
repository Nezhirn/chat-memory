from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from psycopg import sql

from app.config import SCHEMA_PATTERN, Settings, get_settings
from app.ingest import store
from app.ingest.chunker import split_telegram_post

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_SCHEMA = "market_research"
DEFAULT_LIMIT = 3000
ORIGIN = "ai_base.market_research"

VIEWS_PATTERN = re.compile(r"^\s*([\d.,]+)\s*([KkMm])?\s*$")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#(\w{1,64})")


@dataclass(frozen=True)
class Post:
    channel: str
    message_id: int
    posted_at: datetime | None
    text: str
    views_text: str | None
    source: str
    title: str | None
    url: str | None
    channel_kind: str | None
    state: str | None
    members: int | None


def parse_views(value: str | None) -> int | None:
    """Parse Telegram view counters such as '1.39K' or '2,4M'."""
    if not value:
        return None
    match = VIEWS_PATTERN.match(str(value))
    if not match:
        return None
    number = float(match.group(1).replace(",", "."))
    multiplier = {"k": 1_000, "m": 1_000_000}.get((match.group(2) or "").lower(), 1)
    return int(number * multiplier)


def extract_hashtags(text: str) -> list[str]:
    return sorted({tag.lower() for tag in HASHTAG_PATTERN.findall(text)})


def telegram_url(channel: str, message_id: int) -> str:
    return f"https://t.me/{channel}/{message_id}"


def validate_source_schema(schema: str) -> str:
    if not SCHEMA_PATTERN.match(schema):
        raise ValueError(f"invalid source schema: {schema!r}")
    return schema


def load_posts(
    conn: psycopg.Connection,
    source_schema: str,
    limit: int,
    channels: list[str] | None = None,
) -> list[Post]:
    posts_table = sql.Identifier(source_schema, "tg_posts")
    channels_table = sql.Identifier(source_schema, "tg_channels")
    query = sql.SQL(
        "select p.channel, p.message_id, p.posted_at, p.text, p.views_text, p.source, "
        "c.title, c.url, c.channel_kind, c.state, c.members "
        "from {posts} p "
        "left join {channels} c on c.channel = p.channel "
    ).format(posts=posts_table, channels=channels_table)

    # Posts without text (NULL or blank) have nothing to index or cite.
    query += sql.SQL("where p.text is not null and btrim(p.text) <> '' ")
    params: list[Any] = []
    if channels:
        query += sql.SQL("and p.channel = any(%s) ")
        params.append(channels)
    query += sql.SQL("order by p.posted_at desc nulls last, p.message_id desc limit %s")
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return [Post(*row) for row in rows]


def ingest_posts(conn: psycopg.Connection, posts: list[Post], schema: str) -> dict[str, int]:
    stats = {
        "sources_created": 0,
        "sources_updated": 0,
        "items_inserted": 0,
        "items_updated": 0,
        "chunks_written": 0,
    }
    source_ids: dict[str, int] = {}

    for index, post in enumerate(posts, start=1):
        source_id = source_ids.get(post.channel)
        if source_id is None:
            source_id, inserted = store.upsert_source(
                conn,
                schema,
                kind="telegram",
                external_id=post.channel,
                title=post.title or post.channel,
                url=post.url or f"https://t.me/{post.channel}",
                meta={
                    "origin": ORIGIN,
                    "channel_kind": post.channel_kind,
                    "state": post.state,
                    "members": post.members,
                },
            )
            source_ids[post.channel] = source_id
            stats["sources_created" if inserted else "sources_updated"] += 1

        body = post.text.strip()
        item_id, inserted = store.upsert_item(
            conn,
            schema,
            source_id=source_id,
            external_id=str(post.message_id),
            body=body,
            channel=post.channel,
            message_id=post.message_id,
            published_at=post.posted_at,
            hashtags=extract_hashtags(body),
            views=parse_views(post.views_text),
            url=telegram_url(post.channel, post.message_id),
            meta={
                "origin": ORIGIN,
                "source": post.source,
                "views_text": post.views_text,
            },
        )
        stats["items_inserted" if inserted else "items_updated"] += 1
        chunks = [(None, chunk) for chunk in split_telegram_post(body)]
        stats["chunks_written"] += store.replace_chunks(
            conn,
            schema,
            item_id,
            chunks,
            {"message_id": post.message_id},
        )

        if index % 500 == 0:
            logger.info("ingested %s/%s posts", index, len(posts))

    return stats


def run_ingest(
    settings: Settings,
    source_url: str,
    source_schema: str = DEFAULT_SOURCE_SCHEMA,
    limit: int = DEFAULT_LIMIT,
    channels: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    validate_source_schema(source_schema)
    if limit < 1:
        raise ValueError("limit must be positive")

    with psycopg.connect(source_url) as source_conn:
        source_conn.read_only = True
        posts = load_posts(source_conn, source_schema, limit, channels)

    if not posts:
        print("В источнике нет постов по заданным фильтрам.")
        return {}

    unique_channels = len({post.channel for post in posts})
    total_chunks = sum(len(split_telegram_post(post.text)) for post in posts)

    if dry_run:
        print(f"posts: {len(posts)}")
        print(f"channels: {unique_channels}")
        print(f"chunks: {total_chunks}")
        print("sample:")
        for post in posts[:3]:
            print(f"  {telegram_url(post.channel, post.message_id)}")
        return {
            "posts": len(posts),
            "channels": unique_channels,
            "chunks": total_chunks,
        }

    started = time.perf_counter()
    with psycopg.connect(settings.database_url) as conn:
        run_id = store.start_run(conn, settings.db_schema, f"from-db:{source_schema}")
        conn.commit()

        try:
            with conn.transaction():
                stats = ingest_posts(conn, posts, settings.db_schema)
            stats["posts"] = len(posts)
            stats["channels"] = unique_channels
            stats["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
            store.finish_run(conn, settings.db_schema, run_id, status="ok", stats=stats)
            conn.commit()
        except Exception as exc:
            store.finish_run(
                conn,
                settings.db_schema,
                run_id,
                status="failed",
                error=str(exc),
            )
            conn.commit()
            raise

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest from-db",
        description="Import Telegram posts from the ai_base.market_research schema.",
    )
    parser.add_argument("--source-url", default=None, help="read-only source DATABASE_URL")
    parser.add_argument("--schema", default=DEFAULT_SOURCE_SCHEMA, help="source schema")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="max posts")
    parser.add_argument(
        "--channel",
        action="append",
        dest="channels",
        help="only these channels; repeatable",
    )
    parser.add_argument("--dry-run", action="store_true", help="count without writing")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    source_url = args.source_url or settings.source_database_url
    if not source_url:
        parser.error("--source-url or SOURCE_DATABASE_URL is required")

    run_ingest(
        settings=settings,
        source_url=source_url,
        source_schema=args.schema,
        limit=args.limit,
        channels=args.channels,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

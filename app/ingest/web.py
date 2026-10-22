from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import requests
import trafilatura

from app.config import Settings, get_settings
from app.ingest import store
from app.ingest.chunker import split_document_text

logger = logging.getLogger(__name__)

DEFAULT_URLS_FILE = Path("data/web_urls.txt")
DEFAULT_LIMIT = 20
USER_AGENT = "chat-memory-demo/0.1"


@dataclass(frozen=True)
class WebPage:
    url: str
    requested_url: str
    title: str
    text: str
    published_at: datetime | None
    status_code: int


@dataclass(frozen=True)
class FetchError:
    url: str
    error: str


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def parse_web_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def clean_title(value: str | None, fallback: str) -> str:
    title = (value or fallback).replace("¶", "").strip()
    return title or fallback


def fetch_page(url: str, timeout: int = 30) -> WebPage:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    final_url = normalize_url(response.url or url)
    metadata = trafilatura.extract_metadata(response.text, default_url=final_url)
    text = trafilatura.extract(
        response.text,
        url=final_url,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    if not text or len(text.strip()) < 200:
        raise ValueError(f"no extractable text ({len(text or '')} chars)")

    published_at = parse_web_date(getattr(metadata, "date", None))
    title = clean_title(getattr(metadata, "title", None), final_url)
    return WebPage(
        url=final_url,
        requested_url=url,
        title=title,
        text=text.strip(),
        published_at=published_at,
        status_code=response.status_code,
    )


def load_urls(explicit: list[str] | None, urls_file: Path | None, limit: int) -> list[str]:
    if explicit:
        urls = explicit
    else:
        path = urls_file or DEFAULT_URLS_FILE
        if not path.is_file():
            raise FileNotFoundError(f"URL list not found: {path}")
        urls = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    unique: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = normalize_url(url)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique[:limit]


def ingest_web(
    settings: Settings,
    *,
    urls: list[str] | None = None,
    urls_file: Path | None = None,
    limit: int = DEFAULT_LIMIT,
    timeout: int = 30,
    dry_run: bool = False,
) -> dict[str, int | list[str]]:
    if limit < 1:
        raise ValueError("limit must be positive")

    targets = load_urls(urls, urls_file, limit)
    pages: list[WebPage] = []
    errors: list[FetchError] = []
    for url in targets:
        try:
            page = fetch_page(url, timeout=timeout)
            pages.append(page)
            logger.info("fetched %s (%s chars)", page.url, len(page.text))
        except Exception as exc:
            logger.warning("failed to fetch %s: %s", url, exc)
            errors.append(FetchError(url=url, error=f"{type(exc).__name__}: {exc}"))

    chunk_count = sum(len(split_document_text(page.text)) for page in pages)
    if dry_run:
        print(f"urls: {len(targets)}")
        print(f"pages: {len(pages)}")
        print(f"chunks: {chunk_count}")
        print(f"errors: {len(errors)}")
        for page in pages[:5]:
            print(f"  {page.status_code} {len(page.text):6d}  {page.title[:80]}")
        return {
            "urls": len(targets),
            "pages": len(pages),
            "chunks": chunk_count,
        }

    started = time.perf_counter()
    with psycopg.connect(settings.database_url) as conn:
        run_id = store.start_run(conn, settings.db_schema, "web:batch")
        conn.commit()

        stats: dict = {
            "urls": len(targets),
            "pages": 0,
            "sources_created": 0,
            "sources_updated": 0,
            "items_inserted": 0,
            "items_updated": 0,
            "chunks_written": 0,
            "errors": [error.__dict__ for error in errors],
        }
        try:
            with conn.transaction():
                for page in pages:
                    source_id, inserted = store.upsert_source(
                        conn,
                        settings.db_schema,
                        kind="web",
                        external_id=page.url,
                        title=page.title,
                        url=page.url,
                        meta={
                            "origin": "web",
                            "requested_url": page.requested_url,
                            "host": urlsplit(page.url).netloc,
                            "status_code": page.status_code,
                        },
                    )
                    stats["sources_created" if inserted else "sources_updated"] += 1

                    item_id, item_inserted = store.upsert_item(
                        conn,
                        settings.db_schema,
                        source_id=source_id,
                        external_id="page",
                        body=page.text,
                        published_at=page.published_at,
                        url=page.url,
                        meta={
                            "origin": "web",
                            "title": page.title,
                            "requested_url": page.requested_url,
                        },
                    )
                    stats["items_inserted" if item_inserted else "items_updated"] += 1
                    chunks = [
                        (page.title if index == 0 else None, chunk)
                        for index, chunk in enumerate(split_document_text(page.text))
                    ]
                    stats["chunks_written"] += store.replace_chunks(
                        conn,
                        settings.db_schema,
                        item_id,
                        chunks,
                    )
                    stats["pages"] += 1

            stats["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
            if pages:
                store.finish_run(conn, settings.db_schema, run_id, status="ok", stats=stats)
            else:
                store.finish_run(
                    conn,
                    settings.db_schema,
                    run_id,
                    status="failed",
                    stats=stats,
                    error="no pages fetched",
                )
            conn.commit()
        except Exception as exc:
            store.finish_run(
                conn,
                settings.db_schema,
                run_id,
                status="failed",
                stats=stats,
                error=str(exc),
            )
            conn.commit()
            raise

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest web",
        description="Fetch web pages, extract main text with trafilatura, and index them.",
    )
    parser.add_argument("--url", action="append", dest="urls", help="page URL; repeatable")
    parser.add_argument(
        "--urls-file",
        type=Path,
        default=None,
        help=f"file with one URL per line (default: {DEFAULT_URLS_FILE})",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="max pages")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout, seconds")
    parser.add_argument("--dry-run", action="store_true", help="fetch without writing")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ingest_web(
        get_settings(),
        urls=args.urls,
        urls_file=args.urls_file,
        limit=args.limit,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

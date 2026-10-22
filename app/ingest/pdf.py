from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import psycopg
import pymupdf
from psycopg import sql

from app.config import Settings, get_settings
from app.ingest import store
from app.ingest.chunker import split_document_text
from app.ingest.ocr import extract_ocr_text

logger = logging.getLogger(__name__)

DEFAULT_LIMIT_PAGES = 50
MIN_TEXT_CHARS = 100
PDF_DATE_PATTERN = re.compile(
    r"D:(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?(?:([+-])(\d{2})'?(\d{2})?'?)?"
)


@dataclass(frozen=True)
class Page:
    number: int
    text: str
    method: str


def parse_pdf_date(value: str | None) -> datetime | None:
    if not value:
        return None
    match = PDF_DATE_PATTERN.match(value)
    if not match:
        return None
    year, month, day, hour, minute, second = (int(part or 0) for part in match.groups()[:6])
    offset = timedelta(0)
    if match.group(7):
        sign = 1 if match.group(7) == "+" else -1
        offset = sign * timedelta(
            hours=int(match.group(8)),
            minutes=int(match.group(9) or 0),
        )
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone(offset))
    except ValueError:
        return None


def ocr_page(
    page: pymupdf.Page,
    settings: Settings,
    *,
    provider: str | None,
    model: str | None,
) -> str:
    pixmap = page.get_pixmap(dpi=300, colorspace=pymupdf.csRGB)
    return extract_ocr_text(
        pixmap.tobytes("png"),
        settings,
        provider=provider,
        model=model,
    )


def extract_pages(
    path: Path,
    limit_pages: int,
    settings: Settings,
    *,
    ocr_provider: str | None = None,
    ocr_model: str | None = None,
) -> tuple[list[Page], dict]:
    pages: list[Page] = []
    with pymupdf.open(path) as document:
        metadata = document.metadata or {}
        for index in range(min(document.page_count, limit_pages)):
            page = document[index]
            text = page.get_text("text").strip()
            method = "text"
            if len(text) < MIN_TEXT_CHARS:
                logger.info("page %s has little text, running OCR", index + 1)
                text = ocr_page(
                    page,
                    settings,
                    provider=ocr_provider,
                    model=ocr_model,
                )
                method = "ocr"
            pages.append(Page(number=index + 1, text=text, method=method))
    return pages, metadata


def ingest_pdf(
    settings: Settings,
    path: Path,
    *,
    title: str | None = None,
    limit_pages: int = DEFAULT_LIMIT_PAGES,
    ocr_provider: str | None = None,
    ocr_model: str | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if limit_pages < 1:
        raise ValueError("limit_pages must be positive")

    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    pages, metadata = extract_pages(
        path,
        limit_pages,
        settings,
        ocr_provider=ocr_provider,
        ocr_model=ocr_model,
    )
    non_empty = [page for page in pages if page.text]
    chunk_count = sum(len(split_document_text(page.text)) for page in non_empty)
    document_title = title or metadata.get("title") or path.stem
    published_at = parse_pdf_date(metadata.get("creationDate")) or datetime.fromtimestamp(
        path.stat().st_mtime, tz=UTC
    )
    methods = {
        page.method: sum(1 for item in pages if item.method == page.method) for page in pages
    }

    if dry_run:
        provider = ocr_provider or settings.ocr_provider
        model = ocr_model or settings.ocr_model
        ocr_label = provider if provider == "tesseract" else f"{provider} / {model}"
        print(f"pages: {len(pages)} (with text: {len(non_empty)})")
        print(f"chunks: {chunk_count}")
        print(f"methods: {methods}")
        print(f"ocr: {ocr_label}")
        print(f"title: {document_title}")
        return {"pages": len(pages), "chunks": chunk_count}

    started = time.perf_counter()
    with psycopg.connect(settings.database_url) as conn:
        run_id = store.start_run(conn, settings.db_schema, f"pdf:{path.name}")
        conn.commit()

        try:
            with conn.transaction():
                source_id, source_inserted = store.upsert_source(
                    conn,
                    settings.db_schema,
                    kind="pdf",
                    external_id=file_hash,
                    title=document_title,
                    url=None,
                    meta={
                        "origin": "pdf",
                        "file_name": path.name,
                        "file_path": str(path),
                        "file_hash": file_hash,
                        "pages": len(pages),
                        "title": document_title,
                    },
                )
                conn.execute(
                    sql.SQL("update {sources} set url = %s where id = %s").format(
                        sources=sql.Identifier(settings.db_schema, "sources")
                    ),
                    (f"/files/{source_id}", source_id),
                )

                stats = {
                    "source_inserted": int(source_inserted),
                    "pages": len(pages),
                    "pages_with_text": len(non_empty),
                    "items_inserted": 0,
                    "items_updated": 0,
                    "chunks_written": 0,
                    "methods": methods,
                    "ocr_provider": ocr_provider or settings.ocr_provider,
                    "ocr_model": ocr_model or settings.ocr_model,
                }

                for page in non_empty:
                    item_id, inserted = store.upsert_item(
                        conn,
                        settings.db_schema,
                        source_id=source_id,
                        external_id=f"page-{page.number}",
                        body=page.text,
                        published_at=published_at,
                        url=f"/files/{source_id}#page={page.number}",
                        meta={
                            "origin": "pdf",
                            "page": page.number,
                            "method": page.method,
                            "file_name": path.name,
                        },
                    )
                    stats["items_inserted" if inserted else "items_updated"] += 1
                    chunks = [
                        (document_title if index == 0 else None, chunk)
                        for index, chunk in enumerate(split_document_text(page.text))
                    ]
                    stats["chunks_written"] += store.replace_chunks(
                        conn,
                        settings.db_schema,
                        item_id,
                        chunks,
                        {"page": page.number},
                    )

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
        prog="python -m app.ingest pdf",
        description="Ingest a PDF: text layer, OCR fallback, one item per page.",
    )
    parser.add_argument("--path", required=True, help="PDF file")
    parser.add_argument("--title", default=None, help="override document title")
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=DEFAULT_LIMIT_PAGES,
        help="max pages to ingest",
    )
    parser.add_argument(
        "--ocr-provider",
        choices=("tesseract", "openrouter"),
        default=None,
        help="OCR backend; defaults to OCR_PROVIDER",
    )
    parser.add_argument(
        "--ocr-model",
        default=None,
        help="OpenRouter vision model; defaults to OCR_MODEL",
    )
    parser.add_argument("--dry-run", action="store_true", help="extract without writing")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ingest_pdf(
        get_settings(),
        Path(args.path),
        title=args.title,
        limit_pages=args.limit_pages,
        ocr_provider=args.ocr_provider,
        ocr_model=args.ocr_model,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from datetime import UTC, datetime
from pathlib import Path

from app.ingest.chunker import split_document_text
from app.ingest.pdf import parse_pdf_date
from app.ingest.web import clean_title, load_urls, normalize_url, parse_web_date


def test_parse_pdf_date():
    assert parse_pdf_date("D:20261021154600+03'00'") == datetime(
        2026, 10, 21, 12, 46, 0, tzinfo=UTC
    )
    assert parse_pdf_date(None) is None
    assert parse_pdf_date("garbage") is None


def test_parse_web_date():
    assert parse_web_date("2026-10-21") == datetime(2026, 10, 21, tzinfo=UTC)
    assert parse_web_date(None) is None
    assert parse_web_date("not-a-date") is None


def test_normalize_url_drops_fragment():
    assert normalize_url("https://example.com/a?b=1#frag") == "https://example.com/a?b=1"


def test_clean_title_removes_trafilatura_marker():
    assert clean_title("FastAPI¶", "fallback") == "FastAPI"
    assert clean_title(None, "fallback") == "fallback"


def test_load_urls_deduplicates_and_limits(tmp_path: Path):
    path = tmp_path / "urls.txt"
    path.write_text(
        "# comment\nhttps://example.com/a#one\nhttps://example.com/a#two\n\nhttps://example.com/b\n",
        encoding="utf-8",
    )
    urls = load_urls(None, path, limit=10)
    assert urls == ["https://example.com/a", "https://example.com/b"]
    assert load_urls(None, path, limit=1) == ["https://example.com/a"]


def test_split_document_text_respects_size():
    text = "\n\n".join(["Абзац про поиск и эмбеддинги. " * 30] * 5)
    chunks = split_document_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)

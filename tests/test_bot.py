from datetime import UTC, datetime

from app.bot import (
    dedupe_hybrid_hits,
    format_hybrid_hits,
    format_search_hits,
    format_sources,
    format_stats,
    format_status,
    is_allowed,
    split_message,
)
from app.retrieval.retriever import HybridHit
from app.search import Hit


def test_empty_allowlist_allows_everyone():
    assert is_allowed(42, set()) is True


def test_allowlist_blocks_unknown_user():
    assert is_allowed(42, {7, 42}) is True
    assert is_allowed(99, {7, 42}) is False


def test_format_status_contains_counts_and_schema():
    health = {
        "status": "ok",
        "database": "chat_memory",
        "schema": "rag_demo",
        "pgvector": "0.8.6",
        "tables": {"sources": 1, "items": 2921, "chunks": 3100},
        "migrations": 1,
        "latency_ms": 7.63,
    }
    text = format_status(health)
    assert "rag_demo" in text
    assert "2921" in text
    assert "error" not in text


def test_format_status_reports_error():
    text = format_status({"status": "error", "error": "pgvector extension is missing"})
    assert "error" in text
    assert "pgvector extension is missing" in text


def test_format_sources_empty():
    assert "Источников пока нет" in format_sources([])


def test_format_sources_escapes_html():
    rows = [("telegram", "<b>chan</b>", "https://t.me/x?a=1&b=2", 3)]
    text = format_sources(rows)
    assert "&lt;b&gt;chan&lt;/b&gt;" in text
    assert "a=1&amp;b=2" in text
    assert "3 элементов" in text


def test_format_stats_with_empty_range():
    text = format_stats({"tables": {"sources": 0, "items": 0, "chunks": 0}}, None, None)
    assert "Items: 0" in text
    assert "— —" in text


def test_format_search_hits_contains_link_and_date():
    hit = Hit(
        channel="job_python",
        message_id=123,
        published_at=datetime(2026, 10, 21, 15, 10, tzinfo=UTC),
        url="https://t.me/job_python/123",
        rank=0.5,
        headline="Ищем [Python] разработчика",
    )
    text = format_search_hits("python", [hit])
    assert "Найдено 1" in text
    assert "job_python/123" in text
    assert "https://t.me/job_python/123" in text
    assert "2026-10-21" in text
    assert "[Python]" in text


def test_format_search_hits_empty():
    assert "ничего не найдено" in format_search_hits("no-such-topic", [])


def test_format_hybrid_hits_labels_document_sources():
    hit = HybridHit(
        chunk_id=1,
        item_id=2,
        channel=None,
        message_id=None,
        published_at=datetime(2026, 10, 1, tzinfo=UTC),
        url="/files/107#page=3",
        source_kind="pdf",
        source_title="Поиск работы: отчёт",
        body="Текст страницы PDF",
        score=0.02,
    )
    text = format_hybrid_hits("зарплаты", [hit], base_url="https://memory.example.com")
    assert "Поиск работы: отчёт" in text
    assert "https://memory.example.com/files/107#page=3" in text
    assert "Текст страницы PDF" in text

    plain = format_hybrid_hits("зарплаты", [hit])
    assert "<a href" not in plain


def test_format_hybrid_hits_escapes_source_title():
    hit = HybridHit(
        chunk_id=1,
        item_id=2,
        channel=None,
        message_id=None,
        published_at=None,
        url="https://example.com/r-and-d",
        source_kind="web",
        source_title="R&D <beta>",
        body="text",
        score=0.02,
    )
    text = format_hybrid_hits("rd", [hit])
    assert "R&amp;D &lt;beta&gt;" in text
    assert "<beta>" not in text


def test_format_hybrid_hits_empty():
    assert "ничего не найдено" in format_hybrid_hits("no-such-topic", [])


def test_dedupe_hybrid_hits_keeps_one_chunk_per_item():
    base = dict(
        published_at=None,
        url=None,
        source_kind="telegram",
        source_title="job_python",
        body="text",
        score=0.01,
    )
    hits = [
        HybridHit(chunk_id=1, item_id=10, channel="job_python", message_id=1, **base),
        HybridHit(chunk_id=2, item_id=10, channel="job_python", message_id=1, **base),
        HybridHit(chunk_id=3, item_id=11, channel="job_python", message_id=2, **base),
    ]
    deduped = dedupe_hybrid_hits(hits, limit=5)
    assert [hit.item_id for hit in deduped] == [10, 11]


def test_split_message_short():
    assert split_message("hello") == ["hello"]


def test_split_message_long_lines():
    text = "\n".join(["x" * 100] * 100)
    parts = split_message(text, limit=500)
    assert len(parts) > 1
    assert all(len(part) <= 500 for part in parts)


def test_split_message_single_long_line():
    parts = split_message("y" * 1_200, limit=500)
    assert len(parts) == 3
    assert all(len(part) <= 500 for part in parts)


def test_format_search_hits_labels_documents_and_resolves_relative_urls():
    hit = Hit(
        channel=None,
        message_id=None,
        published_at=None,
        url="/files/7#page=2",
        rank=0.1,
        headline="текст",
        source_title="Отчёт & итоги",
    )
    text = format_search_hits("отчёт", [hit], base_url="https://memory.example.com/")
    assert "None" not in text
    assert "Отчёт &amp; итоги" in text
    assert 'href="https://memory.example.com/files/7#page=2"' in text

    plain = format_search_hits("отчёт", [hit])
    assert "<a href" not in plain

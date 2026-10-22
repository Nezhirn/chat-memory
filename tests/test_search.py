from datetime import UTC, datetime

from app.search import Hit, format_hit


def test_format_hit_contains_source_and_rank():
    hit = Hit(
        channel="job_python",
        message_id=123,
        published_at=datetime(2026, 10, 21, 15, 10, tzinfo=UTC),
        url="https://t.me/job_python/123",
        rank=0.0603,
        headline="Ищем [Python] разработчика",
    )
    text = format_hit(1, hit)
    assert "job_python/123" in text
    assert "2026-10-21" in text
    assert "rank=0.060300" in text
    assert "https://t.me/job_python/123" in text
    assert "[Python]" in text


def test_format_hit_without_date():
    hit = Hit("chan", 1, None, None, 0.1, "text")
    assert "—" in format_hit(2, hit)


def test_format_hit_labels_documents_by_title():
    hit = Hit(None, None, None, "/files/7#page=2", 0.1, "text", source_title="Отчёт")
    text = format_hit(1, hit)
    assert "Отчёт" in text
    assert "None" not in text

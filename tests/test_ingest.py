import pytest

from app.ingest.chunker import split_telegram_post, token_count
from app.ingest.from_db import (
    extract_hashtags,
    load_posts,
    parse_views,
    telegram_url,
    validate_source_schema,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("12", 12),
        ("1.39K", 1390),
        ("2,4K", 2400),
        ("1.2M", 1_200_000),
        ("abc", None),
    ],
)
def test_parse_views(raw, expected):
    assert parse_views(raw) == expected


def test_extract_hashtags_deduplicates_and_lowercases():
    text = "#RAG #rag #n8n #чатботы, #chat_bot"
    assert extract_hashtags(text) == ["chat_bot", "n8n", "rag", "чатботы"]


def test_telegram_deep_link():
    assert telegram_url("freelance_zakazy", 24517565) == ("https://t.me/freelance_zakazy/24517565")


def test_source_schema_validation():
    assert validate_source_schema("market_research") == "market_research"
    with pytest.raises(ValueError):
        validate_source_schema("market_research; drop schema public")


def test_short_post_is_one_chunk():
    assert split_telegram_post("короткий пост") == ["короткий пост"]


def test_long_post_is_split_and_keeps_limit():
    paragraph = "Это длинный пост про RAG, эмбеддинги и гибридный поиск. " * 20
    text = "\n\n".join([paragraph] * 8)
    chunks = split_telegram_post(text, max_chars=500, overlap=80)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert all(chunk for chunk in chunks)


def test_empty_post_has_no_chunks():
    assert split_telegram_post("   \n\n ") == []


def test_token_count_is_positive():
    assert token_count("") == 1
    assert token_count("один два три четыре") >= 1


def test_load_posts_skips_posts_without_text():
    class FakeConn:
        def execute(self, query, params):
            self.query = query.as_string(None)
            self.params = params
            return self

        def fetchall(self):
            return []

    conn = FakeConn()
    assert load_posts(conn, "market_research", 10, ["job_python"]) == []
    assert "p.text is not null and btrim(p.text) <> ''" in conn.query
    assert "and p.channel = any(%s)" in conn.query
    assert conn.params == [["job_python"], 10]

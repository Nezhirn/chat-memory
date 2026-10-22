import pytest

from app.migrate import MIGRATIONS_DIR, migration_files, quote_schema, render_migration


def test_render_migration_quotes_schema():
    assert render_migration("create schema {{schema}};", "rag_demo") == 'create schema "rag_demo";'


def test_render_migration_rejects_injection():
    with pytest.raises(ValueError):
        render_migration("{{schema}}", 'x"; drop schema public; --')


def test_render_migration_substitutes_embedding_dim():
    assert (
        render_migration("embedding vector({{embedding_dim}})", "rag_demo", 256)
        == "embedding vector(256)"
    )


def test_quote_schema_rejects_bad_names():
    for name in ("", "1abc", "rag demo", "rag-demo", "x" * 64):
        with pytest.raises(ValueError):
            quote_schema(name)


def test_migration_files_are_sorted_and_use_placeholder():
    files = migration_files()
    assert files, f"no migrations found in {MIGRATIONS_DIR}"
    assert [path.name for path in files] == sorted(path.name for path in files)
    assert all("{{schema}}" in path.read_text(encoding="utf-8") for path in files)

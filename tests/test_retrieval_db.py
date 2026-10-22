import os

import pytest

from app.config import get_settings
from app.retrieval.embeddings import Embedder, vector_literal
from app.retrieval.retriever import search_hybrid

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_DB_TESTS") != "1",
        reason="set RUN_DB_TESTS=1 to test against a live PostgreSQL",
    ),
]


def test_hybrid_search_returns_telegram_and_document_hits():
    import psycopg

    settings = get_settings()
    embedder = Embedder(settings)
    query_vector = vector_literal(embedder.embed_query("RAG retrieval"))

    with psycopg.connect(settings.database_url) as conn:
        hits = search_hybrid(
            conn,
            settings.db_schema,
            "RAG retrieval",
            query_vector,
            limit=10,
        )
        pdf_hits = search_hybrid(
            conn,
            settings.db_schema,
            "зарплаты кандидата",
            vector_literal(embedder.embed_query("зарплаты кандидата")),
            limit=5,
            source_kind="pdf",
        )

    assert hits
    assert all(0 < hit.score <= 2 / 61 for hit in hits)
    assert pdf_hits
    assert all(hit.source_kind == "pdf" for hit in pdf_hits)

import os

import psycopg
import pytest
from psycopg import sql

from app.config import get_settings
from app.migrate import apply_migrations, current_embedding_dim, resize_embeddings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_DB_TESTS") != "1",
        reason="set RUN_DB_TESTS=1 to test against a live PostgreSQL",
    ),
]

TEST_SCHEMA = "rag_test_resize"


def test_resize_embeddings_changes_dimension_and_clears_vectors():
    settings = get_settings()
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        drop = sql.SQL("drop schema if exists {} cascade").format(sql.Identifier(TEST_SCHEMA))
        conn.execute(drop)
        try:
            apply_migrations(conn, TEST_SCHEMA, embedding_dim=4)
            assert current_embedding_dim(conn, TEST_SCHEMA) == 4

            source_id = conn.execute(
                sql.SQL(
                    "insert into {} (kind, external_id, title) values ('web', 'x', 'x') "
                    "returning id"
                ).format(sql.Identifier(TEST_SCHEMA, "sources"))
            ).fetchone()[0]
            item_id = conn.execute(
                sql.SQL(
                    "insert into {} (source_id, external_id, body, content_hash) "
                    "values (%s, 'x', 'x', 'x') returning id"
                ).format(sql.Identifier(TEST_SCHEMA, "items")),
                (source_id,),
            ).fetchone()[0]
            conn.execute(
                sql.SQL(
                    "insert into {} (item_id, body, content_hash, embedding) "
                    "values (%s, 'x', 'x', '[1,0,0,0]')"
                ).format(sql.Identifier(TEST_SCHEMA, "chunks")),
                (item_id,),
            )

            resize_embeddings(conn, TEST_SCHEMA, 8)

            assert current_embedding_dim(conn, TEST_SCHEMA) == 8
            left = conn.execute(
                sql.SQL("select count(*) from {} where embedding is not null").format(
                    sql.Identifier(TEST_SCHEMA, "chunks")
                )
            ).fetchone()[0]
            assert left == 0
        finally:
            conn.execute(drop)

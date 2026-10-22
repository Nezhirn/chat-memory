import os

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_DB_TESTS") != "1",
        reason="set RUN_DB_TESTS=1 to test against a live PostgreSQL",
    ),
]


def test_pdf_route_returns_the_original_file():
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        row = conn.execute("select id from rag_demo.sources where kind = 'pdf' limit 1").fetchone()
    assert row is not None

    with TestClient(create_app(settings)) as client:
        response = client.get(f"/files/{row[0]}")
        missing = client.get("/files/999999")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    assert missing.status_code == 404

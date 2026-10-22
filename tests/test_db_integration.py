import os

import pytest

from app.config import get_settings
from app.db import Database

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_DB_TESTS") != "1",
        reason="set RUN_DB_TESTS=1 to test against a live PostgreSQL",
    ),
]


async def test_live_database_health():
    db = Database(get_settings())
    await db.open()
    try:
        health = await db.health()
    finally:
        await db.close()

    assert health["status"] == "ok", health.get("error")
    assert health["schema_exists"] is True
    assert health["pgvector"]
    assert health["migrations"] >= 1

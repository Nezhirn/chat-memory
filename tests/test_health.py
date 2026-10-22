from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


class FakeDatabase:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.opened = False
        self.closed = False

    async def open(self) -> None:
        self.opened = True

    async def ensure_ready(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    async def health(self) -> dict:
        return self.payload


def make_app(payload: dict) -> tuple:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://test:test@127.0.0.1:1/test",
    )
    fake = FakeDatabase(payload)
    return create_app(settings=settings, db=fake), fake


def health_payload(status: str = "ok") -> dict:
    return {
        "status": status,
        "schema": "rag_demo",
        "schema_exists": True,
        "database": "chat_memory",
        "user": "chat_memory",
        "pg_version": "17.10",
        "pgvector": "0.8.6",
        "tables": {"sources": 1, "items": 2921, "chunks": 3100},
        "migrations": 1,
        "latency_ms": 1.5,
    }


def test_health_ok_and_lifespan():
    app, fake = make_app(health_payload())
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "chat-memory"
    assert body["db"]["schema"] == "rag_demo"
    assert body["db"]["tables"]["items"] == 2921
    assert fake.opened and fake.closed


def test_health_reports_unavailable_database_as_503():
    payload = health_payload("error")
    payload["error"] = "pgvector extension is missing"
    app, _ = make_app(payload)
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json()["status"] == "error"

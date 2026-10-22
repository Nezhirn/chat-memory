import pytest
from pydantic import ValidationError

from app.config import Settings


def make_settings(**overrides):
    values = {"database_url": "postgresql://test:test@127.0.0.1:1/test", **overrides}
    return Settings(_env_file=None, **values)


def test_defaults():
    settings = make_settings()
    assert settings.db_schema == "rag_demo"
    assert settings.db_pool_min == 1
    assert settings.db_pool_max == 10


def test_schema_rejects_sql_injection():
    with pytest.raises(ValidationError):
        make_settings(db_schema="rag_demo; drop schema public")


def test_schema_rejects_uppercase():
    with pytest.raises(ValidationError):
        make_settings(db_schema="RagDemo")


def test_pool_min_cannot_exceed_max():
    with pytest.raises(ValidationError):
        make_settings(db_pool_min=5, db_pool_max=2)


def test_allowlist_parsing():
    settings = make_settings(telegram_allowlist="42, 7  1001")
    assert settings.telegram_allowlist_ids == {42, 7, 1001}

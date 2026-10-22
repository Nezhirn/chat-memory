import base64

import pytest

from app.config import Settings
from app.ingest import ocr
from app.ingest.ocr import OCR_PROMPT, build_openrouter_payload, extract_ocr_text


def test_openrouter_payload_contains_data_url_and_model():
    payload = build_openrouter_payload(b"\x89PNG fake", "google/gemini-2.5-flash")
    content = payload["messages"][0]["content"]
    assert payload["model"] == "google/gemini-2.5-flash"
    assert content[0]["text"] == OCR_PROMPT
    data_url = content[1]["image_url"]["url"]
    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == b"\x89PNG fake"


def test_openrouter_requires_key(tmp_path):
    settings = Settings(
        _env_file=None,
        database_url="postgresql://test:test@127.0.0.1:1/test",
        ocr_provider="openrouter",
        ocr_cache_dir=str(tmp_path),
        openrouter_api_key=None,
    )
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        extract_ocr_text(b"unique-image-bytes", settings)


def test_tesseract_cache_ignores_ocr_model(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ocr, "_tesseract", lambda image: calls.append(image) or "page text")

    def settings_for(model: str) -> Settings:
        return Settings(
            _env_file=None,
            database_url="postgresql://test:test@127.0.0.1:1/test",
            ocr_provider="tesseract",
            ocr_model=model,
            ocr_cache_dir=str(tmp_path),
        )

    assert extract_ocr_text(b"image", settings_for("model-a")) == "page text"
    assert extract_ocr_text(b"image", settings_for("model-b")) == "page text"
    assert len(calls) == 1

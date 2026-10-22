from __future__ import annotations

import base64
import hashlib
import io
import logging
import re
import time
from pathlib import Path

import pytesseract
import requests
from PIL import Image

from app.config import Settings

logger = logging.getLogger(__name__)

TESSERACT_LANG = "rus+eng"
OCR_PROMPT = (
    "Extract all readable text from this document page exactly as it appears. "
    "Preserve reading order, line breaks, headings, table rows and bullet points. "
    "Do not translate, summarize, explain or add commentary. "
    "Return plain text only; if there is no text, return an empty string."
)


def _cache_path(
    cache_dir: Path,
    provider: str,
    model: str,
    image_bytes: bytes,
) -> Path:
    key = hashlib.sha256(image_bytes).hexdigest()
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    return cache_dir / f"{provider}-{safe_model}-{key}.txt"


def _tesseract(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image, lang=TESSERACT_LANG).strip()


def build_openrouter_payload(image_bytes: bytes, model: str) -> dict:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 8192,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
    }


def _openrouter(image_bytes: bytes, settings: Settings, model: str) -> str:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; use OCR_PROVIDER=tesseract or add a key")
    url = f"{settings.openrouter_base_url.rstrip('/')}/chat/completions"
    response = None
    for attempt in range(3):
        response = requests.post(
            url,
            json=build_openrouter_payload(image_bytes, model),
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": settings.openrouter_site_url,
                "X-Title": settings.openrouter_app_name,
            },
            timeout=120,
        )
        if response.status_code == 200:
            break
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
            break
        wait = 2**attempt
        logger.warning("OpenRouter OCR HTTP %s, retry in %ss", response.status_code, wait)
        time.sleep(wait)
    if response is None or response.status_code != 200:
        status = response.status_code if response is not None else "no response"
        body = response.text[:300] if response is not None else ""
        raise RuntimeError(f"OpenRouter OCR failed: HTTP {status}: {body}")
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    text = (content or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()
    usage = data.get("usage", {})
    logger.info(
        "OpenRouter OCR: %s chars, tokens prompt=%s completion=%s",
        len(text),
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
    )
    return text


def extract_ocr_text(
    image_bytes: bytes,
    settings: Settings,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    provider = provider or settings.ocr_provider
    model = model or settings.ocr_model
    cache_dir = Path(settings.ocr_cache_dir).expanduser()
    # OCR_MODEL only applies to OpenRouter; keep the Tesseract cache stable when it changes.
    cache_model = TESSERACT_LANG if provider == "tesseract" else model
    cache_file = _cache_path(cache_dir, provider, cache_model, image_bytes)
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    if provider == "tesseract":
        text = _tesseract(image_bytes)
    elif provider == "openrouter":
        text = _openrouter(image_bytes, settings, model)
    else:
        raise ValueError(f"unknown OCR provider: {provider!r}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text, encoding="utf-8")
    return text

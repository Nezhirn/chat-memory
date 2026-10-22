from __future__ import annotations

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENT_CHUNK_SIZE = 2_800
DOCUMENT_CHUNK_OVERLAP = 300


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_telegram_post(
    text: str,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[str]:
    """Split a Telegram post into overlapping chunks.

    The split prefers paragraph and line boundaries, so a chunk does not start
    in the middle of a sentence when there is a better break nearby. The
    original message remains the citation unit; chunks only feed retrieval.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            window = text[start:end]
            split_at = window.rfind("\n\n")
            if split_at < max_chars // 2:
                split_at = window.rfind("\n")
            if split_at < max_chars // 2:
                split_at = window.rfind(" ")
            if split_at >= max_chars // 3:
                end = start + split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks


def token_count(text: str) -> int:
    """Cheap approximation used until the embedder tokenizer is wired in."""
    return max(1, len(text) // 4)


def split_document_text(
    text: str,
    chunk_size: int = DOCUMENT_CHUNK_SIZE,
    chunk_overlap: int = DOCUMENT_CHUNK_OVERLAP,
) -> list[str]:
    """Split PDF/web text with LangChain's recursive splitter.

    2 800 characters is roughly 500-800 tokens for Russian and English prose,
    matching the chunk budget in the plan.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]

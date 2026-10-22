from __future__ import annotations

from collections.abc import Iterable

from fastembed import TextEmbedding

from app.config import Settings


def model_prefixes(model_name: str) -> tuple[str, str]:
    """Return (query, passage) prefixes required by the model."""
    if "e5" in model_name.lower():
        return "query: ", "passage: "
    return "", ""


def vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{value:.7f}" for value in values) + "]"


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = TextEmbedding(
            model_name=settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
            threads=settings.embedding_threads,
        )
        self.dim = int(self.model.embedding_size)
        if self.dim != settings.embedding_dim:
            raise ValueError(
                f"model {settings.embedding_model!r} produces {self.dim}-d vectors, "
                f"but EMBEDDING_DIM={settings.embedding_dim}; set EMBEDDING_DIM={self.dim} "
                "and run python -m app.migrate --resize-embeddings"
            )
        self.query_prefix, self.passage_prefix = model_prefixes(settings.embedding_model)

    def embed_passages(self, texts: list[str], batch_size: int = 256) -> list[list[float]]:
        prefixed = [f"{self.passage_prefix}{text}" for text in texts]
        return [vector.tolist() for vector in self.model.embed(prefixed, batch_size=batch_size)]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self.model.embed([f"{self.query_prefix}{text}"], batch_size=1)))
        return vector.tolist()

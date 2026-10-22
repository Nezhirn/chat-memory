-- Resize the embedding column to the configured model dimension.
-- Demo embeddings are disposable: `using null` drops old vectors and the
-- HNSW index is rebuilt. To switch models later, change EMBEDDING_DIM, run
-- `python -m app.migrate --resize-embeddings` (re-executes this file) and then
-- the embed command.

DROP INDEX IF EXISTS {{schema}}.chunks_embedding_hnsw;

ALTER TABLE {{schema}}.chunks
    ALTER COLUMN embedding TYPE vector({{embedding_dim}}) USING NULL;

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON {{schema}}.chunks USING hnsw (embedding vector_cosine_ops);

-- Initial schema for Chat Memory.
-- {{schema}} is replaced by app.migrate with the configured DB_SCHEMA.

CREATE SCHEMA IF NOT EXISTS {{schema}};

CREATE TABLE IF NOT EXISTS {{schema}}.sources (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind        text NOT NULL CHECK (kind IN ('telegram', 'pdf', 'web')),
    external_id text NOT NULL,
    title       text NOT NULL,
    url         text,
    meta        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT sources_kind_external_id_key UNIQUE (kind, external_id)
);

CREATE TABLE IF NOT EXISTS {{schema}}.items (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id    bigint NOT NULL REFERENCES {{schema}}.sources (id) ON DELETE CASCADE,
    external_id  text NOT NULL,
    channel      text,
    message_id   bigint,
    thread_id    bigint,
    reply_to_id  bigint,
    author       text,
    published_at timestamptz,
    body         text NOT NULL,
    hashtags     text[] NOT NULL DEFAULT '{}',
    views        integer,
    url          text,
    content_hash text NOT NULL,
    meta         jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT items_source_external_id_key UNIQUE (source_id, external_id)
);

CREATE INDEX IF NOT EXISTS items_source_date_idx
    ON {{schema}}.items (source_id, published_at);
CREATE INDEX IF NOT EXISTS items_published_at_idx
    ON {{schema}}.items (published_at);
CREATE INDEX IF NOT EXISTS items_hashtags_gin
    ON {{schema}}.items USING gin (hashtags);

CREATE TABLE IF NOT EXISTS {{schema}}.chunks (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    item_id     bigint NOT NULL REFERENCES {{schema}}.items (id) ON DELETE CASCADE,
    ord         integer NOT NULL DEFAULT 0,
    heading     text,
    body        text NOT NULL,
    token_count integer,
    tsv         tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('russian', coalesce(heading, '')), 'A')
        ||
        setweight(to_tsvector('russian', body), 'B')
    ) STORED,
    embedding   vector(1024),
    content_hash text NOT NULL,
    meta        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT chunks_item_id_ord_key UNIQUE (item_id, ord)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin
    ON {{schema}}.chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON {{schema}}.chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS {{schema}}.ingest_runs (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source      text NOT NULL,
    status      text NOT NULL CHECK (status IN ('running', 'ok', 'failed')),
    stats       jsonb NOT NULL DEFAULT '{}'::jsonb,
    error       text,
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

CREATE INDEX IF NOT EXISTS ingest_runs_started_at_idx
    ON {{schema}}.ingest_runs (started_at DESC);

CREATE TABLE IF NOT EXISTS {{schema}}.qa_log (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    question          text NOT NULL,
    answer            text,
    citations         jsonb NOT NULL DEFAULT '[]'::jsonb,
    tool_calls        jsonb NOT NULL DEFAULT '[]'::jsonb,
    model             text,
    latency_ms        integer,
    prompt_tokens     integer,
    completion_tokens integer,
    cost_usd          numeric(12, 6),
    feedback          smallint CHECK (feedback IN (-1, 1)),
    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS qa_log_created_at_idx
    ON {{schema}}.qa_log (created_at DESC);

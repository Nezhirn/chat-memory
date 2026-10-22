# chat-memory

«Память чата» — RAG-система поверх архива Telegram-канала: гибридный поиск по постам,
цитаты на уровне конкретных сообщений, агент с инструментами и измеримое качество на
golden set.

Pet-проект для портфолио, не коммерческий опыт. Полный план и обоснования решений —
в [PLAN.md](PLAN.md).

**Статус:** вечер 3 закрыт: Telegram + PDF + web, эмбеддинги, гибридный поиск
FTS + dense через RRF, бот с `/search`. Остались HTML/JSON-адаптеры из вечера 2,
затем реранкер, агент с инструментами и eval.

## Стек

- Python 3.12, FastAPI, Pydantic Settings, psycopg 3 (async pool).
- PostgreSQL 17, pgvector 0.8.6, HNSW, GIN/tsvector.
- fastembed + `potion-multilingual-128M` (256d) для демо; `e5-large` — замена под GPU.
- PyMuPDF + OCR: Tesseract `rus+eng` локально или OpenRouter vision
  (`google/gemini-2.5-flash`), trafilatura, LangChain text splitters.
- pyTelegramBotAPI (`AsyncTeleBot`), Docker Compose, pytest, ruff.

## Быстрый старт

### Локально

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env          # указать DATABASE_URL, DB_PASSWORD и TELEGRAM_BOT_TOKEN
make migrate
make run
make health
```

`/healthz` возвращает состояние базы, версию pgvector, наличие схемы и количество строк.
Если база недоступна, расширения или схемы нет — ответ `503` с причиной.

Бот запускается отдельным процессом:

```bash
make bot
```

Пока он умеет `/start`, `/search`, `/health`, `/sources`, `/stats`, `/id` и `/help`.
`/search` — гибридный поиск по Telegram, PDF и веб-страницам, с ссылками на исходники;
ответы с цитатами и агент появятся после подключения retrieval. `TELEGRAM_ALLOWLIST` —
список Telegram user id через запятую; пустой список означает открытый доступ (только
для локальной разработки). Если Bot API доступен только через прокси, задай
`TELEGRAM_PROXY` — он передаётся в aiohttp.

### Docker Compose

```bash
cp .env.example .env          # задать DB_PASSWORD
docker compose up --build
curl -s localhost:8088/healthz | python3 -m json.tool
```

`api` при старте применяет миграции и затем запускает Uvicorn. `bot` — отдельный
контейнер с long polling; `db` — `pgvector/pgvector:pg17` с volume и healthcheck.

### Существующий PostgreSQL

Если база уже есть (например, контейнер `postgres` в сети `python_api_base`):

```bash
docker compose -f compose.external.yml up --build
```

`DATABASE_URL` берётся из `.env`; контейнер подключается к внешней сети
`python_api_base`. Расширение `vector` и роль должны быть созданы заранее:

```sql
CREATE ROLE chat_memory LOGIN PASSWORD '<password>';
CREATE DATABASE chat_memory OWNER chat_memory;
\c chat_memory
CREATE EXTENSION IF NOT EXISTS vector;
```

## Структура

```text
app/
  main.py            FastAPI, lifespan, /healthz, /files/{id}
  bot.py             Telegram-бот: /search, /health, /sources, /stats
  search.py          CLI: FTS или гибридный поиск
  config.py          настройки из env и .env
  db.py              async pool, health, ensure_ready
  models.py          Pydantic-модели ответа /healthz
  migrate.py         идемпотентный раннер SQL-миграций
  ingest/
    from_db.py       импорт Telegram из ai_base.market_research
    pdf.py           PyMuPDF, OCR fallback, страница = item
    web.py           trafilatura, страница = item
    embed.py         заполнение chunks.embedding
    store.py         общие upsert-ы sources/items/chunks
    chunker.py       Telegram-чанкер и RecursiveCharacterTextSplitter
  retrieval/
    embeddings.py    fastembed-обёртка, префиксы, vector literal
    retriever.py     FTS + dense, RRF k=60, фильтры
db/
  migrations/001_init.sql, 002_embedding_dim.sql
  init/00_extensions.sql
data/web_urls.txt    демо-список веб-страниц
tests/
  test_config.py, test_migrate.py, test_health.py, test_bot.py,
  test_ingest.py, test_ingest_documents.py, test_search.py,
  test_retrieval.py, test_db_integration.py, test_retrieval_db.py,
  test_files_api.py
docker-compose.yml, compose.external.yml, Dockerfile, Makefile
```

## Схема данных

Схема задаётся `DB_SCHEMA` (по умолчанию `rag_demo`) и подставляется в SQL миграций
через плейсхолдер `{{schema}}`. В коде нет захардкоженного имени схемы — все запросы
строятся через `psycopg.sql.Identifier`.

Таблицы: `sources`, `items`, `chunks`, `ingest_runs`, `qa_log`.

Решения, которые стоит отметить:

- `chunks.tsv` — генерируемая колонка с весами `A` (заголовок) и `B` (тело), а не
  поле, которое заполняет приложение. Индекс не может рассинхронизироваться с текстом.
- Миграции применяются в транзакции вместе с записью в `schema_migrations`;
  параллельные запуски защищены advisory lock.
- `hnsw.ef_search` выставляется на каждое соединение пула.
- `/healthz` не падает при недоступной базе: возвращает `503` и причину, чтобы
  healthcheck и мониторинг видели деградацию.
- Локальный `.env` имеет приоритет над переменными окружения: рабочие шеллы часто
  экспортируют `DATABASE_URL` от другого проекта. В контейнере файла `.env` нет,
  поэтому там настройки приходят из окружения.

## Ингест

### Telegram

Импорт из read-only источника `ai_base.market_research`:

```bash
make ingest
# или с параметрами
python -m app.ingest from-db --limit 3000 --dry-run
python -m app.ingest from-db --channel job_python --limit 100
```

Импортёр читает `tg_posts` и `tg_channels`, строит deep link
`https://t.me/{channel}/{message_id}`, парсит счётчики просмотров (`1.39K` → 1390),
извлекает хештеги, пишет `ingest_runs` и идемпотентно обновляет `items`/`chunks`.
Длинные посты режутся на чанки до 1 200 символов с перекрытием 150; цитата всегда
остаётся на исходное сообщение. Повторный прогон даёт 0 вставок и 2 921 обновление.

### PDF

```bash
make ingest-pdf PDF=/path/to/report.pdf
# или
python -m app.ingest pdf --path report.pdf --limit-pages 50 --dry-run
```

PyMuPDF сначала берёт текстовый слой. Если на странице меньше 100 символов —
рендер 300 DPI и OCR. Провайдер выбирается через `OCR_PROVIDER`:

- `tesseract` — локально, офлайн, без стоимости;
- `openrouter` — vision-модель (`OCR_MODEL`, по умолчанию `google/gemini-2.5-flash`),
  нужен `OPENROUTER_API_KEY`.

Результат кешируется в `data/cache/ocr` по хешу страницы, провайдеру и модели.
Один item — одна страница, ссылка вида `/files/{source_id}#page=N`; FastAPI отдаёт
оригинальный PDF по `/files/{id}`. На отчёте из 13 страниц OCR не понадобился;
синтетический скан прошёл через обе ветки — Tesseract и Gemini Flash. Чтобы PDF-ссылки
были кликабельными в Telegram, задай `PUBLIC_BASE_URL`; без него бот показывает
название документа без ссылки.

### Web

```bash
make ingest-web
# или
python -m app.ingest web --url https://example.com/page --dry-run
```

trafilatura извлекает основной текст, заголовок и дату; один item — одна страница.
Список демо-страниц лежит в `data/web_urls.txt` (13 официальных документаций стека:
pgvector, PostgreSQL FTS, FastAPI, LangChain, LangGraph, fastembed, trafilatura и др.).

## Эмбеддинги и гибридный поиск

```bash
make embed                 # только чанки без вектора
make embed ARGS="--rebuild"  # пересчитать все
```

Dense-ветка: `chunks.embedding` (pgvector, HNSW, cosine). FTS-ветка: `chunks.tsv`
(GIN, `websearch_to_tsquery('russian')`). Гибрид берёт top-50 из каждой ветки и
объединяет через RRF с `k=60`; фильтры — канал, тип источника, даты, хештеги.

```bash
python -m app.search "HNSW pgvector" --limit 5
python -m app.search "зарплаты кандидата" --source-kind pdf
python -m app.search "n8n" --channel rabota_v_ii --mode fts
python -m app.search "RAG" --date-from 2026-09-01 --json
```

Демо-корпус: **67 источников, 2 947 items, 4 277 чанков**, все векторы заполнены.
Для модели `potion-multilingual-128M` (256d) индексация занимает ~10 секунд
(~400 чанков/с). `e5-large` (1024d) на этом N100 даёт ~0.34 текста/с — около 3,5 часов
на корпус, поэтому в демо выбрана быстрая модель. Это осознанный trade-off: качество
dense-ветки ниже, но FTS + RRF и полный корпус работают сразу.

Смена модели на другую размерность (например, e5-large, 1024d):

```bash
# в .env: EMBEDDING_MODEL=intfloat/multilingual-e5-large, EMBEDDING_DIM=1024
make migrate ARGS="--resize-embeddings"   # меняет vector(N), обнуляет все векторы
make embed                                # пересчитывает векторы заново
```

Без флага `make migrate` при расхождении `EMBEDDING_DIM` и колонки
`chunks.embedding` завершается с ошибкой и ничего не удаляет.

Бот использует тот же гибридный поиск, но схлопывает результаты до одного лучшего
чанка на item, чтобы одна страница PDF не занимала всю выдачу.

## Тесты

```bash
make test
make lint
```

Интеграционный тест работает только с живой базой и явным флагом:

```bash
RUN_DB_TESTS=1 .venv/bin/pytest -q tests/test_db_integration.py \
  tests/test_retrieval_db.py tests/test_files_api.py
```

## Дальше

HTML-адаптер `t.me/s`, Telegram Desktop `result.json`, реранкер, LangGraph-агент
с цитатами и golden set на 50 вопросах.

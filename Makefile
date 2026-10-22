VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install migrate run bot health test lint fmt up down logs db-shell ingest ingest-pdf ingest-web embed search

venv:
	test -d $(VENV) || python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

migrate:
	$(PYTHON) -m app.migrate $(ARGS)

run:
	$(VENV)/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8080

bot:
	$(PYTHON) -m app.bot

ingest:
	$(PYTHON) -m app.ingest from-db --limit 3000

ingest-pdf:
	$(PYTHON) -m app.ingest pdf --path "$(PDF)" --limit-pages 50

ingest-web:
	$(PYTHON) -m app.ingest web --limit 20

embed:
	$(PYTHON) -m app.ingest embed $(ARGS)

search:
	$(PYTHON) -m app.search "$(Q)"

health:
	curl -s http://127.0.0.1:8080/healthz | $(PYTHON) -m json.tool

test:
	$(VENV)/bin/pytest -q

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

fmt:
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api

db-shell:
	docker compose exec db psql -U chat_memory -d chat_memory

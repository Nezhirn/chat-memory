from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from psycopg import sql

from app.config import Settings, get_settings
from app.db import Database
from app.models import HealthResponse


def create_app(settings: Settings | None = None, db: Database | None = None) -> FastAPI:
    settings = settings or get_settings()
    database = db or Database(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await database.open()
        try:
            await database.ensure_ready()
        except Exception:
            await database.close()
            raise
        try:
            yield
        finally:
            await database.close()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="RAG over a Telegram archive with citations",
        lifespan=lifespan,
    )
    app.state.db = database
    app.state.settings = settings

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        responses={503: {"description": "Database is not ready"}},
    )
    async def healthz(response: Response) -> HealthResponse:
        health = await database.health()
        if health["status"] != "ok":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status=health["status"],
            service=settings.app_name,
            version=settings.app_version,
            db=health,
        )

    @app.get("/files/{source_id}", response_class=FileResponse)
    async def get_pdf(source_id: int) -> FileResponse:
        query = sql.SQL(
            "select meta->>'file_path', title from {sources} where id = %s and kind = 'pdf'"
        ).format(sources=sql.Identifier(settings.db_schema, "sources"))
        async with database.connection() as conn:
            result = await conn.execute(query, (source_id,))
            row = await result.fetchone()
        if row is None or not row[0]:
            raise HTTPException(status_code=404, detail="PDF source not found")
        path = Path(row[0]).expanduser()
        if not path.is_file():
            raise HTTPException(status_code=404, detail="PDF file is missing on disk")
        return FileResponse(path, media_type="application/pdf", filename=path.name)

    return app


logging.basicConfig(level=get_settings().log_level)
app = create_app()

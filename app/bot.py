from __future__ import annotations

import asyncio
import logging
import os
from html import escape
from typing import Any

from psycopg import sql
from telebot import asyncio_helper
from telebot.async_telebot import AsyncTeleBot
from telebot.types import BotCommand, Message

from app.config import Settings, get_settings
from app.db import Database
from app.retrieval.embeddings import Embedder, vector_literal
from app.retrieval.retriever import HybridHit, search_hybrid_async
from app.search import Hit, search_chunks_async

logger = logging.getLogger(__name__)

START_TEXT = (
    "Я «Память чата» — RAG-бот по архиву Telegram, PDF и веб-страницам.\n\n"
    "Работает гибридный поиск (FTS + эмбеддинги + RRF): /search &lt;запрос&gt;. "
    "Ответы с цитатами и агент подключу после retrieval.\n\n"
    "Команды: /search, /health, /sources, /stats, /id, /help"
)

HELP_TEXT = (
    "<b>Команды</b>\n"
    "/search &lt;запрос&gt; — гибридный поиск по всем источникам\n"
    "/health — состояние базы, pgvector и схемы\n"
    "/sources — какие источники уже загружены\n"
    "/stats — размер корпуса и диапазон дат\n"
    "/id — твой Telegram user id для allowlist\n"
    "/help — это сообщение\n\n"
    "FTS и dense-ветка объединяются через RRF; ответы с цитатами и агент — дальше."
)

NOT_READY_TEXT = (
    "Для поиска используй /search &lt;запрос&gt;. Свободный чат с агентом подключу после retrieval."
)


def is_allowed(user_id: int, allowlist: set[int]) -> bool:
    """An empty allowlist means the bot is open (local development)."""
    return not allowlist or user_id in allowlist


def format_status(health: dict[str, Any]) -> str:
    tables = health.get("tables", {})
    status = "ok" if health.get("status") == "ok" else "error"
    lines = [
        f"<b>Статус:</b> {status}",
        f"<b>База:</b> {escape(str(health.get('database') or '—'))}",
        f"<b>Схема:</b> {escape(str(health.get('schema') or '—'))}",
        f"<b>pgvector:</b> {escape(str(health.get('pgvector') or '—'))}",
        "<b>Источники / items / chunks:</b> "
        f"{tables.get('sources', 0)} / {tables.get('items', 0)} / {tables.get('chunks', 0)}",
        f"<b>Миграции:</b> {health.get('migrations', 0)}",
        f"<b>Ответ, мс:</b> {health.get('latency_ms', 0)}",
    ]
    if health.get("error"):
        lines.append(f"<b>Ошибка:</b> {escape(str(health['error']))}")
    return "\n".join(lines)


def format_sources(rows: list[tuple[str, str, str | None, int]]) -> str:
    if not rows:
        return "Источников пока нет. Ингест корпуса — следующий шаг по плану."
    lines = ["<b>Источники</b>"]
    for kind, title, url, count in rows:
        label = escape(title)
        if url:
            label = f'<a href="{escape(url, quote=True)}">{label}</a>'
        lines.append(f"• {escape(kind)} — {label}: {count} элементов")
    return "\n".join(lines)


def format_stats(health: dict[str, Any], first: Any, last: Any) -> str:
    tables = health.get("tables", {})
    lines = [
        "<b>Корпус</b>",
        f"Источники: {tables.get('sources', 0)}",
        f"Items: {tables.get('items', 0)}",
        f"Chunks: {tables.get('chunks', 0)}",
        f"Период: {format_date(first)} — {format_date(last)}",
        f"pgvector: {health.get('pgvector') or '—'}",
    ]
    return "\n".join(lines)


def format_date(value: Any) -> str:
    if value is None:
        return "—"
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def format_link(label: str, url: str | None, base_url: str | None = None) -> str:
    """Escape a label and wrap it in a link; relative URLs need PUBLIC_BASE_URL."""
    label = escape(label)
    if url and url.startswith("/"):
        url = f"{base_url.rstrip('/')}{url}" if base_url else None
    return f'<a href="{escape(url, quote=True)}">{label}</a>' if url else label


def format_search_hits(query: str, hits: list[Hit], base_url: str | None = None) -> str:
    if not hits:
        return f"По запросу «{escape(query)}» ничего не найдено."
    lines = [f"<b>Найдено {len(hits)}</b> по запросу «{escape(query)}»"]
    for index, hit in enumerate(hits, start=1):
        title = format_link(hit.label, hit.url, base_url)
        lines.append("")
        lines.append(f"{index}. {title} · {format_date(hit.published_at)}")
        lines.append(escape(hit.headline.strip()))
    return "\n".join(lines)


def format_hybrid_hits(
    query: str,
    hits: list[HybridHit],
    base_url: str | None = None,
) -> str:
    if not hits:
        return f"По запросу «{escape(query)}» ничего не найдено."
    lines = [f"<b>Найдено {len(hits)}</b> · гибридный поиск: «{escape(query)}»"]
    for index, hit in enumerate(hits, start=1):
        label = f"{hit.channel}/{hit.message_id}" if hit.channel else hit.source_title
        title = format_link(label, hit.url, base_url)
        snippet = escape(" ".join(hit.body.split())[:240])
        lines.append("")
        lines.append(f"{index}. {title} · {format_date(hit.published_at)}")
        lines.append(snippet)
    return "\n".join(lines)


def dedupe_hybrid_hits(hits: list[HybridHit], limit: int = 5) -> list[HybridHit]:
    """Keep the best chunk per item so one page does not flood the answer."""
    result: list[HybridHit] = []
    seen: set[int] = set()
    for hit in hits:
        if hit.item_id in seen:
            continue
        seen.add(hit.item_id)
        result.append(hit)
        if len(result) >= limit:
            break
    return result


def split_message(text: str, limit: int = 4_000) -> list[str]:
    """Split a long Telegram message on line boundaries."""
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current = ""
    for line in text.splitlines():
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit and current:
            parts.append(current)
            current = line
        else:
            current = candidate
        while len(current) > limit:
            parts.append(current[:limit])
            current = current[limit:]
    if current:
        parts.append(current)
    return parts


class ChatMemoryBot:
    def __init__(self, settings: Settings, db: Database) -> None:
        if not settings.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is not set")
        self.settings = settings
        self.db = db
        self.bot = AsyncTeleBot(settings.telegram_bot_token)
        self._embedder: Embedder | None = None
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.bot.message_handler(commands=["start"])
        async def handle_start(message: Message) -> None:
            if await self._check_access(message):
                await self.bot.reply_to(message, START_TEXT, parse_mode="HTML")

        @self.bot.message_handler(commands=["help"])
        async def handle_help(message: Message) -> None:
            if await self._check_access(message):
                await self.bot.reply_to(message, HELP_TEXT, parse_mode="HTML")

        @self.bot.message_handler(commands=["health"])
        async def handle_health(message: Message) -> None:
            if not await self._check_access(message):
                return
            health = await self.db.health()
            await self.bot.reply_to(message, format_status(health), parse_mode="HTML")

        @self.bot.message_handler(commands=["sources"])
        async def handle_sources(message: Message) -> None:
            if not await self._check_access(message):
                return
            try:
                rows = await self._source_rows()
                text = format_sources(rows)
            except Exception:
                logger.exception("failed to load sources")
                text = "Не смогла прочитать источники: база недоступна. Подробности в логах."
            await self._reply_long(message, text)

        @self.bot.message_handler(commands=["search"])
        async def handle_search(message: Message) -> None:
            if not await self._check_access(message):
                return
            query = (message.text or "").partition(" ")[2].strip()
            if not query:
                await self.bot.reply_to(
                    message,
                    "Использование: <code>/search запрос</code>",
                    parse_mode="HTML",
                )
                return
            try:
                embedder = await asyncio.to_thread(self._get_embedder)
                query_vector = vector_literal(await asyncio.to_thread(embedder.embed_query, query))
                hits = await search_hybrid_async(self.db, query, query_vector, limit=20)
                text = format_hybrid_hits(
                    query,
                    dedupe_hybrid_hits(hits, 5),
                    base_url=self.settings.public_base_url,
                )
            except Exception:
                logger.exception("hybrid search failed, falling back to fts")
                try:
                    hits = await search_chunks_async(self.db, query, limit=5)
                    text = format_search_hits(query, hits, base_url=self.settings.public_base_url)
                except Exception:
                    logger.exception("fallback fts search failed")
                    text = "Поиск упал: база недоступна. Подробности в логах."
            await self._reply_long(message, text)

        @self.bot.message_handler(commands=["stats"])
        async def handle_stats(message: Message) -> None:
            if not await self._check_access(message):
                return
            try:
                health = await self.db.health()
                first, last = await self._date_range()
                text = format_stats(health, first, last)
            except Exception:
                logger.exception("failed to load stats")
                text = "Не смогла посчитать статистику: база недоступна. Подробности в логах."
            await self.bot.reply_to(message, text, parse_mode="HTML")

        @self.bot.message_handler(commands=["id"])
        async def handle_id(message: Message) -> None:
            if not await self._check_access(message):
                return
            user_id = message.from_user.id if message.from_user else "unknown"
            await self.bot.reply_to(
                message,
                f"<b>user_id:</b> <code>{user_id}</code>\n"
                f"<b>chat_id:</b> <code>{message.chat.id}</code>\n\n"
                "Добавь user_id в <code>TELEGRAM_ALLOWLIST</code>, чтобы закрыть доступ.",
                parse_mode="HTML",
            )

        @self.bot.message_handler(content_types=["text"])
        async def handle_text(message: Message) -> None:
            if await self._check_access(message):
                await self.bot.reply_to(message, NOT_READY_TEXT, parse_mode="HTML")

    async def _reply_long(self, message: Message, text: str) -> None:
        for part in split_message(text):
            await self.bot.reply_to(
                message,
                part,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.settings)
        return self._embedder

    async def _check_access(self, message: Message) -> bool:
        user = message.from_user
        if user is not None and is_allowed(user.id, self.settings.telegram_allowlist_ids):
            return True
        logger.warning("telegram access denied: user_id=%s", user.id if user else None)
        await self.bot.reply_to(
            message,
            "Это приватный демо-бот: доступ только по allowlist.",
        )
        return False

    async def _source_rows(self) -> list[tuple[str, str, str | None, int]]:
        sources = sql.Identifier(self.db.schema, "sources")
        items = sql.Identifier(self.db.schema, "items")
        query = sql.SQL(
            "select s.kind, s.title, s.url, count(i.id) "
            "from {sources} s "
            "left join {items} i on i.source_id = s.id "
            "group by s.id, s.kind, s.title, s.url "
            "order by s.title"
        ).format(sources=sources, items=items)
        async with self.db.connection() as conn:
            result = await conn.execute(query)
            return [tuple(row) for row in await result.fetchall()]

    async def _date_range(self) -> tuple[Any, Any]:
        items = sql.Identifier(self.db.schema, "items")
        query = sql.SQL("select min(published_at), max(published_at) from {}").format(items)
        async with self.db.connection() as conn:
            result = await conn.execute(query)
            row = await result.fetchone()
        if row is None:
            return None, None
        return row[0], row[1]

    async def _set_commands(self) -> None:
        commands = [
            BotCommand("start", "о боте"),
            BotCommand("search", "поиск по корпусу"),
            BotCommand("health", "состояние базы"),
            BotCommand("sources", "источники"),
            BotCommand("stats", "статистика корпуса"),
            BotCommand("id", "мой Telegram user id"),
            BotCommand("help", "помощь"),
        ]
        try:
            await self.bot.set_my_commands(commands)
        except Exception:
            logger.warning("failed to set bot commands", exc_info=True)

    async def run(self) -> None:
        await self.db.open()
        try:
            await self.db.ensure_ready()
        except Exception:
            await self.db.close()
            raise

        try:
            await self._set_commands()
            logger.info("telegram bot polling started")
            await self.bot.infinity_polling(timeout=25, request_timeout=40)
        finally:
            await self.db.close()


def main() -> None:
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")

    proxy = settings.telegram_proxy or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if proxy:
        asyncio_helper.proxy = proxy

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bot = ChatMemoryBot(settings, Database(settings))
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()

"""Провайдер Crossref. Ключ не нужен.

Документация: https://api.crossref.org
Контактная почта передаётся в User-Agent — это polite pool, запросы с ним
обслуживаются приоритетно. Параметром mailto Crossref тоже принимает,
но заголовок надёжнее.

Чем дополняет OpenAlex: у Crossref первичные метаданные от самих издателей,
включая часть российских журналов с DOI, а также записи типа "patent"
(побочный, но бесплатный патентный канал).

Аннотации приходят в разметке JATS — её надо чистить.
"""

import re
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

import httpx
from pydantic import Field

from src.models import Document
from src.MODULE_parser.providers.base import BaseProvider, SourceQuery
from src.MODULE_parser.utils.lang import detect as detect_language

API_URL = "https://api.crossref.org/works"
PAGE_SIZE = 100          # Crossref допускает до 1000, но большие страницы часто таймаутят
TIMEOUT = 30.0

# <jats:p>, </jats:title>, <p> и прочая разметка внутри аннотации
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


class CrossrefQuery(SourceQuery):
    """Параметры, специфичные для Crossref."""

    # journal-article, proceedings-article, posted-content (препринты), patent, ...
    types: list[str] = Field(default_factory=lambda: ["journal-article"])
    # Требовать наличие аннотации. Без неё документ для нас почти бесполезен.
    require_abstract: bool = False


class CrossrefProvider(BaseProvider):
    name: ClassVar[str] = "crossref"
    source_type: ClassVar[str] = "science"
    query_model: ClassVar[type[SourceQuery]] = CrossrefQuery

    def __init__(self, mailto: str = "", user_agent: str = "lct-signals/0.1"):
        # Crossref просит контакт: с ним запросы попадают в polite pool.
        agent = f"{user_agent} (mailto:{mailto})" if mailto else user_agent
        self._client = httpx.Client(
            timeout=TIMEOUT,
            headers={"User-Agent": agent},
            follow_redirects=True,
        )

    # --- публичный метод контракта -------------------------------------

    def parse_source(self, query: CrossrefQuery) -> list[Document]:
        documents: list[Document] = []
        cursor = "*"

        while len(documents) < query.limit and cursor:
            rows = min(PAGE_SIZE, query.limit - len(documents))
            payload = self._fetch_page(query, cursor, rows)

            message = payload.get("message", {})
            items = message.get("items", [])
            if not items:
                break

            for item in items:
                document = self._parse_item(item)
                if query.require_abstract and not document.abstract:
                    continue
                documents.append(document)

            cursor = message.get("next-cursor")

        return documents[: query.limit]

    # --- внутреннее -----------------------------------------------------

    def _fetch_page(self, query: CrossrefQuery, cursor: str, rows: int) -> dict[str, Any]:
        params = {
            # query.bibliographic ищет по заголовку, авторам и изданию.
            "query.bibliographic": " ".join(query.terms),
            "rows": rows,
            "cursor": cursor,
            "select": ",".join([
                "DOI", "title", "abstract", "author", "issued", "created",
                "type", "container-title", "URL", "subject", "language",
                "is-referenced-by-count", "publisher",
            ]),
        }

        filters = self._build_filters(query)
        if filters:
            params["filter"] = filters

        response = self._client.get(API_URL, params=params)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _build_filters(query: CrossrefQuery) -> str:
        parts: list[str] = []
        if query.date_from:
            parts.append(f"from-pub-date:{query.date_from.isoformat()}")
        if query.date_to:
            parts.append(f"until-pub-date:{query.date_to.isoformat()}")
        for item_type in query.types:
            parts.append(f"type:{item_type}")
        return ",".join(parts)

    def _parse_item(self, item: dict[str, Any]) -> Document:
        titles = item.get("title") or []
        title = titles[0] if titles else ""
        abstract = self._clean_jats(item.get("abstract", ""))
        doi = item.get("DOI", "")

        return Document(
            provider=self.name,
            doc_id=doi,
            url=item.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
            title=title,
            abstract=abstract,
            published_at=self._parse_date(item),
            # У Crossref язык заполнен редко — тогда определяем по тексту.
            language=item.get("language") or detect_language(title, abstract),
            source_type=self.source_type,
            authors=[
                " ".join(filter(None, [a.get("given"), a.get("family")]))
                for a in item.get("author", [])
            ],
            raw={
                "type": item.get("type"),
                "publisher": item.get("publisher"),
                "container": (item.get("container-title") or [None])[0],
                "subject": item.get("subject", []),
                "cited_by_count": item.get("is-referenced-by-count"),
            },
        )

    @staticmethod
    def _clean_jats(text: str) -> str:
        """Убрать разметку JATS и схлопнуть пробелы."""
        if not text:
            return ""
        return _SPACE_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()

    @staticmethod
    def _parse_date(item: dict[str, Any]) -> Optional[datetime]:
        """Crossref отдаёт дату списком [год, месяц, день], месяца и дня может не быть."""
        for field in ("published", "issued", "created"):
            parts = (item.get(field) or {}).get("date-parts") or []
            if not parts or not parts[0] or parts[0][0] is None:
                continue
            values = parts[0] + [1, 1]  # дополняем до года-месяца-дня
            try:
                return datetime(
                    int(values[0]), int(values[1]), int(values[2]), tzinfo=timezone.utc
                )
            except (ValueError, TypeError):
                continue
        return None

    def close(self) -> None:
        self._client.close()

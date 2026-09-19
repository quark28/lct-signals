"""Провайдер OpenAlex. Нужен бесплатный ключ (api_key в query-параметрах).

Документация: https://developers.openalex.org
С 13.02.2026 ключ обязателен, polite pool через mailto отменён.
Лимиты кредитные: бесплатный ключ ~ $1/сутки, поисковый запрос стоит $0.001,
то есть около 1000 запросов в день. Кеш обязателен.

Аннотация приходит в виде abstract_inverted_index — словаря
{слово: [позиции]}, его надо разворачивать обратно в текст.
"""

from datetime import date, datetime, timezone
from typing import Any, ClassVar, Optional

import httpx
from pydantic import Field

from src.models import Document
from src.MODULE_parser.providers.base import BaseProvider, SourceQuery
from src.MODULE_parser.utils.lang import detect as detect_language

API_URL = "https://api.openalex.org/works"
PAGE_SIZE = 100          # максимум, который принимает OpenAlex
TIMEOUT = 30.0


class OpenAlexQuery(SourceQuery):
    """Параметры, специфичные для OpenAlex."""

    # Ограничить выдачу языками, например ["en", "ru"]. Пусто — без фильтра.
    languages: list[str] = Field(default_factory=list)
    # Идентификаторы концептов OpenAlex (C41008148 и т.п.) как тематический фильтр.
    concepts: list[str] = Field(default_factory=list)


class OpenAlexProvider(BaseProvider):
    name: ClassVar[str] = "openalex"
    source_type: ClassVar[str] = "science"
    query_model: ClassVar[type[SourceQuery]] = OpenAlexQuery

    def __init__(self, api_key: str, user_agent: str = "lct-signals/0.1"):
        if not api_key:
            raise ValueError("OpenAlex требует ключ: задай OPENALEX_KEY в .env")
        self._api_key = api_key
        self._client = httpx.Client(
            timeout=TIMEOUT,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )

    # --- публичный метод контракта -------------------------------------

    def parse_source(self, query: OpenAlexQuery) -> list[Document]:
        filters = self._build_filters(query)
        documents: list[Document] = []
        cursor = "*"

        while len(documents) < query.limit and cursor:
            page_size = min(PAGE_SIZE, query.limit - len(documents))
            payload = self._fetch_page(filters, cursor, page_size)

            results = payload.get("results", [])
            if not results:
                break

            documents.extend(self._parse_work(w) for w in results)
            cursor = payload.get("meta", {}).get("next_cursor")

        return documents[: query.limit]

    # --- внутреннее -----------------------------------------------------

    def _build_filters(self, query: OpenAlexQuery) -> str:
        """Собрать строку filter. Несколько значений одного поля — через '|' (ИЛИ),
        разные поля — через запятую (И)."""
        parts: list[str] = []

        # Поиск по заголовку и аннотации: дешевле и точнее полнотекстового search.
        terms = "|".join(t.replace(",", " ").replace("|", " ") for t in query.terms)
        parts.append(f"title_and_abstract.search:{terms}")

        if query.date_from:
            parts.append(f"from_publication_date:{query.date_from.isoformat()}")
        if query.date_to:
            parts.append(f"to_publication_date:{query.date_to.isoformat()}")
        if query.languages:
            parts.append("language:" + "|".join(query.languages))
        if query.concepts:
            parts.append("concepts.id:" + "|".join(query.concepts))

        return ",".join(parts)

    def _fetch_page(self, filters: str, cursor: str, page_size: int) -> dict[str, Any]:
        params = {
            "filter": filters,
            "per_page": page_size,
            "cursor": cursor,
            "api_key": self._api_key,
        }
        response = self._client.get(API_URL, params=params)
        response.raise_for_status()
        return response.json()

    def _parse_work(self, work: dict[str, Any]) -> Document:
        openalex_id = (work.get("id") or "").rstrip("/").split("/")[-1]  # "W2741809807"
        title = work.get("display_name") or work.get("title") or ""
        abstract = self._restore_abstract(work.get("abstract_inverted_index"))

        # Язык OpenAlex сообщает сам; если поля нет — определяем по тексту.
        language = work.get("language") or detect_language(title, abstract)

        return Document(
            provider=self.name,
            doc_id=openalex_id,
            url=self._best_url(work),
            title=title,
            abstract=abstract,
            published_at=self._parse_date(work.get("publication_date")),
            language=language,
            source_type=self.source_type,
            authors=[
                a.get("author", {}).get("display_name", "")
                for a in work.get("authorships", [])
                if a.get("author")
            ],
            raw={
                "doi": work.get("doi"),
                "type": work.get("type"),
                "cited_by_count": work.get("cited_by_count"),
                "topics": [t.get("display_name") for t in work.get("topics", [])],
                "concepts": [c.get("display_name") for c in work.get("concepts", [])],
                "is_oa": (work.get("open_access") or {}).get("is_oa"),
            },
        )

    @staticmethod
    def _restore_abstract(inverted: Optional[dict[str, list[int]]]) -> str:
        """Развернуть abstract_inverted_index обратно в текст.

        OpenAlex хранит аннотацию как {слово: [позиции в тексте]} —
        по юридическим причинам, чтобы формально не публиковать текст целиком.
        """
        if not inverted:
            return ""

        positions: list[tuple[int, str]] = [
            (pos, word) for word, places in inverted.items() for pos in places
        ]
        positions.sort()
        return " ".join(word for _, word in positions)

    @staticmethod
    def _best_url(work: dict[str, Any]) -> str:
        """Ссылка на первоисточник: DOI, потом посадочная страница, потом OpenAlex."""
        if work.get("doi"):
            return work["doi"]  # уже в виде https://doi.org/...
        landing = (work.get("primary_location") or {}).get("landing_page_url")
        return landing or work.get("id", "")

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def close(self) -> None:
        self._client.close()
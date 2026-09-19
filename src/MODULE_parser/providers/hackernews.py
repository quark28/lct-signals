"""Провайдер Hacker News через поиск Algolia. Ключ не нужен.

Документация: https://hn.algolia.com/api
Нужен как индикатор интереса разработчиков — ранний сигнал, который
появляется до публикаций и до раундов.

Algolia не поддерживает ИЛИ между фразами в одном запросе,
поэтому по каждой фразе идёт отдельный запрос, результаты склеиваются.
"""

import time
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

import httpx
from pydantic import Field

from src.models import Document
from src.MODULE_parser.providers.base import BaseProvider, SourceQuery
from src.MODULE_parser.utils.lang import detect as detect_language

API_URL = "https://hn.algolia.com/api/v1/search"
ITEM_URL = "https://news.ycombinator.com/item?id={}"
PAGE_SIZE = 100          # максимум, который отдаёт Algolia за запрос
REQUEST_DELAY = 0.2      # Algolia щедрая, но не злоупотребляем
TIMEOUT = 20.0


class HackerNewsQuery(SourceQuery):
    """Параметры, специфичные для Hacker News."""

    # story — посты, comment — комментарии, poll — опросы.
    tags: list[str] = Field(default_factory=lambda: ["story"])
    # Минимальное число баллов: отсекает совсем незамеченные посты.
    min_points: int = 0


class HackerNewsProvider(BaseProvider):
    name: ClassVar[str] = "hackernews"
    source_type: ClassVar[str] = "social"
    query_model: ClassVar[type[SourceQuery]] = HackerNewsQuery

    def __init__(self, user_agent: str = "lct-signals/0.1"):
        self._client = httpx.Client(
            timeout=TIMEOUT,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )

    # --- публичный метод контракта -------------------------------------

    def parse_source(self, query: HackerNewsQuery) -> list[Document]:
        seen: set[str] = set()
        documents: list[Document] = []

        # По фразе на запрос: Algolia не умеет ИЛИ между фразами.
        per_term = max(1, query.limit // len(query.terms))

        for term in query.terms:
            for hit in self._search(term, query, per_term):
                object_id = str(hit.get("objectID", ""))
                if not object_id or object_id in seen:
                    continue
                seen.add(object_id)
                documents.append(self._parse_hit(hit))

            if len(documents) >= query.limit:
                break

        return documents[: query.limit]

    # --- внутреннее -----------------------------------------------------

    def _search(self, term: str, query: HackerNewsQuery, limit: int) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        page = 0

        while len(hits) < limit:
            params = {
                "query": term,
                "tags": ",".join(query.tags) if query.tags else "story",
                "hitsPerPage": min(PAGE_SIZE, limit - len(hits)),
                "page": page,
            }

            numeric = self._numeric_filters(query)
            if numeric:
                params["numericFilters"] = ",".join(numeric)

            response = self._client.get(API_URL, params=params)
            response.raise_for_status()
            payload = response.json()

            batch = payload.get("hits", [])
            if not batch:
                break

            hits.extend(batch)
            page += 1
            if page >= payload.get("nbPages", 0):
                break
            time.sleep(REQUEST_DELAY)

        return hits[:limit]

    def _numeric_filters(self, query: HackerNewsQuery) -> list[str]:
        """Algolia фильтрует по created_at_i — unix-времени в секундах."""
        filters: list[str] = []

        if query.date_from:
            ts = int(datetime.combine(
                query.date_from, datetime.min.time(), tzinfo=timezone.utc
            ).timestamp())
            filters.append(f"created_at_i>={ts}")

        if query.date_to:
            ts = int(datetime.combine(
                query.date_to, datetime.max.time(), tzinfo=timezone.utc
            ).timestamp())
            filters.append(f"created_at_i<={ts}")

        if query.min_points > 0:
            filters.append(f"points>={query.min_points}")

        return filters

    def _parse_hit(self, hit: dict[str, Any]) -> Document:
        object_id = str(hit.get("objectID", ""))
        title = hit.get("title") or hit.get("story_title") or ""
        # У ссылочных постов своего текста нет — тогда аннотация пустая.
        abstract = hit.get("story_text") or hit.get("comment_text") or ""

        return Document(
            provider=self.name,
            doc_id=object_id,
            # Ссылка на обсуждение, а не на внешний материал: нам важен сам факт
            # обсуждения на HN, а внешний url сохраняем в raw.
            url=ITEM_URL.format(object_id),
            title=title,
            abstract=abstract,
            published_at=self._parse_date(hit.get("created_at")),
            language=detect_language(title, abstract),
            source_type=self.source_type,
            authors=[hit.get("author")] if hit.get("author") else [],
            raw={
                "external_url": hit.get("url"),
                "points": hit.get("points"),
                "num_comments": hit.get("num_comments"),
                "tags": hit.get("_tags", []),
            },
        )

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return None

    def close(self) -> None:
        self._client.close()

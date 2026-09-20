"""Провайдер Хабра. Ключ не нужен.

У Хабра нет API и нет поискового RSS — RSS отдаёт только ленты хабов и
"лучшее за период". Поэтому разбираем HTML страницы поиска:
https://habr.com/ru/search/?q=...&target_type=posts&order=date

Разметка может измениться в любой момент — это не API, а вёрстка сайта.
Поэтому селекторы вынесены в константы, а при нулевом разборе непустой
страницы бросается HabrLayoutChanged, чтобы поломка была заметна сразу,
а не выглядела как "по запросу ничего не найдено".

Зачем нужен: русскоязычный техносегмент и признак доли медиа.
"""

import time
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

import httpx
from bs4 import BeautifulSoup
from pydantic import Field

from src.models import Document
from src.MODULE_parser.providers.base import BaseProvider, SourceQuery
from src.MODULE_parser.utils.lang import detect as detect_language

SEARCH_URL = "https://habr.com/ru/search/"
BASE_URL = "https://habr.com"
REQUEST_DELAY = 1.5      # вёрстку парсим вежливо
TIMEOUT = 25.0

# Селекторы вынесены отдельно: если Хабр поменяет вёрстку, править тут.
SEL_ARTICLE = "article.tm-articles-list__item"
SEL_TITLE_LINK = "a.tm-title__link"
SEL_SNIPPET = "div.article-formatted-body"
SEL_AUTHOR = "a.tm-user-info__username"
SEL_HUB = "a.tm-publication-hub__link"


class HabrLayoutChanged(RuntimeError):
    """Страница пришла непустой, но ни одной статьи разобрать не удалось."""


class HabrQuery(SourceQuery):
    """Параметры, специфичные для Хабра."""

    # posts — статьи, comments — комментарии, users — пользователи.
    target_type: str = "posts"
    # date — новые сначала (нужно нам), relevance — по релевантности.
    order: str = "date"
    flows: list[str] = Field(default_factory=list)  # потоки: develop, admin, design...


class HabrProvider(BaseProvider):
    name: ClassVar[str] = "habr"
    source_type: ClassVar[str] = "media"
    query_model: ClassVar[type[SourceQuery]] = HabrQuery

    def __init__(self, user_agent: str = "Mozilla/5.0 (compatible; lct-signals/0.1)"):
        self._client = httpx.Client(
            timeout=TIMEOUT,
            headers={
                "User-Agent": user_agent,
                "Accept-Language": "ru,en;q=0.9",
            },
            follow_redirects=True,
        )

    # --- публичный метод контракта -------------------------------------

    def parse_source(self, query: HabrQuery) -> list[Document]:
        seen: set[str] = set()
        documents: list[Document] = []
        per_term = max(1, query.limit // len(query.terms))

        for term in query.terms:
            for document in self._search(term, query, per_term):
                if document.doc_id in seen:
                    continue
                seen.add(document.doc_id)
                documents.append(document)

            if len(documents) >= query.limit:
                break

        return documents[: query.limit]

    # --- внутреннее -----------------------------------------------------

    def _search(self, term: str, query: HabrQuery, limit: int) -> list[Document]:
        documents: list[Document] = []
        page = 1

        while len(documents) < limit:
            html = self._fetch_page(term, query, page)
            batch = self._parse_page(html)
            if not batch:
                break

            for document in batch:
                # order=date даёт свежие сначала: как только ушли раньше
                # нижней границы, листать дальше смысла нет.
                if query.date_from and document.published_at:
                    if document.published_at.date() < query.date_from:
                        return documents
                if query.date_to and document.published_at:
                    if document.published_at.date() > query.date_to:
                        continue
                documents.append(document)

            page += 1
            time.sleep(REQUEST_DELAY)

        return documents[:limit]

    def _fetch_page(self, term: str, query: HabrQuery, page: int) -> str:
        params: dict[str, Any] = {
            "q": term,
            "target_type": query.target_type,
            "order": query.order,
        }
        if page > 1:
            params["page"] = page
        if query.flows:
            params["flows"] = ",".join(query.flows)

        response = self._client.get(SEARCH_URL, params=params)
        response.raise_for_status()
        return response.text

    def _parse_page(self, html: str) -> list[Document]:
        soup = BeautifulSoup(html, "lxml")
        articles = soup.select(SEL_ARTICLE)

        if not articles:
            # Отличаем «ничего не найдено» от «вёрстка поменялась»:
            # в первом случае на странице не будет и контейнера списка.
            if soup.select_one("div.tm-articles-list") is not None:
                return []
            raise HabrLayoutChanged(
                "Хабр: ни одной статьи не разобрано. Проверь селекторы в habr.py"
            )

        documents: list[Document] = []
        for article in articles:
            document = self._parse_article(article)
            if document:
                documents.append(document)
        return documents

    def _parse_article(self, article) -> Optional[Document]:
        link = article.select_one(SEL_TITLE_LINK)
        if not link or not link.get("href"):
            return None

        href = link["href"]
        url = href if href.startswith("http") else BASE_URL + href
        # /ru/articles/123456/ -> "123456"
        parts = [p for p in href.split("/") if p]
        doc_id = parts[-1] if parts else url

        title = link.get_text(strip=True)

        snippet_node = article.select_one(SEL_SNIPPET)
        snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""

        time_node = article.find("time")
        published_at = self._parse_date(time_node.get("datetime")) if time_node else None

        author_node = article.select_one(SEL_AUTHOR)
        authors = [author_node.get_text(strip=True)] if author_node else []

        return Document(
            provider=self.name,
            doc_id=doc_id,
            url=url,
            title=title,
            abstract=snippet,
            published_at=published_at,
            language=detect_language(title, snippet),
            source_type=self.source_type,
            authors=authors,
            raw={
                "hubs": [h.get_text(strip=True) for h in article.select(SEL_HUB)],
            },
        )

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[datetime]:
        """Хабр отдаёт ISO-время в атрибуте datetime тега <time>."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        except ValueError:
            return None

    def close(self) -> None:
        self._client.close()

"""Провайдер технологических медиа через RSS. Ключи не нужны.

ЗАЧЕМ. Разбор датасета заказчика показал: 285 ссылок, 206 уникальных
доменов, и повторяется меньше двадцати. Методологи искали по вебу, а не
по списку сайтов. Полностью это воспроизводит только поисковый API
(Yandex Search API), но он платный. RSS — бесплатное приближение,
покрывающее самые частые домены из датасета: TechCrunch, SiliconANGLE,
EE Times, DataCenterDynamics, GeekWire, EU-Startups и другие.

Именно этих источников не хватало: у них компании, раунды и суммы, то есть
ровно те признаки, на которых построены обоснования в датасете
(слово "раунд" встречается там 26 раз, "seed" — 17).

ОГРАНИЧЕНИЕ. RSS отдаёт только последние записи ленты, обычно 20–50 штук,
и не умеет искать по ключевым словам. Поэтому провайдер забирает свежие
записи всех лент и фильтрует их локально по фразам запроса. Для глубокой
истории RSS не годится, для текущих сигналов — вполне.

Список лент задаётся параметром feeds: его можно менять, не трогая код.
Ленты, которые не открылись или изменили формат, пропускаются молча —
одна мёртвая лента не должна ронять источник целиком.
"""

import concurrent.futures
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional
from urllib.parse import urlparse

import feedparser
import httpx
from pydantic import Field

from src.models import Document
from src.MODULE_parser.providers.base import BaseProvider, SourceQuery
from src.MODULE_parser.utils.lang import detect as detect_language

TIMEOUT = 15.0
MAX_PARALLEL_FEEDS = 6

# Ленты, отобранные по частоте домена в датасете заказчика.
# Каждую стоит проверить в браузере: сайты меняют адреса лент.
DEFAULT_FEEDS: list[str] = [
    # англоязычные технологические медиа и новости о финансировании
    "https://techcrunch.com/feed/",
    "https://siliconangle.com/feed/",
    "https://venturebeat.com/feed/",
    "https://www.eetimes.com/feed/",
    "https://www.nextplatform.com/feed/",
    "https://www.geekwire.com/feed/",
    "https://www.datacenterdynamics.com/en/rss/",
    "https://www.eu-startups.com/feed/",
    "https://www.therobotreport.com/feed/",
    "https://roboticsandautomationnews.com/feed/",
    "https://www.securityweek.com/feed/",
    "https://www.tomshardware.com/feeds/all",
    # русскоязычные
    "https://habr.com/ru/rss/articles/?fl=ru",
]


class TechMediaQuery(SourceQuery):
    """Параметры, специфичные для медиа-провайдера."""

    # Список лент. Пусто — берутся DEFAULT_FEEDS.
    feeds: list[str] = Field(default_factory=list)
    # Требовать совпадения хотя бы одной фразы запроса в заголовке
    # или описании. False — забрать ленты целиком (для пакетного сбора).
    filter_by_terms: bool = True


class TechMediaProvider(BaseProvider):
    name: ClassVar[str] = "tech_media"
    source_type: ClassVar[str] = "media"
    query_model: ClassVar[type[SourceQuery]] = TechMediaQuery

    def __init__(
        self,
        feeds: Optional[list[str]] = None,
        user_agent: str = "Mozilla/5.0 (compatible; lct-signals/0.1)",
    ):
        self._feeds = feeds or DEFAULT_FEEDS
        self._client = httpx.Client(
            timeout=TIMEOUT,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )

    # --- публичный метод контракта -------------------------------------

    def parse_source(self, query: TechMediaQuery) -> list[Document]:
        feeds = query.feeds or self._feeds
        needles = [t.lower() for t in query.terms] if query.filter_by_terms else []

        documents: list[Document] = []
        seen: set[str] = set()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(MAX_PARALLEL_FEEDS, len(feeds) or 1)
        ) as pool:
            futures = {pool.submit(self._fetch_feed, url): url for url in feeds}
            for future in concurrent.futures.as_completed(futures):
                try:
                    entries = future.result()
                except Exception:
                    continue  # мёртвая лента не роняет источник

                for entry in entries:
                    document = self._parse_entry(entry, futures[future])
                    if document is None or document.doc_id in seen:
                        continue
                    if not self._matches(document, needles):
                        continue
                    if not self._in_period(document, query):
                        continue
                    seen.add(document.doc_id)
                    documents.append(document)

        documents.sort(
            key=lambda d: (d.published_at is None, d.published_at), reverse=True
        )
        return documents[: query.limit]

    # --- внутреннее -----------------------------------------------------

    def _fetch_feed(self, url: str) -> list[Any]:
        """Скачать и разобрать одну ленту. feedparser сам переваривает
        RSS и Atom, поэтому формат ленты значения не имеет."""
        response = self._client.get(url)
        response.raise_for_status()
        parsed = feedparser.parse(response.content)
        return list(parsed.entries)

    def _parse_entry(self, entry: Any, feed_url: str) -> Optional[Document]:
        link = getattr(entry, "link", "")
        if not link:
            return None

        title = self._clean(getattr(entry, "title", ""))
        summary = self._clean(
            getattr(entry, "summary", "") or getattr(entry, "description", "")
        )

        return Document(
            provider=self.name,
            # Ссылка как идентификатор: своих id у лент нет.
            doc_id=link,
            url=link,
            title=title,
            abstract=summary,
            published_at=self._parse_date(entry),
            language=detect_language(title, summary),
            source_type=self.source_type,
            authors=[getattr(entry, "author", "")] if getattr(entry, "author", "") else [],
            raw={
                "feed": feed_url,
                "feed_domain": urlparse(feed_url).netloc,
                "tags": [t.get("term") for t in getattr(entry, "tags", []) if t.get("term")],
            },
        )

    @staticmethod
    def _matches(document: Document, needles: list[str]) -> bool:
        """Фильтр по фразам запроса. RSS не умеет искать, фильтруем сами."""
        if not needles:
            return True
        haystack = f"{document.title} {document.abstract}".lower()
        return any(needle in haystack for needle in needles)

    @staticmethod
    def _in_period(document: Document, query: TechMediaQuery) -> bool:
        if document.published_at is None:
            return True  # без даты не отбрасываем: дату дополнит извлечение
        moment = document.published_at.date()
        if query.date_from and moment < query.date_from:
            return False
        if query.date_to and moment > query.date_to:
            return False
        return True

    @staticmethod
    def _clean(text: str) -> str:
        """Из описаний RSS убираем разметку: ленты часто отдают HTML."""
        import re

        text = re.sub(r"<[^>]+>", " ", text or "")
        return " ".join(text.split())

    @staticmethod
    def _parse_date(entry: Any) -> Optional[datetime]:
        """feedparser сам разбирает дату в struct_time, независимо
        от того, в каком формате она была в ленте."""
        for field in ("published_parsed", "updated_parsed"):
            parsed = getattr(entry, field, None)
            if parsed:
                return datetime(*parsed[:6], tzinfo=timezone.utc)
        return None

    def close(self) -> None:
        self._client.close()

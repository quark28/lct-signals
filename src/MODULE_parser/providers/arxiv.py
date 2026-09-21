# Провайдер arXiv. Открытый API, ключ не нужен.

# Документация: http://export.arxiv.org/api/query
# Формат ответа — Atom XML, разбирается стандартной библиотекой.
# arXiv просит не чаще одного запроса в 3 секунды.


import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import ClassVar, Optional
from urllib.parse import quote, urlencode

import httpx
from pydantic import Field

from src.models import Document
from src.MODULE_parser.providers.base import BaseProvider, SourceQuery
from src.MODULE_parser.utils.lang import detect as detect_language

API_URL = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

PAGE_SIZE = 100          # arXiv рекомендует не больше 100 за запрос
REQUEST_DELAY = 3.0      # секунд между запросами, требование arXiv
TIMEOUT = 60.0


class ArxivQuery(SourceQuery):
    """Параметры, специфичные для arXiv."""

    categories: list[str] = Field(default_factory=list)  # ["cs.LG", "cs.AI"] — как фильтр


class ArxivProvider(BaseProvider):
    name: ClassVar[str] = "arxiv"
    source_type: ClassVar[str] = "science"
    query_model: ClassVar[type[SourceQuery]] = ArxivQuery

    def __init__(self, user_agent: str = "lct-signals/0.1"):
        self._client = httpx.Client(
            timeout=TIMEOUT,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )

    # --- публичный метод контракта -------------------------------------

    def parse_source(self, query: ArxivQuery) -> list[Document]:
        search_query = self._build_search_query(query)
        documents: list[Document] = []
        start = 0

        while len(documents) < query.limit:
            page_size = min(PAGE_SIZE, query.limit - len(documents))
            entries = self._fetch_page(search_query, start, page_size)
            if not entries:
                break

            documents.extend(self._parse_entry(e) for e in entries)

            if len(entries) < page_size:
                break  # выдача закончилась
            start += page_size
            time.sleep(REQUEST_DELAY)

        return documents[: query.limit]

    # --- внутреннее -----------------------------------------------------

    def _build_search_query(self, query: ArxivQuery) -> str:
        """Собрать строку search_query в синтаксисе arXiv."""
        terms = " OR ".join(f'all:"{t}"' for t in query.terms)
        parts = [f"({terms})"]

        if query.categories:
            cats = " OR ".join(f"cat:{c}" for c in query.categories)
            parts.append(f"({cats})")

        if query.date_from or query.date_to:
            lo = query.date_from.strftime("%Y%m%d") if query.date_from else "19910101"
            hi = query.date_to.strftime("%Y%m%d") if query.date_to else "29991231"
            parts.append(f"submittedDate:[{lo}0000 TO {hi}2359]")

        return " AND ".join(parts)

    def _fetch_page(self, search_query: str, start: int, page_size: int) -> list[ET.Element]:
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        # quote_via=quote кодирует пробел как %20: arXiv не понимает "+" внутри кавычек
        url = f"{API_URL}?{urlencode(params, quote_via=quote)}"
        response = self._client.get(url)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        return root.findall("atom:entry", NS)

    def _parse_entry(self, entry: ET.Element) -> Document:
        abs_url = self._text(entry, "atom:id")
        doc_id = abs_url.rstrip("/").split("/")[-1]  # "2401.01234v1"

        title = self._clean(self._text(entry, "atom:title"))
        abstract = self._clean(self._text(entry, "atom:summary"))

        return Document(
            provider=self.name,
            doc_id=doc_id,
            url=abs_url,
            title=title,
            abstract=abstract,
            published_at=self._parse_date(self._text(entry, "atom:published")),
            language=detect_language(title, abstract),
            source_type=self.source_type,
            authors=[
                self._clean(name.text or "")
                for name in entry.findall("atom:author/atom:name", NS)
            ],
            raw={
                "categories": [
                    c.attrib.get("term", "") for c in entry.findall("atom:category", NS)
                ],
                "primary_category": entry.find("arxiv:primary_category", NS).attrib.get("term")
                if entry.find("arxiv:primary_category", NS) is not None
                else None,
                "updated": self._text(entry, "atom:updated"),
            },
        )

    @staticmethod
    def _text(entry: ET.Element, path: str) -> str:
        node = entry.find(path, NS)
        return (node.text or "") if node is not None else ""

    @staticmethod
    def _clean(text: str) -> str:
        """arXiv переносит строки внутри title и summary — склеиваем."""
        return " ".join(text.split())

    @staticmethod
    def _parse_date(value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return None

    def close(self) -> None:
        self._client.close()
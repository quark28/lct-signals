"""Провайдер Google Patents. Ключ не нужен.

ВНИМАНИЕ: это недокументированный внутренний эндпоинт, который вызывает
поисковый интерфейс самого Google Patents. Официального API у него нет,
и Google может поменять форму ответа без предупреждения — поэтому ошибка
разбора JSON ловится отдельно и не должна ронять весь прогон.

Держим как ВТОРОЙ патентный источник. Первым должен быть USPTO ODP или EPO OPS,
когда придут ключи.

Отдаёт только метаданные и сниппеты: номер публикации, заголовок, даты,
заявитель. Полных текстов и массовой выгрузки здесь нет.
"""

import time
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

import httpx
from pydantic import Field

from src.models import Document
from src.MODULE_parser.providers.base import BaseProvider, SourceQuery
from src.MODULE_parser.utils.lang import detect as detect_language

API_URL = "https://patents.google.com/xhr/query"
PATENT_URL = "https://patents.google.com/patent/{}"
PAGE_SIZE = 100          # максимум, который принимает эндпоинт
REQUEST_DELAY = 1.0      # не документировано, поэтому ведём себя скромно
TIMEOUT = 25.0


class GooglePatentsUnavailable(RuntimeError):
    """Эндпоинт ответил не тем, чего мы ждём. Скорее всего, Google поменял формат."""


class GooglePatentsQuery(SourceQuery):
    """Параметры, специфичные для Google Patents."""

    # Коды классификации CPC, например ["G06N", "H04L"].
    cpc_codes: list[str] = Field(default_factory=list)
    # Фильтровать по дате приоритета (когда идею заявили) или публикации.
    date_field: str = "priority"  # priority | filing | publication
    # Ограничить странами выдачи патента, например ["US", "EP", "RU"].
    countries: list[str] = Field(default_factory=list)


class GooglePatentsProvider(BaseProvider):
    name: ClassVar[str] = "google_patents"
    source_type: ClassVar[str] = "patent"
    query_model: ClassVar[type[SourceQuery]] = GooglePatentsQuery

    def __init__(self, user_agent: str = "lct-signals/0.1"):
        self._client = httpx.Client(
            timeout=TIMEOUT,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
                ),
                "Accept": "application/json",
                "Referer": "https://patents.google.com/",
            },
            follow_redirects=True,
        )

    # --- публичный метод контракта -------------------------------------

    def parse_source(self, query: GooglePatentsQuery) -> list[Document]:
        seen: set[str] = set()
        documents: list[Document] = []
        page = 0

        while len(documents) < query.limit:
            payload = self._fetch_page(query, page)
            patents = self._extract_patents(payload)
            if not patents:
                break

            for patent in patents:
                pub_id = patent.get("publication_number")
                if not pub_id or pub_id in seen:
                    continue
                seen.add(pub_id)
                documents.append(self._parse_patent(patent))

            if len(patents) < PAGE_SIZE:
                break
            page += 1
            time.sleep(REQUEST_DELAY)

        return documents[: query.limit]

    # --- внутреннее -----------------------------------------------------

    def _fetch_page(self, query: GooglePatentsQuery, page: int) -> dict[str, Any]:
        params = {"url": self._build_url_param(query, page), "exp": ""}

        response = self._client.get(API_URL, params=params)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            # Не JSON — обычно означает, что Google отдал HTML-капчу
            # или сменил формат эндпоинта.
            raise GooglePatentsUnavailable(
                "Google Patents вернул не JSON: эндпоинт недокументирован и мог измениться"
            ) from exc

    def _build_url_param(self, query: GooglePatentsQuery, page: int) -> str:
        """Собрать строку, которую обычно формирует поисковый интерфейс.

        Все фильтры складываются в один query-string, который передаётся
        целиком в параметре url.
        """
        # Фраза в кавычках = точное совпадение. Без кавычек Google разбивает
        # её на слова и фильтр фактически перестаёт работать.
        terms = " OR ".join(f'"{t}"' for t in query.terms)

        # CPC добавляем в тот же q: второй параметр q перезатирает первый.
        if query.cpc_codes:
            terms = f"({terms}) AND ({' OR '.join(query.cpc_codes)})"

        parts = [f"q={terms}"]

        field = query.date_field
        # after = не раньше указанной даты, before = не позже.
        if query.date_from:
            parts.append(f"after={field}:{query.date_from.strftime('%Y%m%d')}")
        if query.date_to:
            parts.append(f"before={field}:{query.date_to.strftime('%Y%m%d')}")

        for country in query.countries:
            parts.append(f"country={country}")

        # parts.append("sort=new")
        # parts.append(f"num={PAGE_SIZE}")
        # if page:
        #     parts.append(f"page={page}")

        return "&".join(parts)

    @staticmethod
    def _extract_patents(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Достать патенты из вложенной структуры results.cluster[].result[].patent."""
        results = payload.get("results")
        if not isinstance(results, dict):
            return []

        patents: list[dict[str, Any]] = []
        for cluster in results.get("cluster", []):
            for item in cluster.get("result", []):
                patent = item.get("patent")
                if patent:
                    patents.append(patent)
        return patents

    def _parse_patent(self, patent: dict[str, Any]) -> Document:
        pub_id = patent.get("publication_number", "")
        title = (patent.get("title") or "").strip()
        snippet = (patent.get("snippet") or "").strip()

        # Дата приоритета — момент, когда идею заявили. Для детекции ранних
        # сигналов она информативнее даты публикации, которая отстаёт на 18 месяцев.
        published_at = self._parse_date(
            patent.get("priority_date")
            or patent.get("filing_date")
            or patent.get("publication_date")
        )

        return Document(
            provider=self.name,
            doc_id=pub_id,
            url=PATENT_URL.format(pub_id),
            title=title,
            abstract=snippet,
            published_at=published_at,
            language=detect_language(title, snippet),
            source_type=self.source_type,
            # Заявитель — это компания, поэтому кладём его в авторы:
            # отсюда берётся признак «число уникальных заявителей».
            authors=[patent["assignee"]] if patent.get("assignee") else [],
            raw={
                "assignee": patent.get("assignee"),
                "inventor": patent.get("inventor"),
                "priority_date": patent.get("priority_date"),
                "filing_date": patent.get("filing_date"),
                "grant_date": patent.get("grant_date"),
                "publication_date": patent.get("publication_date"),
                # Первые две буквы номера публикации — код страны: US, EP, RU, CN.
                "country": pub_id[:2] if len(pub_id) >= 2 else None,
            },
        )

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[datetime]:
        """Даты приходят как "2024-01-15" либо как "20240115"."""
        if not value:
            return None
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def close(self) -> None:
        self._client.close()

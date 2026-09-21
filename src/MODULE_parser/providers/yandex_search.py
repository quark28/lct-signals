"""Провайдер веб-поиска через Yandex Search API v2.

ЗАЧЕМ. Разбор датасета заказчика показал: 285 ссылок на 206 уникальных
доменов, повторяется меньше двадцати. Список сайтов вроде RSS-провайдера
(tech_media.py) покрывает только верхушку. Методологи искали по вебу —
это и делает данный провайдер: полноценный поиск по индексу Яндекса,
а не по заранее заданному списку источников.

ДВА РЕЖИМА.
  Синхронный  — ответ в том же HTTP-запросе, около секунды, дороже
                (~0.49 ₽/запрос по прайсу на конец 2026 года).
                Используется на живом демо, где важна задержка.
  Отложенный  — отправил запрос, получил operation_id, дальше опрашиваешь
                операцию, пока не появится done: true. Дешевле в ~16 раз
                (~0.03 ₽/запрос). Используется для пакетных прогонов
                и валидации на известных сигналах.
Режим выбирается параметром query.deferred.

ФОРМАТ ОТВЕТА. Сервис отдаёт XML в кодировке UTF-8, обёрнутый в Base64
в поле rawData. Схема — классический Yandex XML: results/grouping/group/doc
с полями url, domain, title, passages (сниппеты, не полный текст).

Дата публикации в ответе почти никогда не приходит: Search API — это
поиск, а не архив с метаданными. Дата достаётся позже, на извлечении
через LLM, если она вообще есть в тексте.
"""

import time
from base64 import b64decode
from typing import Any, ClassVar, Optional
from xml.etree import ElementTree as ET

import httpx
from pydantic import Field

from src.models import Document
from src.MODULE_parser.providers.base import BaseProvider, SourceQuery
from src.MODULE_parser.utils.lang import detect as detect_language

BASE_URL = "https://searchapi.api.cloud.yandex.net/v2/web"
TIMEOUT = 30.0
POLL_INTERVAL = 3.0
POLL_TIMEOUT = 600.0   # отложенный запрос может выполняться долго


class YandexSearchUnavailable(RuntimeError):
    """Сервис ответил не тем, чего мы ждём, или квота исчерпана."""


class YandexSearchQuery(SourceQuery):
    """Параметры, специфичные для веб-поиска."""

    deferred: bool = True                  # дешёвый режим по умолчанию
    search_type: str = "SEARCH_TYPE_COM"   # RU — россия, COM — весь мир
    docs_per_query: int = 20               # групп в выдаче на один термин
    # Ограничить поиск сайтами (максимум 5). Пусто — весь индекс.
    sites: list[str] = Field(default_factory=list)


class YandexSearchProvider(BaseProvider):
    name: ClassVar[str] = "yandex_search"
    source_type: ClassVar[str] = "media"
    query_model: ClassVar[type[SourceQuery]] = YandexSearchQuery

    def __init__(self, api_key: str, folder_id: str):
        if not api_key:
            raise ValueError("Yandex Search API требует ключ: YANDEX_API_KEY в .env")
        if not folder_id:
            raise ValueError("Yandex Search API требует каталог: YANDEX_FOLDER_ID в .env")

        self._folder_id = folder_id
        self._client = httpx.Client(
            timeout=TIMEOUT,
            headers={"Authorization": f"Api-Key {api_key}"},
        )

    # --- публичный метод контракта -------------------------------------

    def parse_source(self, query: YandexSearchQuery) -> list[Document]:
        documents: list[Document] = []
        seen: set[str] = set()

        # У Search API нет ИЛИ между фразами внутри одного запроса —
        # по термину на запрос, как у GitHub и Hacker News.
        per_term = max(1, query.limit // max(len(query.terms), 1))

        for term in query.terms:
            for doc_element in self._search_one(term, query):
                document = self._parse_doc(doc_element, term)
                if document is None or document.doc_id in seen:
                    continue
                seen.add(document.doc_id)
                documents.append(document)
                if len(documents) >= query.limit:
                    return documents[: query.limit]

        return documents[: query.limit]

    # --- внутреннее -----------------------------------------------------

    def _search_one(self, term: str, query: YandexSearchQuery) -> list[ET.Element]:
        body: dict[str, Any] = {
            "query": {
                "searchType": query.search_type,
                "queryText": term,
            },
            "groupSpec": {
                "groupMode": "GROUP_MODE_DEEP",
                "groupsOnPage": str(query.docs_per_query),
                "docsInGroup": "1",
            },
            "maxPassages": "3",
            "folderId": self._folder_id,
        }
        if len(query.sites) == 1:
            body["site"] = query.sites[0]

        xml_bytes = (
            self._run_deferred(body) if query.deferred else self._run_sync(body)
        )
        return self._extract_docs(xml_bytes)

    def _run_sync(self, body: dict[str, Any]) -> bytes:
        response = self._client.post(f"{BASE_URL}/search", json=body)
        response.raise_for_status()
        return self._raw_from_response(response.json())

    def _run_deferred(self, body: dict[str, Any]) -> bytes:
        started = self._client.post(f"{BASE_URL}/searchAsync", json=body)
        started.raise_for_status()
        operation_id = started.json().get("id")
        if not operation_id:
            raise YandexSearchUnavailable("нет id операции в ответе searchAsync")

        deadline = time.monotonic() + POLL_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL)
            status = self._client.get(
                f"https://operation.api.cloud.yandex.net/operations/{operation_id}"
            )
            status.raise_for_status()
            payload = status.json()
            if payload.get("done"):
                if "error" in payload:
                    raise YandexSearchUnavailable(f"операция завершилась ошибкой: {payload['error']}")
                return self._raw_from_response(payload)

        raise YandexSearchUnavailable(
            f"отложенный запрос не завершился за {POLL_TIMEOUT:.0f} с"
        )

    @staticmethod
    def _raw_from_response(payload: dict[str, Any]) -> bytes:
        raw_b64 = (payload.get("response") or payload).get("rawData")
        if not raw_b64:
            raise YandexSearchUnavailable(f"нет rawData в ответе: {payload}")
        return b64decode(raw_b64)

    def _extract_docs(self, xml_bytes: bytes) -> list[ET.Element]:
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            raise YandexSearchUnavailable(f"не удалось разобрать XML ответа: {exc}") from exc

        # Ошибки уровня API приходят тем же XML с тегом <error>.
        error = root.find(".//error")
        if error is not None:
            raise YandexSearchUnavailable(f"Search API вернул ошибку: {error.text}")

        return root.findall(".//doc")

    def _parse_doc(self, doc: ET.Element, term: str) -> Optional[Document]:
        url = self._text(doc, "url")
        if not url:
            return None

        title = self._text(doc, "title") or self._text(doc, "headline")
        passages = [self._text(p, ".") for p in doc.findall(".//passage")]
        snippet = " ".join(p for p in passages if p)

        domain = doc.attrib.get("domain") or self._text(doc, "domain")

        return Document(
            provider=self.name,
            doc_id=url,
            url=url,
            title=title or url,
            abstract=snippet,
            # Search API дат не отдаёт — их достанет извлечение через LLM,
            # если дата есть в тексте статьи.
            published_at=None,
            language=detect_language(title, snippet),
            source_type=self.source_type,
            raw={"search_term": term, "domain": domain},
        )

    @staticmethod
    def _text(element: ET.Element, path: str) -> str:
        if path == ".":
            return "".join(element.itertext()).strip()
        node = element.find(path)
        if node is None:
            return ""
        return "".join(node.itertext()).strip()

    def close(self) -> None:
        self._client.close()

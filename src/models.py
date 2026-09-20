# Общие контракты данных. Используются и парсером, и модулем признаков

from datetime import datetime
from typing import Any, Literal, Optional
import hashlib

from pydantic import BaseModel, Field

# Тип источника. Нужен для признаков «доля по типам» и для уровня доверенности.
SourceType = Literal[
    "science",     # arXiv, OpenAlex, Crossref, КиберЛенинка
    "patent",      # PatentsView, EPO, i.moscow
    "media",       # Хабр, новостные издания
    "social",      # Hacker News, Telegram
    "code",        # GitHub
    "government",  # госсайты, регуляторы
]


class Document(BaseModel):
    """Один документ, как его вернул провайдер. До извлечения полей через LLM."""

    provider: str                      # имя провайдера: "arxiv", "openalex", ...
    doc_id: str                        # id внутри провайдера, уникален в его пространстве
    url: str                           # абсолютная ссылка на первоисточник
    title: str
    abstract: str = ""                 # аннотация или первые ~2000 символов
    published_at: Optional[datetime] = None
    language: Optional[str] = None          # язык оригинала, требование ТЗ
    source_type: SourceType
    authors: list[str] = Field(default_factory=list)

    # Заполняется на этапе дедупликации: id первоисточника, если это перепечатка.
    duplicate_of: Optional[str] = None

    # Сырой ответ провайдера — на случай, если понадобится поле, которое мы не разобрали.
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        """Устойчивый ключ документа. Если у источника нет своего
        идентификатора, считаем хеш от ссылки или от заголовка с датой."""
        if self.doc_id:
            return f"{self.provider}:{self.doc_id}"
        seed = self.url or f"{self.title}|{self.published_at}"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
        return f"{self.provider}:{digest}"
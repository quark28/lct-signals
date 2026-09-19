# Контракт источника данных.

# Весь пайплайн работает только через BaseProvider и ничего не знает о том,
# как устроен конкретный API. Добавление источника не меняет ни строки в pipeline/.

from abc import ABC, abstractmethod
from datetime import date
from typing import ClassVar, Optional

from pydantic import BaseModel, Field

from src.models import Document, SourceType


class SourceQuery(BaseModel):
    """Базовые параметры запроса, общие для всех источников.

    Конкретный провайдер объявляет свой подкласс и добавляет то,
    что понимает только он: категории arXiv, коды CPC, язык репозитория.
    """

    terms: list[str] = Field(..., min_length=1)  # поисковые фразы, объединяются по ИЛИ
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    limit: int = 100                             # сколько документов максимум вернуть


class BaseProvider(ABC):
    """Базовый класс источника. Один экземпляр на провайдера."""

    name: ClassVar[str]                                   # "arxiv"
    source_type: ClassVar[SourceType]                     # "science"
    query_model: ClassVar[type[SourceQuery]] = SourceQuery

    def build_query(self, raw: dict) -> SourceQuery:
        """Собрать и провалидировать параметры запроса из плана планировщика.

        Планировщик отдаёт словарь, провайдер сам решает, как его понимать.
        Лишние ключи, которых этот источник не знает, отбрасываются.
        """
        allowed = set(self.query_model.model_fields)
        return self.query_model(**{k: v for k, v in raw.items() if k in allowed})

    @abstractmethod
    def parse_source(self, query: SourceQuery) -> list[Document]:
        """Сходить в источник и вернуть документы.

        Исключения не глушим здесь — их ловит collector, чтобы падение
        одного источника не роняло весь прогон.
        """
        raise NotImplementedError
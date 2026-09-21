# Реестр провайдеров.

# Единственное место, где перечислены конкретные источники.
# Всё остальное работает через BaseProvider и имя источника строкой.

from typing import Iterable

from src.MODULE_parser.providers.base import BaseProvider
from src.MODULE_parser.providers.arxiv import ArxivProvider
from src.MODULE_parser.providers.openalex import OpenAlexProvider
from src.MODULE_parser.providers.hackernews import HackerNewsProvider
from src.MODULE_parser.providers.github import GitHubProvider
from src.MODULE_parser.providers.crossref import CrossrefProvider
from src.MODULE_parser.providers.google_patents import GooglePatentsProvider
from src.MODULE_parser.providers.habr import HabrProvider
from src.MODULE_parser.providers.tech_media import TechMediaProvider
from src.MODULE_parser.providers.yandex_search import YandexSearchProvider

_PROVIDER_CLASSES: list[type[BaseProvider]] = [
    ArxivProvider,
    OpenAlexProvider,
    HackerNewsProvider,
    GitHubProvider,
    CrossrefProvider,
    GooglePatentsProvider,
    HabrProvider,
    TechMediaProvider,
    YandexSearchProvider
]

_REGISTRY: dict[str, type[BaseProvider]] = {cls.name: cls for cls in _PROVIDER_CLASSES}


def available() -> list[str]:
    """Имена всех зарегистрированных источников."""
    return sorted(_REGISTRY)


def get(name: str, **kwargs) -> BaseProvider:
    """Создать провайдера по имени. Бросает KeyError, если имя неизвестно."""
    if name not in _REGISTRY:
        raise KeyError(f"неизвестный источник: {name}. Доступны: {available()}")
    return _REGISTRY[name](**kwargs)


def get_many(names: Iterable[str], **kwargs) -> list[BaseProvider]:
    """Создать несколько провайдеров. Неизвестные имена пропускаются молча —
    план от LLM может назвать источник, который мы ещё не подключили."""
    return [_REGISTRY[n](**kwargs) for n in names if n in _REGISTRY]
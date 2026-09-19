"""Реестр моделей LLM.

Единственное место, которое знает про конкретных провайдеров. Состав
моделей описан в config/models.yaml, ключи берутся из настроек.

Двухуровневый роутинг: лёгкая модель на извлечении полей (сотни вызовов),
тяжёлая — на финальных карточках (пятнадцать вызовов). Это и экономия,
и выполнение требования ТЗ раскрыть, как выбирается модель под ответ.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.llms.base import BaseLLM
from src.llms.gigachat import GigaChatLLM
from src.llms.yandex import YandexLLM

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"

_PROVIDERS: dict[str, type[BaseLLM]] = {
    "gigachat": GigaChatLLM,
    "yandex": YandexLLM,
}


@lru_cache(maxsize=1)
def _config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"нет конфига моделей: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def available() -> list[str]:
    """Псевдонимы всех описанных моделей."""
    return sorted(_config().get("models", {}))


def describe(alias: str) -> dict[str, Any]:
    """Настройки модели из конфига: провайдер, разрешена ли на ЛЦТ, роль."""
    models = _config().get("models", {})
    if alias not in models:
        raise KeyError(f"неизвестная модель: {alias}. Доступны: {available()}")
    return models[alias]


def for_role(role: str) -> BaseLLM:
    """Модель под задачу: extract | name_cluster | write_card.

    Соответствие задаче задаётся в config/models.yaml в разделе roles.
    """
    roles = _config().get("roles", {})
    if role not in roles:
        raise KeyError(f"нет модели для роли {role}. Заданы: {sorted(roles)}")
    return get(roles[role])


def get(alias: str, **overrides) -> BaseLLM:
    """Создать модель по псевдониму из конфига."""
    spec = dict(describe(alias))
    provider = spec.pop("provider")

    if provider not in _PROVIDERS:
        raise KeyError(f"неизвестный провайдер: {provider}")

    # Служебные поля конфига в конструктор не передаём.
    for service_field in ("allowed_on_lct", "note"):
        spec.pop(service_field, None)

    spec.update(overrides)
    spec.update(_credentials(provider))

    return _PROVIDERS[provider](**spec)


def _credentials(provider: str) -> dict[str, str]:
    """Ключи берутся из окружения, а не из конфига: конфиг коммитится."""
    if provider == "gigachat":
        return {
            "auth_key": os.getenv("GIGACHAT_AUTH_KEY", ""),
            "scope": os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        }
    if provider == "yandex":
        return {
            "api_key": os.getenv("YANDEX_API_KEY", ""),
            "folder_id": os.getenv("YANDEX_FOLDER_ID", ""),
        }
    return {}
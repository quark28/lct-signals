"""Контракт LLM-провайдера.

Весь код обращается к моделям только через BaseLLM. Смена провайдера —
строка в настройках, изменений в пайплайне не требуется. Это же закрывает
требование заказчика: он оставляет за собой право протестировать решение
на другой модели.

Здесь же лежит журнал вызовов. ТЗ запрещает формировать выдачу на знаниях
языковой модели без подтверждённого поиска по источникам, и журнал —
единственное доказательство, что запрет соблюдён.
"""

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "system_prompts"
LOG_PATH = Path("logs/llm_calls.jsonl")

# LLM любит оборачивать JSON в ```json ... ``` даже когда просят этого не делать.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class LLMResponse(BaseModel):
    """Ответ модели плюс всё, что нужно для журнала."""

    text: str
    provider: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[int] = None


class LLMError(RuntimeError):
    """Модель не ответила или ответила не тем."""


def load_prompt(name: str) -> str:
    """Прочитать системный промпт из src/system_prompts/<name>.txt."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"нет файла промпта: {path}")
    return path.read_text(encoding="utf-8")


def log_call(
    response: LLMResponse,
    *,
    purpose: str,
    prompt_name: str,
    doc_keys: list[str],
    reason: str,
) -> None:
    """Записать вызов в журнал аудита.

    purpose    — что делали: extract | name_cluster | write_card
    prompt_name — какой системный промпт использовали (и какой его версии)
    doc_keys   — идентификаторы документов, переданных в контекст
    reason     — почему выбрана именно эта модель
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "purpose": purpose,
        "provider": response.provider,
        "model": response.model,
        "model_choice_reason": reason,
        "prompt": prompt_name,
        "doc_keys": doc_keys,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "latency_ms": response.latency_ms,
    }
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


class BaseLLM(ABC):
    """Базовый класс провайдера моделей."""

    provider: ClassVar[str]

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 2000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def complete(self, system: str, user: str) -> LLMResponse:
        """Один вызов модели. Исключения наружу — их ловит вызывающий код."""
        raise NotImplementedError

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        """Вызов с разбором ответа в pydantic-модель.

        При невалидном JSON делается одна попытка починки: модели возвращают
        её собственный ответ с указанием ошибки.
        """
        response = self.complete(system, user)
        try:
            return self._parse(response.text, schema)
        except (json.JSONDecodeError, ValidationError) as first_error:
            repair = (
                f"{user}\n\n"
                f"Предыдущий ответ не удалось разобрать: {first_error}\n"
                f"Верни ТОЛЬКО валидный JSON без пояснений и без markdown."
            )
            retry = self.complete(system, repair)
            try:
                return self._parse(retry.text, schema)
            except (json.JSONDecodeError, ValidationError) as second_error:
                raise LLMError(
                    f"{self.provider}/{self.model}: не удалось получить валидный JSON "
                    f"за две попытки: {second_error}"
                ) from second_error

    @staticmethod
    def _parse(text: str, schema: type[T]) -> T:
        cleaned = _FENCE_RE.sub("", text).strip()
        payload: Any = json.loads(cleaned)
        return schema.model_validate(payload)

    def close(self) -> None:
        """Переопределяется, если у провайдера есть что закрывать."""
        return None
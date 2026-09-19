"""Провайдер GigaChat (Сбер).

Авторизация: ключ авторизации из личного кабинета Studio — это Base64 от
пары Client ID:Client Secret. Библиотека gigachat сама меняет его на
access token и обновляет каждые ~30 минут.

Подводный камень: без корневого сертификата НУЦ Минцифры запросы падают
с ошибкой SSL, и это выглядит как "интернет не работает". Сертификат
ставится один раз в систему. Если поставить не получается, на время
отладки помогает verify_ssl_certs=False — но в сдаваемом решении так
оставлять не надо.

Модели из разрешённого списка ЛЦТ: GigaChat-2, GigaChat-2-Pro, GigaChat-2-Max.
"""

import time
from typing import ClassVar

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from src.llms.base import BaseLLM, LLMError, LLMResponse


class GigaChatLLM(BaseLLM):
    provider: ClassVar[str] = "gigachat"

    def __init__(
        self,
        auth_key: str,
        model: str = "GigaChat-2",
        scope: str = "GIGACHAT_API_PERS",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        verify_ssl_certs: bool = True,
    ):
        if not auth_key:
            raise ValueError("GigaChat требует ключ: задай GIGACHAT_AUTH_KEY в .env")

        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self._client = GigaChat(
            credentials=auth_key,
            scope=scope,
            model=model,
            verify_ssl_certs=verify_ssl_certs,
        )

    def complete(self, system: str, user: str) -> LLMResponse:
        payload = Chat(
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=system),
                Messages(role=MessagesRole.USER, content=user),
            ],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        started = time.monotonic()
        try:
            response = self._client.chat(payload)
        except Exception as exc:  # библиотека кидает свои типы исключений
            raise LLMError(f"gigachat/{self.model}: {exc}") from exc
        latency_ms = int((time.monotonic() - started) * 1000)

        if not response.choices:
            raise LLMError(f"gigachat/{self.model}: пустой ответ")

        usage = getattr(response, "usage", None)

        return LLMResponse(
            text=response.choices[0].message.content or "",
            provider=self.provider,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close:
            close()
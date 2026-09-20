"""Провайдер GigaChat (Сбер).

Авторизация: ключ авторизации из личного кабинета Studio — это Base64 от
пары Client ID:Client Secret. Библиотека gigachat сама меняет его на
access token и обновляет каждые ~30 минут.

Ограничение частоты. Бесплатный тариф отдаёт 429 Too Many Requests даже
при двух одновременных запросах, поэтому здесь два механизма:
  - общий на процесс ограничитель интервала между вызовами;
  - повтор с нарастающей задержкой именно на 429.
Ограничитель сделан общим для всех экземпляров: иначе два объекта модели
разнесут лимит по отдельности и всё равно упрутся в отказ.

Подводный камень с SSL: без корневого сертификата НУЦ Минцифры запросы
падают с ошибкой проверки сертификата. На время разработки помогает
verify_ssl_certs=False, в сдаваемом решении так оставлять не надо.

Модели из разрешённого списка ЛЦТ: GigaChat-2, GigaChat-2-Pro, GigaChat-2-Max.
"""

import time
from typing import ClassVar

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from src.llms.base import BaseLLM, LLMError, LLMResponse
from src.MODULE_parser.utils.limits import RateLimiter

# Один ограничитель на весь процесс: лимит у провайдера общий,
# а не на каждый объект модели.
_RATE_LIMITER: RateLimiter | None = None


def _limiter(min_interval: float) -> RateLimiter:
    global _RATE_LIMITER
    if _RATE_LIMITER is None or _RATE_LIMITER.min_interval < min_interval:
        _RATE_LIMITER = RateLimiter(min_interval=min_interval)
    return _RATE_LIMITER


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
        min_interval: float = 1.2,
        max_retries: int = 5,
        retry_backoff: float = 2.0,
    ):
        if not auth_key:
            raise ValueError("GigaChat требует ключ: задай GIGACHAT_AUTH_KEY в .env")

        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._rate = _limiter(min_interval)
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

        delay = self.retry_backoff
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self._rate.wait()
            started = time.monotonic()
            try:
                response = self._client.chat(payload)
            except Exception as exc:
                last_error = exc
                if not self._is_rate_limit(exc) or attempt == self.max_retries:
                    raise LLMError(f"gigachat/{self.model}: {exc}") from exc
                # Отказ по частоте: ждём дольше и пробуем снова.
                time.sleep(delay)
                delay *= 2
                continue

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

        raise LLMError(
            f"gigachat/{self.model}: отказ по частоте после {self.max_retries} попыток: {last_error}"
        )

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        """Библиотека заворачивает HTTP-ошибку в свой тип, поэтому смотрим текст."""
        text = str(exc)
        return "429" in text or "Too Many Requests" in text

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close:
            close()
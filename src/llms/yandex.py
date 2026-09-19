"""Провайдер Yandex AI Studio.

AI Studio совместим с OpenAI, поэтому используется обычный клиент openai:
base_url = https://ai.api.cloud.yandex.net/v1, project = идентификатор каталога.

Особенность Yandex: модель адресуется строкой вида
gpt://<folder_id>/yandexgpt-lite/latest — каталог зашит в имя модели.
Поэтому провайдер принимает короткое имя ("yandexgpt-lite") и сам
достраивает полный адрес.

Этим же клиентом ходят Qwen из галереи AI Studio — они в разрешённом
списке ЛЦТ и работают на том же ключе.

Точный список доступных в твоём каталоге моделей: client.models.list().
"""

import time
from typing import ClassVar

from openai import OpenAI

from src.llms.base import BaseLLM, LLMError, LLMResponse

BASE_URL = "https://ai.api.cloud.yandex.net/v1"


class YandexLLM(BaseLLM):
    provider: ClassVar[str] = "yandex"

    def __init__(
        self,
        api_key: str,
        folder_id: str,
        model: str = "yandexgpt-lite",
        version: str = "latest",
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ):
        if not api_key:
            raise ValueError("Yandex требует ключ: задай YANDEX_API_KEY в .env")
        if not folder_id:
            raise ValueError("Yandex требует каталог: задай YANDEX_FOLDER_ID в .env")

        # Если передали уже полный адрес gpt://... — не трогаем.
        full_model = (
            model if model.startswith("gpt://") else f"gpt://{folder_id}/{model}/{version}"
        )
        super().__init__(model=full_model, temperature=temperature, max_tokens=max_tokens)

        self._client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            project=folder_id,
        )

    def complete(self, system: str, user: str) -> LLMResponse:
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise LLMError(f"yandex/{self.model}: {exc}") from exc
        latency_ms = int((time.monotonic() - started) * 1000)

        if not response.choices:
            raise LLMError(f"yandex/{self.model}: пустой ответ")

        usage = response.usage

        return LLMResponse(
            text=response.choices[0].message.content or "",
            provider=self.provider,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=latency_ms,
        )

    def list_models(self) -> list[str]:
        """Какие модели реально включены в этом каталоге."""
        return [m.id for m in self._client.models.list().data]
"""Векторизация текстов локальной моделью.

Модель multilingual-e5-small: русский и английский в одном пространстве,
около 470 МБ, работает на процессоре без видеокарты. Скачивается
автоматически при первом запуске и кешируется.

Модель локальная намеренно: заказчик отдельно отметил, что предпочтительна
локальная развёртка. Решение, принимающее содержательные выводы, у нас
не в облаке — облачная LLM только извлекает поля и пишет текст.

У e5 есть особенность: тексты нужно подавать с префиксами "query: " и
"passage: ", иначе качество заметно хуже. Для симметричных задач
(сравнение текста с текстом) используется "query: " с обеих сторон.

Эмбеддинги считаются батчами: это загрузка процессора, потоки тут
не помогают.
"""

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import Document

MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 64
MAX_CONTEXT_CHARS = 300


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Модель грузится при первом обращении: импорт модуля не должен
        занимать полминуты."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Векторизовать тексты. Результат нормирован по длине, поэтому
        скалярное произведение двух векторов равно косинусной близости."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        prefixed = [f"query: {t}" for t in texts]
        vectors = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_documents(self, documents: list[Document]) -> np.ndarray:
        """Эмбеддинги содержимого документов. Используются для поиска
        перепечаток: сравнивается сам материал."""
        texts = [
            f"{d.title}. {(d.abstract or '')[:MAX_CONTEXT_CHARS]}" for d in documents
        ]
        return self.encode(texts)

    def encode_technology_names(
        self, names: list[str], contexts: Optional[list[str]] = None
    ) -> np.ndarray:
        """Эмбеддинги названий технологий. Используются для кластеризации.

        Векторизуется именно название, а не резюме документа: иначе близость
        начинает определяться типом издания и стилистикой, а не предметом.
        Короткий контекст добавляется, чтобы различить омонимичные названия.
        """
        if contexts is None:
            return self.encode(names)

        if len(names) != len(contexts):
            raise ValueError("число названий и контекстов не совпадает")

        texts = [
            f"{name}. {(context or '')[:MAX_CONTEXT_CHARS]}"
            for name, context in zip(names, contexts)
        ]
        return self.encode(texts)

"""Дедупликация: схлопывание перепечаток одного материала.

Выполняется ДО подсчёта признаков. Один пресс-релиз, перепечатанный пятью
изданиями, иначе даст ложный всплеск — именно так маркетинговый хайп
маскируется под сигнал.

Дубликаты не удаляются: им проставляется duplicate_of с идентификатором
первоисточника. Ссылки нужны, чтобы показать в интерфейсе подтверждение,
а доля дублей — это признак модели-оценщика.

Два прохода:
  1. дешёвый  — по нормализованному заголовку, без вызовов LLM и без эмбеддингов
  2. дорогой  — по эмбеддингам, ловит переписанные заголовки
"""

import re
import unicodedata
from typing import Optional

import numpy as np

from src.models import Document

# Порог косинусной близости, выше которого документы считаются одним материалом.
# Ставится высоко: мы ищем перепечатки, а не похожие темы.
NEAR_DUPLICATE_THRESHOLD = 0.93

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Привести заголовок к виду, устойчивому к мелким расхождениям изданий."""
    text = unicodedata.normalize("NFKD", title).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def dedup_by_title(documents: list[Document]) -> list[Document]:
    """Первый проход. Первым встреченным считается первоисточник:
    документы предварительно сортируются по дате, чтобы им оказался
    самый ранний."""
    ordered = sorted(
        documents,
        key=lambda d: (d.published_at is None, d.published_at),
    )

    originals: dict[str, str] = {}  # нормализованный заголовок -> ключ первоисточника
    for document in ordered:
        key = normalize_title(document.title)
        if not key:
            continue
        if key in originals:
            document.duplicate_of = originals[key]
        else:
            originals[key] = document.key

    return documents


def dedup_by_embedding(
    documents: list[Document],
    embeddings: np.ndarray,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[Document]:
    """Второй проход. embeddings должны соответствовать documents по порядку
    и быть нормированными по длине (тогда скалярное произведение = косинус).

    Ловит случай, когда издание переписало заголовок своими словами.
    """
    if len(documents) != len(embeddings):
        raise ValueError("число документов и эмбеддингов не совпадает")
    if len(documents) < 2:
        return documents

    order = sorted(
        range(len(documents)),
        key=lambda i: (
            documents[i].published_at is None,
            documents[i].published_at,
        ),
    )

    similarity = embeddings @ embeddings.T

    for position, index in enumerate(order):
        document = documents[index]
        if document.duplicate_of:
            continue  # уже помечен первым проходом

        # Сравниваем только с более ранними: первоисточником считается старший.
        for earlier_index in order[:position]:
            earlier = documents[earlier_index]
            if earlier.duplicate_of:
                continue
            if similarity[index, earlier_index] >= threshold:
                document.duplicate_of = earlier.key
                break

    return documents


def originals(documents: list[Document]) -> list[Document]:
    """Только первоисточники. По ним считаются все признаки кластера."""
    return [d for d in documents if not d.duplicate_of]


def duplicate_ratio(documents: list[Document]) -> float:
    """Доля перепечаток. Признак модели-оценщика: высокая доля при малом
    числе уникальных источников — характерный профиль хайпа."""
    if not documents:
        return 0.0
    duplicates = sum(1 for d in documents if d.duplicate_of)
    return duplicates / len(documents)

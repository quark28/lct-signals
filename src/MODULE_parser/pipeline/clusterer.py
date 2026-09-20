"""Кластеризация: группировка документов по технологиям.

Единица анализа системы — технология, а не документ. Признаки слабого
сигнала (всплеск, число компаний, доля соцсетей, давность первого
упоминания) существуют только на уровне группы документов.

Метод: агломеративная кластеризация с порогом расстояния. Выбран потому,
что оставляет одиночные документы отдельными кластерами. У слабого сигнала
документов мало по определению, и алгоритм, помечающий редкие точки как
шум, выбросит именно то, что мы ищем. Минимальный размер кластера
не задаётся намеренно.

Ограничение метода — матрица попарных расстояний, квадрат по памяти.
На сотнях документов несущественно.
"""

from collections import Counter
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering

from src.models import Document

# Порог косинусного расстояния (1 - косинусная близость), ниже которого
# названия считаются одной технологией. Подбирается эмпирически на
# известных сигналах из датасета заказчика.
DISTANCE_THRESHOLD = 0.15


class Cluster(BaseModel):
    """Технология: группа документов плюс варианты её названия."""

    cluster_id: int
    document_keys: list[str] = Field(default_factory=list)
    name_variants: list[str] = Field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.document_keys)

    def most_common_name(self) -> str:
        """Самое частое название. Используется для кластеров, которым
        не писали карточку — то есть для всех, кроме топ-15."""
        if not self.name_variants:
            return ""
        return Counter(self.name_variants).most_common(1)[0][0]

GENERIC_NAMES = {
    "искусственный интеллект", "машинное обучение", "глубокое обучение",
    "нейронные сети", "большие данные", "блокчейн", "криптовалюты",
    "облачные вычисления", "интернет вещей", "кибербезопасность",
    "финтех", "цифровизация", "автоматизация", "аналитика данных",
    "artificial intelligence", "machine learning", "deep learning",
    "blockchain", "big data", "cloud computing", "cybersecurity",
    "цифровые технологии", "финансовые технологии", "информационные технологии",
    "программное обеспечение", "мобильные приложения", "веб-технологии",
}


def is_generic(name: str) -> bool:
    return name.strip().lower() in GENERIC_NAMES

def cluster_documents(
    document_keys: list[str],
    names: list[str],
    embeddings: np.ndarray,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> list[Cluster]:
    """Сгруппировать документы по названиям технологий.

    Все три списка соответствуют друг другу по порядку. Один документ может
    упоминать несколько технологий — тогда он встречается в списках
    несколько раз с разными названиями и попадёт в несколько кластеров.
    """
    if len(document_keys) != len(names) or len(names) != len(embeddings):
        raise ValueError("списки ключей, названий и эмбеддингов не совпадают по длине")

    if len(names) == 0:
        return []

    if len(names) == 1:
        return [
            Cluster(cluster_id=0, document_keys=[document_keys[0]], name_variants=[names[0]])
        ]

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        # complete: расстояние между кластерами = расстояние между самыми
        # далёкими точками. Не даёт цепочкам слипаться в один ком,
        # как average.
        linkage="complete",
    )
    labels = model.fit_predict(embeddings)

    clusters: dict[int, Cluster] = {}
    for label, key, name in zip(labels, document_keys, names):
        label = int(label)
        if label not in clusters:
            clusters[label] = Cluster(cluster_id=label)
        # Один документ не должен считаться в кластере дважды.
        if key not in clusters[label].document_keys:
            clusters[label].document_keys.append(key)
        clusters[label].name_variants.append(name)

    return sorted(clusters.values(), key=lambda c: c.size, reverse=True)


def flatten_technologies(
    extracted: dict[str, "object"],
) -> tuple[list[str], list[str], list[str]]:
    """Развернуть результат извлечения в три параллельных списка.

    Вход: {ключ документа: ExtractedFields}.
    Выход: ключи документов, названия технологий, контексты.

    Документы, не описывающие технологию, отбрасываются здесь: иначе
    обзоры рынка и новости про кадры попадут в кластеры и испортят
    статистику.
    """
    keys: list[str] = []
    names: list[str] = []
    contexts: list[str] = []

    for document_key, fields in extracted.items():
        if not getattr(fields, "is_technology", False):
            continue
        for technology in getattr(fields, "technologies", []):
            name = technology.name_ru or technology.name_original
            if is_generic(name):
                continue
            keys.append(document_key)
            names.append(name)
            contexts.append(getattr(fields, "summary_ru", ""))

    return keys, names, contexts

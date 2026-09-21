"""Кластеризация: группировка документов по технологиям.

Единица анализа системы — технология, а не документ. Признаки слабого
сигнала (всплеск, число компаний, доля соцсетей, давность первого
упоминания) существуют только на уровне группы документов.

Метод: агломеративная кластеризация с порогом расстояния. Выбран потому,
что оставляет одиночные документы отдельными кластерами. У слабого сигнала
документов мало по определению, и алгоритм, помечающий редкие точки как
шум, выбросит именно то, что мы ищем. Минимальный размер кластера
не задаётся намеренно.

Связь complete, а не average: average даёт цепочки, когда A близко к B,
B к C, и всё сливается в один ком, хотя A и C далеки.

Ограничение метода — матрица попарных расстояний, квадрат по памяти.
На сотнях документов несущественно.
"""

from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering

# Порог косинусного расстояния (1 - косинусная близость), ниже которого
# названия считаются одной технологией. Подбирается эмпирически на
# известных сигналах из датасета заказчика.
DISTANCE_THRESHOLD = 0.15

# Насколько строже режем при повторной кластеризации одноимённых групп.
RESPLIT_FACTOR = 1.0

# Общие области, которые модель иногда возвращает как технологии вопреки
# промпту. Слабым сигналом они быть не могут: про них знают все.
GENERIC_NAMES = {
    "искусственный интеллект", "машинное обучение", "глубокое обучение",
    "нейронные сети", "большие данные", "блокчейн", "криптовалюты",
    "облачные вычисления", "интернет вещей", "кибербезопасность",
    "финтех", "цифровизация", "автоматизация", "аналитика данных",
    "цифровые технологии", "финансовые технологии", "информационные технологии",
    "программное обеспечение", "мобильные приложения", "веб-технологии",
    "artificial intelligence", "machine learning", "deep learning",
    "blockchain", "big data", "cloud computing", "cybersecurity",
}

def is_generic(name: str) -> bool:
    return name.strip().lower() in GENERIC_NAMES


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


def _run_clustering(
    document_keys: list[str],
    names: list[str],
    embeddings: np.ndarray,
    distance_threshold: float,
    id_offset: int = 0,
) -> list[Cluster]:
    """Один проход агломеративной кластеризации."""
    if len(names) == 0:
        return []

    if len(names) == 1:
        return [
            Cluster(
                cluster_id=id_offset,
                document_keys=[document_keys[0]],
                name_variants=[names[0]],
            )
        ]

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="complete",
    )
    labels = model.fit_predict(embeddings)

    clusters: dict[int, Cluster] = {}
    for label, key, name in zip(labels, document_keys, names):
        label = int(label)
        if label not in clusters:
            clusters[label] = Cluster(cluster_id=id_offset + label)
        if key not in clusters[label].document_keys:
            clusters[label].document_keys.append(key)
        clusters[label].name_variants.append(name)

    return list(clusters.values())


def resplit_same_named(
    clusters: list[Cluster],
    embeddings_by_index: dict[str, np.ndarray],
    distance_threshold: float,
) -> list[Cluster]:
    """Разобраться с кластерами, у которых совпало каноническое название.

    Одинаковое название у двух кластеров означает, что порог развёл
    по разным группам то, что человек считает одной темой, либо наоборот —
    склеил внутри каждой группы разное. Простое объединение скрыло бы
    вторую возможность.

    Поэтому одноимённые кластеры убираются из выдачи, их упоминания
    сливаются в общую кучу и кластеризуются заново с более строгим
    порогом. Результат может дать как один кластер, так и несколько
    честно различных.
    """
    by_name: dict[str, list[Cluster]] = defaultdict(list)
    for cluster in clusters:
        by_name[cluster.most_common_name()].append(cluster)

    result: list[Cluster] = []
    next_id = max((c.cluster_id for c in clusters), default=0) + 1

    for name, group in by_name.items():
        if len(group) == 1:
            result.append(group[0])
            continue

        # Собираем упоминания всех одноимённых кластеров в одну кучу.
        keys: list[str] = []
        names: list[str] = []
        vectors: list[np.ndarray] = []
        for cluster in group:
            for index, variant in enumerate(cluster.name_variants):
                key = (
                    cluster.document_keys[index]
                    if index < len(cluster.document_keys)
                    else cluster.document_keys[-1]
                )
                vector = embeddings_by_index.get(f"{key}|{variant}")
                if vector is None:
                    continue
                keys.append(key)
                names.append(variant)
                vectors.append(vector)

        if not vectors:
            result.extend(group)
            continue

        resplit = _run_clustering(
            keys,
            names,
            np.vstack(vectors),
            distance_threshold * RESPLIT_FACTOR,
            id_offset=next_id,
        )
        next_id += len(resplit) + 1
        result.extend(resplit)

    return result


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

    clusters = _run_clustering(document_keys, names, embeddings, distance_threshold)

    # Индекс векторов для повторной кластеризации одноимённых групп.
    embeddings_by_index = {
        f"{key}|{name}": embeddings[i]
        for i, (key, name) in enumerate(zip(document_keys, names))
    }
    clusters = resplit_same_named(clusters, embeddings_by_index, distance_threshold)

    return sorted(clusters, key=lambda c: c.size, reverse=True)


def flatten_technologies(extracted: dict[str, object]) -> tuple[list[str], list[str], list[str]]:
    """Развернуть результат извлечения в три параллельных списка.

    Вход: {ключ документа: ExtractedFields}.
    Выход: ключи документов, названия технологий, контексты.

    Документы, не описывающие технологию, и общие области отбрасываются
    здесь: иначе обзоры рынка и слово "блокчейн" попадут в кластеры
    и испортят статистику.
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

# Косинусная близость названия кластера к самому запросу, выше которой
# кластер считается зонтичным — это само направление, а не технология
# внутри него. Калибруется по выводу: смотри, что отбрасывается.
UMBRELLA_SIMILARITY = 0.85


def drop_umbrella(
    clusters: list[Cluster],
    name_vectors: np.ndarray,
    query_vector: np.ndarray,
    threshold: float = UMBRELLA_SIMILARITY,
) -> tuple[list[Cluster], list[tuple[Cluster, float]]]:
    """Отделить кластеры, которые по смыслу совпадают с запросом.

    На запрос «Edge» модель называет технологией само «Edge computing»
    в пяти переводах. Это зонтичная тема, внутри которой и ищутся
    слабые сигналы, — сама она сигналом быть не может.
    """
    kept: list[Cluster] = []
    dropped: list[tuple[Cluster, float]] = []
    for cluster, vector in zip(clusters, name_vectors):
        similarity = float(vector @ query_vector)
        if similarity >= threshold:
            dropped.append((cluster, similarity))
        else:
            kept.append(cluster)
    return kept, dropped
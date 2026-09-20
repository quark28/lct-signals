"""Запись и чтение данных. Единственное место, где живёт SQL.

Пайплайн работает через эти функции и не знает, что под ними PostgreSQL.

В базе лежат только СЫРЫЕ агрегаты кластера. Производные признаки модели
считаются на лету в src/features/ — формулы будут меняться, а
пересчитывать из-за этого всю базу не нужно.
"""

from datetime import date
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np
from psycopg.types.json import Jsonb

from src.db import pool
from src.models import Document

# Сырые поля кластера. Добавление поля = строка в схеме и строка здесь.
CLUSTER_SCALAR_COLUMNS = [
    "name_original", "name_source_key", "aliases",
    "status", "confidence", "rank_position", "exclusion_reason",
    "doc_count", "mention_count", "duplicate_count",
    "unique_domains", "unique_companies", "company_mentions",
    "first_published_at", "last_published_at", "mean_published_at",
    "patent_count", "patent_applicants", "patent_countries",
    "patent_top_applicant_count", "patent_foreign_count",
    "funding_events", "max_round_stage", "max_round_amount", "total_round_amount",
    "readiness_level", "deployment_markers",
    "github_repos", "github_stars_sum", "github_stars_max",
]

# Поля-словари: оборачиваются в Jsonb при записи.
CLUSTER_JSONB_COLUMNS = [
    "provider_counts", "type_counts", "type_duplicate_counts",
    "timeline", "patent_country_counts", "round_stage_counts",
    "stage_counts", "trust_counts", "card",
]


# --------------------------------------------------------------- queries


def create_query(
    user_query: str,
    plan: dict[str, Any],
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> int:
    """Завести прогон. Возвращает его идентификатор."""
    with pool.cursor() as cur:
        cur.execute(
            """
            INSERT INTO queries (user_query, plan, date_from, date_to)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (user_query, Jsonb(plan), date_from, date_to),
        )
        return cur.fetchone()["id"]


def finish_query(
    query_id: int,
    *,
    sources_processed: int,
    documents_collected: int,
    candidates_found: int,
    signals_confident: int,
    run_totals: dict[str, Any],
    source_report: list[dict[str, Any]],
) -> None:
    """Закрыть прогон. run_totals нужны для нормировки признаков:
    часть из них считается относительно объёма всего прогона."""
    with pool.cursor() as cur:
        cur.execute(
            """
            UPDATE queries
               SET finished_at = now(),
                   sources_processed = %s,
                   documents_collected = %s,
                   candidates_found = %s,
                   signals_confident = %s,
                   run_totals = %s,
                   source_report = %s
             WHERE id = %s
            """,
            (
                sources_processed,
                documents_collected,
                candidates_found,
                signals_confident,
                Jsonb(run_totals),
                Jsonb(source_report),
                query_id,
            ),
        )


def get_query(query_id: int) -> Optional[dict[str, Any]]:
    return pool.fetch_one("SELECT * FROM queries WHERE id = %s", (query_id,))


# ------------------------------------------------------------- documents


def save_documents(
    documents: list[Document],
    trust_by_provider: dict[str, str],
    extracted: Optional[dict[str, Any]] = None,
    embeddings: Optional[np.ndarray] = None,
) -> int:
    """Сохранить документы. Известные обновляются, новые добавляются.

    Вставка в два прохода: сначала все строки без ссылки на первоисточник,
    потом проставляются ссылки. Иначе перепечатка может вставиться раньше
    своего первоисточника и нарушить внешний ключ.
    """
    if not documents:
        return 0

    extracted = extracted or {}
    rows = []
    duplicate_links: list[tuple[str, str]] = []

    for index, document in enumerate(documents):
        fields = extracted.get(document.key)
        vector = (
            embeddings[index].tolist()
            if embeddings is not None and index < len(embeddings)
            else None
        )
        if document.duplicate_of:
            duplicate_links.append((document.duplicate_of, document.key))

        rows.append(
            (
                document.key,
                document.provider,
                document.source_type,
                trust_by_provider.get(document.provider, "medium"),
                document.url,
                urlparse(document.url).netloc or None,
                document.title,
                document.abstract,
                document.published_at,
                document.language,
                list(document.authors),
                Jsonb(fields.model_dump()) if fields is not None else None,
                Jsonb(document.raw),
                vector,
            )
        )

    with pool.cursor() as cur:
        # Первый проход: документы без ссылок на первоисточник.
        cur.executemany(
            """
            INSERT INTO documents (
                doc_key, provider, source_type, trust, url, domain,
                title, abstract, published_at, language, authors,
                extracted, raw, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_key) DO UPDATE SET
                extracted = COALESCE(EXCLUDED.extracted, documents.extracted),
                embedding = COALESCE(EXCLUDED.embedding, documents.embedding)
            """,
            rows,
        )

        # Второй проход: ссылки на первоисточники. К этому моменту
        # все строки уже существуют. Ссылки на документы, которых нет
        # в базе (первоисточник отфильтровался раньше), пропускаются.
        if duplicate_links:
            cur.executemany(
                """
                UPDATE documents
                   SET duplicate_of = %s
                 WHERE doc_key = %s
                   AND EXISTS (SELECT 1 FROM documents o WHERE o.doc_key = %s)
                """,
                [(original, dup, original) for original, dup in duplicate_links],
            )

    return len(rows)


def get_extracted(doc_keys: list[str]) -> dict[str, dict[str, Any]]:
    """Уже разобранные документы. Это кеш извлечения: если факты есть
    в базе, второй раз звать модель незачем."""
    if not doc_keys:
        return {}

    rows = pool.fetch_all(
        """
        SELECT doc_key, extracted
          FROM documents
         WHERE doc_key = ANY(%s) AND extracted IS NOT NULL
        """,
        (doc_keys,),
    )
    return {row["doc_key"]: row["extracted"] for row in rows}


def get_documents(doc_keys: list[str]) -> list[dict[str, Any]]:
    """Документы целиком. Нужны модулю признаков для агрегации."""
    if not doc_keys:
        return []
    return pool.fetch_all(
        "SELECT * FROM documents WHERE doc_key = ANY(%s)", (doc_keys,)
    )


def find_near_duplicates(
    embedding: list[float], threshold: float = 0.93, limit: int = 5
) -> list[dict[str, Any]]:
    """Похожие документы по эмбеддингу. Оператор <=> у pgvector —
    косинусное расстояние, поэтому близость = 1 минус расстояние."""
    return pool.fetch_all(
        """
        SELECT doc_key, title, 1 - (embedding <=> %s::vector) AS similarity
          FROM documents
         WHERE embedding IS NOT NULL
           AND 1 - (embedding <=> %s::vector) >= %s
         ORDER BY embedding <=> %s::vector
         LIMIT %s
        """,
        (embedding, embedding, threshold, embedding, limit),
    )


# -------------------------------------------------------------- clusters


def save_clusters(query_id: int, clusters: list[dict[str, Any]]) -> list[int]:
    """Сохранить кластеры прогона вместе с сырыми агрегатами.

    На вход — список словарей: name_ru, document_keys, name_variants
    и любые поля из CLUSTER_SCALAR_COLUMNS / CLUSTER_JSONB_COLUMNS.
    Отсутствующие поля просто не записываются.
    """
    if not clusters:
        return []

    ids: list[int] = []

    with pool.cursor() as cur:
        for cluster in clusters:
            columns = ["query_id", "name_ru"]
            values: list[Any] = [query_id, cluster.get("name_ru", "")]

            for column in CLUSTER_SCALAR_COLUMNS:
                if cluster.get(column) is not None:
                    columns.append(column)
                    values.append(cluster[column])

            for column in CLUSTER_JSONB_COLUMNS:
                if cluster.get(column) is not None:
                    columns.append(column)
                    values.append(Jsonb(cluster[column]))

            placeholders = ", ".join(["%s"] * len(values))
            cur.execute(
                f"INSERT INTO clusters ({', '.join(columns)}) "
                f"VALUES ({placeholders}) RETURNING id",
                tuple(values),
            )
            cluster_id = cur.fetchone()["id"]
            ids.append(cluster_id)

            keys = cluster.get("document_keys", [])
            variants = cluster.get("name_variants", [])
            if keys:
                cur.executemany(
                    """
                    INSERT INTO cluster_documents (cluster_id, doc_key, name_variant)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    [
                        (cluster_id, key, variants[i] if i < len(variants) else None)
                        for i, key in enumerate(keys)
                    ],
                )

    return ids


def update_cluster(cluster_id: int, **fields: Any) -> None:
    """Обновить поля кластера. Используется оценщиком: он проставляет
    статус, уверенность и место в выдаче после подсчёта признаков."""
    if not fields:
        return

    assignments: list[str] = []
    values: list[Any] = []
    for column, value in fields.items():
        if column in CLUSTER_JSONB_COLUMNS:
            value = Jsonb(value)
        elif column not in CLUSTER_SCALAR_COLUMNS and column != "name_ru":
            continue  # защита от опечатки в имени колонки
        assignments.append(f"{column} = %s")
        values.append(value)

    if not assignments:
        return

    values.append(cluster_id)
    pool.execute(
        f"UPDATE clusters SET {', '.join(assignments)} WHERE id = %s", tuple(values)
    )


def get_clusters(query_id: int, status: Optional[str] = None) -> list[dict[str, Any]]:
    """Кластеры прогона. Без статуса — все, включая отсеянные."""
    if status:
        return pool.fetch_all(
            "SELECT * FROM clusters WHERE query_id = %s AND status = %s "
            "ORDER BY rank_position NULLS LAST, confidence DESC NULLS LAST",
            (query_id, status),
        )
    return pool.fetch_all(
        "SELECT * FROM clusters WHERE query_id = %s "
        "ORDER BY rank_position NULLS LAST, confidence DESC NULLS LAST",
        (query_id,),
    )


def get_cluster_documents(cluster_id: int) -> list[dict[str, Any]]:
    """Документы кластера с полями для карточки источника:
    наименование, ссылка, дата, тип, язык, уровень доверенности."""
    return pool.fetch_all(
        """
        SELECT d.doc_key, d.title, d.url, d.domain, d.published_at,
               d.source_type, d.language, d.trust, d.provider,
               d.duplicate_of, d.machine_translated, cd.name_variant
          FROM cluster_documents cd
          JOIN documents d ON d.doc_key = cd.doc_key
         WHERE cd.cluster_id = %s
         ORDER BY d.published_at DESC NULLS LAST
        """,
        (cluster_id,),
    )


def save_card(cluster_id: int, card: dict[str, Any]) -> None:
    """Сохранить карточку от LLM. Карточка пишется лениво, по клику,
    и второй раз не генерируется."""
    pool.execute(
        "UPDATE clusters SET card = %s WHERE id = %s", (Jsonb(card), cluster_id)
    )
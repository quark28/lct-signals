"""Точка входа парсера.

Запуск из корня репозитория:

    python -m src.MODULE_parser.run_parse "технологии в финтехе"
    python -m src.MODULE_parser.run_parse "квантовые сенсоры" --from 2024-01-01 --limit-sources arxiv,openalex

Это тонкая оболочка: вся логика в pipeline/, и она ничего не знает о том,
откуда пришли параметры. Когда появится веб-интерфейс, он позовёт те же
функции, а этот файл останется для отладки и пакетных прогонов.

Результат сохраняется в базу: документы переиспользуются между прогонами,
кластеры привязаны к конкретному запросу.
"""

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Any, Optional

import yaml

from src.db import pool, repo
from src.llms import registry as llm_registry
from src.models import Document
from src.settings import get_settings
from src.MODULE_parser.pipeline import clusterer, collector, dedup, extractor, planner
from src.MODULE_parser.pipeline.embedder import Embedder
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сбор документов по технологическому направлению"
    )
    parser.add_argument("query", help="запрос на естественном языке")
    parser.add_argument("--from", dest="date_from", help="нижняя граница даты, ГГГГ-ММ-ДД")
    parser.add_argument("--to", dest="date_to", help="верхняя граница даты, ГГГГ-ММ-ДД")
    parser.add_argument(
        "--limit-sources", dest="only",
        help="опросить только эти источники через запятую, для отладки",
    )
    parser.add_argument("--no-cache", action="store_true", help="игнорировать кеш извлечения")
    parser.add_argument("--no-save", action="store_true", help="не писать в базу")
    parser.add_argument("--parallel", type=int, help="одновременных запросов к LLM")
    return parser.parse_args()


def to_date(value: Optional[str]) -> Optional[date]:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date() if value else None


def trust_map() -> dict[str, str]:
    """Уровень доверенности по провайдеру — из конфига источников."""
    settings = get_settings()
    config = yaml.safe_load(settings.sources_config.read_text(encoding="utf-8"))
    return {
        name: spec.get("trust", "medium")
        for name, spec in config.get("sources", {}).items()
    }


def quarter(moment: datetime) -> str:
    return f"{moment.year}Q{(moment.month - 1) // 3 + 1}"


def build_cluster_rows(
    clusters: list[clusterer.Cluster],
    documents_by_key: dict[str, Document],
    extracted: dict[str, Any],
) -> list[dict[str, Any]]:
    """Собрать сырые агрегаты по каждому кластеру.

    Здесь считаются ТОЛЬКО прямые подсчёты: счётчики, даты, временные ряды.
    Производные признаки модели вычисляются в модуле признаков второго
    спринта, из этих самых величин.
    """
    rows: list[dict[str, Any]] = []

    for cluster in clusters:
        documents = [documents_by_key[k] for k in cluster.document_keys if k in documents_by_key]
        if not documents:
            continue

        provider_counts: Counter[str] = Counter()
        type_counts: Counter[str] = Counter()
        type_duplicates: Counter[str] = Counter()
        trust_counts: Counter[str] = Counter()
        country_counts: Counter[str] = Counter()
        stage_counts: Counter[str] = Counter()
        round_stages: Counter[str] = Counter()
        timeline: dict[str, Counter[str]] = defaultdict(Counter)

        domains: set[str] = set()
        companies: set[str] = set()
        applicants: set[str] = set()
        company_mentions = 0
        duplicate_count = 0
        deployment_markers = 0
        funding_events = 0
        patent_count = 0
        patent_foreign = 0
        github_repos = 0
        github_stars_sum = 0
        github_stars_max = 0
        round_amounts: list[float] = []
        dates: list[datetime] = []

        for document in documents:
            provider_counts[document.provider] += 1
            type_counts[document.source_type] += 1

            if document.duplicate_of:
                duplicate_count += 1
                type_duplicates[document.source_type] += 1

            if document.url:
                from urllib.parse import urlparse
                host = urlparse(document.url).netloc
                if host:
                    domains.add(host)

            if document.published_at:
                dates.append(document.published_at)
                label = quarter(document.published_at)
                timeline["all"][label] += 1
                timeline[document.source_type][label] += 1

            if document.source_type == "patent":
                patent_count += 1
                country = (document.raw or {}).get("country")
                if country:
                    country_counts[country] += 1
                assignee = (document.raw or {}).get("assignee")
                if assignee:
                    applicants.add(assignee)

            if document.source_type == "code":
                github_repos += 1
                stars = (document.raw or {}).get("stars") or 0
                github_stars_sum += stars
                github_stars_max = max(github_stars_max, stars)

            fields = extracted.get(document.key)
            if fields:
                for company in getattr(fields, "companies", []):
                    companies.add(company)
                    company_mentions += 1
                stage = getattr(fields, "stage", None)
                if stage:
                    stage_counts[stage] += 1
                deployment_markers += len(getattr(fields, "deployment_markers", []))
                for item in getattr(fields, "funding", []):
                    funding_events += 1
                    if item.round_stage:
                        round_stages[item.round_stage] += 1
                    if item.amount_usd:
                        round_amounts.append(float(item.amount_usd))

        for document in documents:
            trust_counts[document.provider] += 0  # заполняется ниже по карте доверенности

        stage_order = {"концепция": 1, "прототип": 2, "пилот": 3, "раннее_внедрение": 4}
        round_order = {"seed": 1, "series_a": 2, "series_b": 3, "series_c_plus": 4}

        readiness = max(
            (stage_order.get(s, 0) for s in stage_counts), default=None
        ) or None
        max_round = max((round_order.get(s, 0) for s in round_stages), default=None) or None

        # Патенты, где страна публикации не совпадает с преобладающей:
        # грубая оценка доли зарубежных заявителей.
        top_country = country_counts.most_common(1)[0][1] if country_counts else 0
        patent_foreign = patent_count - top_country

        rows.append({
            "name_ru": cluster.most_common_name(),
            "document_keys": cluster.document_keys,
            "name_variants": cluster.name_variants,

            "doc_count": len(documents),
            "mention_count": len(cluster.name_variants),
            "duplicate_count": duplicate_count,
            "provider_counts": dict(provider_counts),
            "type_counts": dict(type_counts),
            "type_duplicate_counts": dict(type_duplicates),
            "unique_domains": len(domains),
            "unique_companies": len(companies),
            "company_mentions": company_mentions,

            "first_published_at": min(dates) if dates else None,
            "last_published_at": max(dates) if dates else None,
            "mean_published_at": (
                datetime.fromtimestamp(
                    sum(d.timestamp() for d in dates) / len(dates), tz=dates[0].tzinfo
                )
                if dates else None
            ),
            "timeline": {k: dict(v) for k, v in timeline.items()},

            "patent_count": patent_count,
            "patent_applicants": len(applicants),
            "patent_countries": len(country_counts),
            "patent_top_applicant_count": top_country,
            "patent_foreign_count": max(patent_foreign, 0),
            "patent_country_counts": dict(country_counts),

            "funding_events": funding_events,
            "max_round_stage": max_round,
            "max_round_amount": max(round_amounts) if round_amounts else None,
            "total_round_amount": sum(round_amounts) if round_amounts else None,
            "round_stage_counts": dict(round_stages),

            "readiness_level": readiness,
            "stage_counts": dict(stage_counts),
            "deployment_markers": deployment_markers,

            "github_repos": github_repos,
            "github_stars_sum": github_stars_sum,
            "github_stars_max": github_stars_max,
        })

    return rows


def main() -> None:
    settings = get_settings()

    problems = settings.validate()
    if problems:
        print("Проблемы с настройками:")
        for problem in problems:
            print(f"  - {problem}")

    args = parse_args()
    save = not args.no_save

    if save and not pool.healthcheck():
        print("База недоступна. Запусти с --no-save или почини POSTGRES_DSN.")
        return

    # 1. План запросов
    print(f"Запрос: {args.query}")
    plan = planner.build_plan(llm_registry.for_role("plan"), args.query)
    print(f"  фраз ru/en: {len(plan.terms_ru)}/{len(plan.terms_en)}")
    for area in plan.adjacent_areas:
        print(f"  смежная область: {area.area} — {area.why}")

    date_from, date_to = to_date(args.date_from), to_date(args.date_to)
    query_id = (
        repo.create_query(args.query, plan.model_dump(mode="json"), date_from, date_to)
        if save else None
    )

    # 2. Сбор
    result = collector.collect(
        plan,
        date_from=date_from,
        date_to=date_to,
        credentials=settings.credentials.as_kwargs(),
        only=args.only.split(",") if args.only else None,
    )
    print(f"\nСобрано документов: {len(result.documents)}")
    for source in result.per_source:
        status = f"{source.documents}" if source.ok else f"ОШИБКА: {source.error}"
        print(f"  {source.source:16} [{source.trust:6}] {status}")

    if not result.documents:
        print("\nНичего не найдено — дальше идти не с чем.")
        return

    # 3. Дедупликация по заголовкам — до извлечения, экономит вызовы модели
    documents = dedup.dedup_by_title(result.documents)
    originals = dedup.originals(documents)
    print(f"\nПосле дедупа по заголовкам: {len(originals)} первоисточников")

    # 4. Извлечение фактов
    cache = None if args.no_cache else extractor.ExtractionCache()
    extracted, stats = extractor.extract_many(
        llm_registry.for_role("extract"),
        originals,
        cache=cache,
        max_parallel=args.parallel or settings.pipeline.llm_parallel,
    )
    print(
        f"Извлечение: {stats.extracted} новых, {stats.from_cache} из кеша, "
        f"потеряно {stats.lost} (ошибок LLM: {stats.llm_errors}, "
        f"ошибок разбора: {stats.parse_errors})"
    )
    for sample in stats.samples:
        print(f"    {sample}")

    if not extracted:
        print("\nНи одного документа не разобрано — дальше идти не с чем.")
        return

    # 5. Эмбеддинги документов и дедуп по смыслу
    embedder = Embedder(
        model_name=settings.pipeline.embedding_model,
        batch_size=settings.pipeline.embedding_batch,
    )
    document_vectors = embedder.encode_documents(originals)
    originals = dedup.dedup_by_embedding(
        originals, document_vectors, threshold=settings.pipeline.duplicate_threshold
    )
    print(f"\nПосле дедупа по эмбеддингам: {len(dedup.originals(originals))} первоисточников")
    print(f"Доля перепечаток: {dedup.duplicate_ratio(documents):.0%}")

    # 6. Сохранение документов
    if save:
        saved = repo.save_documents(
            originals, trust_map(), extracted=extracted, embeddings=document_vectors
        )
        print(f"Сохранено документов: {saved}")

    # 7. Кластеризация
    keys, names, contexts = clusterer.flatten_technologies(extracted)
    print(f"\nУпоминаний технологий: {len(names)}")

    if not names:
        print("Ни одного документа с технологией — кластеризовать нечего.")
        return

    name_vectors = embedder.encode_technology_names(names, contexts)
    clusters = clusterer.cluster_documents(
        keys, names, name_vectors, distance_threshold=settings.pipeline.cluster_distance
    )
     # Отбросить кластеры, совпадающие с самим направлением запроса.
    # Вектор запроса — среднее от исходной фразы и первых фраз плана:
    # одно слово «Edge» слишком коротко для надёжного эмбеддинга.
    anchor_texts = [args.query] + plan.terms_ru[:2] + plan.terms_en[:2]
    query_vector = embedder.encode(anchor_texts).mean(axis=0)
    query_vector /= np.linalg.norm(query_vector)

    cluster_name_vectors = embedder.encode([c.most_common_name() for c in clusters])
    clusters, umbrella = clusterer.drop_umbrella(clusters, cluster_name_vectors, query_vector)

    print(f"Отброшено как само направление запроса: {len(umbrella)}")
    for cluster, similarity in sorted(umbrella, key=lambda x: -x[1])[:10]:
        print(f"    {similarity:.2f}  {cluster.most_common_name()} [{cluster.size}]")
    print(f"Кластеров: {len(clusters)}\n")

    # 8. Агрегаты и сохранение кластеров
    documents_by_key = {d.key: d for d in originals}
    rows = build_cluster_rows(clusters, documents_by_key, extracted)

    if save and query_id is not None:
        cluster_ids = repo.save_clusters(query_id, rows)
        repo.finish_query(
            query_id,
            sources_processed=result.sources_processed,
            documents_collected=len(result.documents),
            candidates_found=len(clusters),
            signals_confident=0,  # проставит оценщик во втором спринте
            run_totals={
                "documents": len(originals),
                "patents": sum(r["patent_count"] for r in rows),
                "by_type": dict(Counter(d.source_type for d in originals)),
                "by_provider": dict(Counter(d.provider for d in originals)),
            },
            source_report=[s.model_dump() for s in result.per_source],
        )
        print(f"Сохранено кластеров: {len(cluster_ids)} (прогон {query_id})\n")

    for row in rows[:20]:
        print(f"  [{row['doc_count']:3}] {row['name_ru']}")

    pool.close_pool()


if __name__ == "__main__":
    main()
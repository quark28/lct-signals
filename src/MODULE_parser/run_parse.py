"""Точка входа парсера.

Запуск из корня репозитория:

    python -m src.MODULE_parser.run_parse "технологии в финтехе"
    python -m src.MODULE_parser.run_parse "квантовые сенсоры" --from 2024-01-01 --limit-sources arxiv,openalex

Это тонкая оболочка: вся логика в pipeline/, и она ничего не знает о том,
откуда пришли параметры. Когда появится веб-интерфейс, он позовёт те же
функции, а этот файл останется для отладки и пакетных прогонов.
"""

import argparse
import os
from datetime import date, datetime
from typing import Optional

from dotenv import load_dotenv

from src.llms import registry as llm_registry
from src.MODULE_parser.pipeline import clusterer, collector, dedup, extractor, planner
from src.MODULE_parser.pipeline.embedder import Embedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сбор документов по технологическому направлению"
    )
    parser.add_argument("query", help="запрос на естественном языке")
    parser.add_argument("--from", dest="date_from", help="нижняя граница даты, ГГГГ-ММ-ДД")
    parser.add_argument("--to", dest="date_to", help="верхняя граница даты, ГГГГ-ММ-ДД")
    parser.add_argument(
        "--limit-sources",
        dest="only",
        help="опросить только эти источники через запятую, для отладки",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="игнорировать кеш извлечения",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=extractor.MAX_PARALLEL,
        help="одновременных запросов к LLM при извлечении",
    )
    return parser.parse_args()


def to_date(value: Optional[str]) -> Optional[date]:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date() if value else None


def credentials_from_env() -> dict[str, dict[str, str]]:
    """Ключи источников. Пустые значения провайдеры обработают сами."""
    return {
        "openalex": {"api_key": os.getenv("OPENALEX_KEY", "")},
        "github": {"token": os.getenv("GITHUB_TOKEN", "")},
        "crossref": {"mailto": os.getenv("CONTACT_EMAIL", "")},
    }


def main() -> None:
    load_dotenv()
    args = parse_args()

    # 1. План запросов
    print(f"Запрос: {args.query}")
    plan_llm = llm_registry.for_role("extract")
    plan = planner.build_plan(plan_llm, args.query)
    print(f"  фраз ru/en: {len(plan.terms_ru)}/{len(plan.terms_en)}")
    for area in plan.adjacent_areas:
        print(f"  смежная область: {area.area} — {area.why}")

    # 2. Сбор
    result = collector.collect(
        plan,
        date_from=to_date(args.date_from),
        date_to=to_date(args.date_to),
        credentials=credentials_from_env(),
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
    extract_llm = llm_registry.for_role("extract")
    cache = None if args.no_cache else extractor.ExtractionCache()
    extracted, stats = extractor.extract_many(
        extract_llm, originals, cache=cache, max_parallel=args.parallel
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
    embedder = Embedder()
    document_vectors = embedder.encode_documents(originals)
    originals = dedup.dedup_by_embedding(originals, document_vectors)
    remaining = dedup.originals(originals)
    print(f"\nПосле дедупа по эмбеддингам: {len(remaining)} первоисточников")
    print(f"Доля перепечаток: {dedup.duplicate_ratio(documents):.0%}")

    # 6. Кластеризация по названиям технологий
    keys, names, contexts = clusterer.flatten_technologies(extracted)
    not_technology = len(extracted) - len(set(keys))
    print(
        f"\nУпоминаний технологий: {len(names)} "
        f"(документов без технологии: {not_technology})"
    )

    if not names:
        print("Ни одного документа с технологией — кластеризовать нечего.")
        return

    name_vectors = embedder.encode_technology_names(names, contexts)
    clusters = clusterer.cluster_documents(keys, names, name_vectors)
    print(f"Кластеров: {len(clusters)}\n")

    for cluster in clusters[:20]:
        print(f"  [{cluster.size:3}] {cluster.most_common_name()}")


if __name__ == "__main__":
    main()
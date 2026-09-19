"""Коллектор: обходит включённые источники и собирает документы.

Источники опрашиваются параллельно. Падение или таймаут одного источника
не роняет прогон — отказоустойчивость парсеров вынесена заказчиком
в отдельный критерий оценки.

Параллельность сделана через потоки, а не asyncio: провайдеры используют
синхронный httpx, и запрос к сети всё равно отпускает GIL, так что для
ожидания сети потоков достаточно.
"""

import concurrent.futures
from datetime import date
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel

from src.models import Document
from src.MODULE_parser.providers import registry
from src.MODULE_parser.pipeline.planner import QueryPlan

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"


class SourceResult(BaseModel):
    """Итог обращения к одному источнику. Нужен для счётчиков в интерфейсе
    и чтобы было видно, какие источники отвалились."""

    source: str
    ok: bool
    documents: int = 0
    error: Optional[str] = None
    trust: str = "medium"


class CollectResult(BaseModel):
    documents: list[Document]
    per_source: list[SourceResult]

    @property
    def sources_processed(self) -> int:
        return sum(1 for r in self.per_source if r.ok)

    @property
    def failed_sources(self) -> list[str]:
        return [r.source for r in self.per_source if not r.ok]


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"нет конфига источников: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def collect(
    plan: QueryPlan,
    *,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    credentials: Optional[dict[str, dict[str, Any]]] = None,
    only: Optional[list[str]] = None,
) -> CollectResult:
    """Собрать документы по плану.

    credentials — ключи по источникам: {"openalex": {"api_key": "..."}}.
    only — опросить только эти источники (для отладки).
    """
    config = load_config()
    defaults = config.get("defaults", {})
    sources = config.get("sources", {})
    credentials = credentials or {}

    tasks = [
        (name, spec)
        for name, spec in sources.items()
        if spec.get("enabled") and (only is None or name in only)
    ]

    documents: list[Document] = []
    per_source: list[SourceResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks) or 1) as pool:
        futures = {
            pool.submit(
                _fetch_one,
                name,
                spec,
                defaults,
                plan,
                date_from,
                date_to,
                credentials.get(name, {}),
            ): (name, spec)
            for name, spec in tasks
        }

        for future in concurrent.futures.as_completed(futures):
            name, spec = futures[future]
            trust = spec.get("trust", "medium")
            try:
                fetched = future.result()
            except Exception as exc:
                per_source.append(
                    SourceResult(source=name, ok=False, error=str(exc), trust=trust)
                )
                continue

            documents.extend(fetched)
            per_source.append(
                SourceResult(
                    source=name, ok=True, documents=len(fetched), trust=trust
                )
            )

    return CollectResult(documents=documents, per_source=per_source)


def _fetch_one(
    name: str,
    spec: dict[str, Any],
    defaults: dict[str, Any],
    plan: QueryPlan,
    date_from: Optional[date],
    date_to: Optional[date],
    credentials: dict[str, Any],
) -> list[Document]:
    """Сходить в один источник. Исключения наружу — их ловит collect."""
    provider = registry.get(name, **credentials)
    try:
        raw_query = _build_raw_query(spec, defaults, plan, date_from, date_to)
        query = provider.build_query(raw_query)
        return provider.parse_source(query)
    finally:
        close = getattr(provider, "close", None)
        if close:
            close()


def _build_raw_query(
    spec: dict[str, Any],
    defaults: dict[str, Any],
    plan: QueryPlan,
    date_from: Optional[date],
    date_to: Optional[date],
) -> dict[str, Any]:
    """Собрать общий словарь параметров. Лишние ключи провайдер отбросит сам
    в build_query — каждый берёт только то, что понимает его query_model."""
    languages = spec.get("languages", ["en"])

    raw: dict[str, Any] = {
        "terms": plan.terms_for(languages),
        "limit": spec.get("limit", defaults.get("limit", 200)),
        "date_from": date_from,
        "date_to": date_to,
        # Поля от планировщика: каждый провайдер возьмёт своё.
        "categories": plan.arxiv_categories,
        "cpc_codes": plan.cpc_codes,
    }

    # Технические настройки источника из конфига перекрывают общие.
    raw.update(spec.get("params") or {})

    # GitHub ищется своими короткими терминами, если планировщик их дал.
    if spec.get("source_type") == "code" and plan.github_terms:
        raw["terms"] = plan.github_terms

    return raw

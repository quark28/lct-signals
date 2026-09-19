"""Извлечение структурированных фактов из документов через LLM.

На вход модели идёт ТОЛЬКО заголовок и аннотация. Полные тексты не
используются нигде в системе: для определения технологии и стадии
достаточно первого экрана.

Результат кешируется по ключу документа. При повторном запросе по той же
области большинство документов уже обработано, и вызывать модель второй
раз незачем — это деньги и минуты на живой демонстрации.

Ошибки не глотаются молча: по каждому документу видно, отвалился он на
вызове модели или на разборе ответа. Без этого отладка идёт вслепую.
"""

import concurrent.futures
import threading
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.llms.base import BaseLLM, load_prompt, log_call
from src.models import Document

PROMPT_NAME = "extract_fields"
CACHE_DIR = Path(".cache/extract")
MAX_ABSTRACT_CHARS = 2000
MAX_PARALLEL = 5
MAX_ERROR_SAMPLES = 5


class TechnologyName(BaseModel):
    name_ru: str
    name_original: str


class Funding(BaseModel):
    company: Optional[str] = None
    round_stage: Optional[str] = None
    amount_usd: Optional[float] = None
    currency: str = "USD"


class ExtractedFields(BaseModel):
    """Что модель вытащила из одного документа."""

    is_technology: bool = False
    technologies: list[TechnologyName] = Field(default_factory=list)
    stage: Optional[str] = None
    stage_evidence: Optional[str] = None
    companies: list[str] = Field(default_factory=list)
    funding: list[Funding] = Field(default_factory=list)
    deployment_markers: list[str] = Field(default_factory=list)
    summary_ru: str = ""


class ExtractionStats(BaseModel):
    """Итог прогона извлечения. Показывает, где именно теряются документы."""

    total: int = 0
    from_cache: int = 0
    extracted: int = 0
    llm_errors: int = 0
    parse_errors: int = 0
    samples: list[str] = Field(default_factory=list)

    @property
    def lost(self) -> int:
        return self.total - self.from_cache - self.extracted


class ExtractionCache:
    """Файловый кеш. Позже переедет в PostgreSQL, интерфейс не изменится."""

    def __init__(self, directory: Path = CACHE_DIR):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, document_key: str) -> Path:
        safe = document_key.replace(":", "__").replace("/", "_")
        return self.directory / f"{safe}.json"

    def get(self, document_key: str) -> Optional[ExtractedFields]:
        path = self._path(document_key)
        if not path.exists():
            return None
        try:
            return ExtractedFields.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None  # битый кеш — просто перезапросим

    def put(self, document_key: str, fields: ExtractedFields) -> None:
        self._path(document_key).write_text(
            fields.model_dump_json(indent=2), encoding="utf-8"
        )


def build_user_message(document: Document) -> str:
    """Что уходит в модель. Только метаданные и первый экран текста."""
    abstract = (document.abstract or "")[:MAX_ABSTRACT_CHARS]
    return (
        f"Источник: {document.provider}\n"
        f"Дата: {document.published_at.date() if document.published_at else 'неизвестна'}\n"
        f"Заголовок: {document.title}\n\n"
        f"Текст:\n{abstract}"
    )


def extract_many(
    llm: BaseLLM,
    documents: list[Document],
    cache: Optional[ExtractionCache] = None,
    max_parallel: int = MAX_PARALLEL,
) -> tuple[dict[str, ExtractedFields], ExtractionStats]:
    """Извлечь факты из пачки документов.

    Параллельность ограничена: у провайдеров LLM есть лимит запросов
    в минуту, и без ограничения прилетит отказ по частоте.

    Возвращает словарь {ключ документа: факты} и статистику прогона.
    """
    cache = cache or ExtractionCache()
    system = load_prompt(PROMPT_NAME)

    results: dict[str, ExtractedFields] = {}
    stats = ExtractionStats(total=len(documents))
    lock = threading.Lock()

    def note_error(kind: str, message: str) -> None:
        with lock:
            if kind == "llm":
                stats.llm_errors += 1
            else:
                stats.parse_errors += 1
            if len(stats.samples) < MAX_ERROR_SAMPLES:
                stats.samples.append(message)

    def work(document: Document) -> None:
        cached = cache.get(document.key)
        if cached:
            with lock:
                results[document.key] = cached
                stats.from_cache += 1
            return

        try:
            response = llm.complete(system, build_user_message(document))
        except Exception as exc:
            note_error("llm", f"LLM [{document.key}]: {exc}")
            return

        log_call(
            response,
            purpose="extract",
            prompt_name=PROMPT_NAME,
            doc_keys=[document.key],
            reason="лёгкая модель на массовом извлечении, роль extract в config/models.yaml",
        )

        try:
            fields = llm._parse(response.text, ExtractedFields)
        except Exception as exc:
            note_error(
                "parse",
                f"РАЗБОР [{document.key}]: {exc} | ответ: {response.text[:200]!r}",
            )
            return

        cache.put(document.key, fields)
        with lock:
            results[document.key] = fields
            stats.extracted += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
        list(pool.map(work, documents))

    return results, stats


def extract_one(
    llm: BaseLLM,
    document: Document,
    cache: Optional[ExtractionCache] = None,
) -> Optional[ExtractedFields]:
    """Извлечь факты из одного документа. Удобно для отладки промпта."""
    results, _ = extract_many(llm, [document], cache=cache, max_parallel=1)
    return results.get(document.key)
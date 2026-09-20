"""Настройки приложения.

Единственное место, которое читает окружение. Остальной код получает
готовый объект Settings и никогда не лезет в os.environ напрямую —
это позволяет запускать пайплайн и из командной строки, и из веб-интерфейса
без изменений в ядре.

Здесь же собраны гиперпараметры, которые потом станут ползунками
в интерфейсе: пороги кластеризации и дедупликации, лимиты, параллельность.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DatabaseSettings:
    dsn: str = ""
    min_pool_size: int = 1
    max_pool_size: int = 10


@dataclass
class SourceCredentials:
    """Ключи источников. Пустое значение допустимо: провайдер либо
    работает без ключа, либо сам сообщит об ошибке."""

    openalex_key: str = ""
    github_token: str = ""
    contact_email: str = ""

    def as_kwargs(self) -> dict[str, dict[str, str]]:
        """В том виде, в каком их ждёт коллектор."""
        return {
            "openalex": {"api_key": self.openalex_key},
            "github": {"token": self.github_token},
            "crossref": {"mailto": self.contact_email},
        }


@dataclass
class PipelineSettings:
    """Гиперпараметры прогона. Будущие ползунки интерфейса."""

    # Дедупликация
    duplicate_threshold: float = 0.93      # косинусная близость перепечаток

    # Кластеризация
    cluster_distance: float = 0.15         # порог расстояния, complete linkage

    # Извлечение
    llm_parallel: int = 5                  # одновременных вызовов модели
    max_abstract_chars: int = 2000         # сколько текста уходит в модель

    # Эмбеддинги
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_batch: int = 64

    # Выдача
    top_n: int = 15                        # размер итоговой выдачи
    min_confidence: float = 0.75           # порог уверенности для счётчика ТЗ


@dataclass
class Settings:
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    credentials: SourceCredentials = field(default_factory=SourceCredentials)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)

    cache_dir: Path = ROOT / ".cache"
    logs_dir: Path = ROOT / "logs"
    models_config: Path = ROOT / "config" / "models.yaml"
    sources_config: Path = ROOT / "src" / "MODULE_parser" / "config" / "sources.yaml"
    prompts_dir: Path = ROOT / "src" / "system_prompts"

    def validate(self) -> list[str]:
        """Что мешает запуску. Пустой список — можно работать."""
        problems: list[str] = []

        if not self.database.dsn:
            problems.append("не задан POSTGRES_DSN")
        if not self.credentials.openalex_key:
            problems.append("не задан OPENALEX_KEY — OpenAlex отдаст 100 запросов в сутки")
        if not self.models_config.exists():
            problems.append(f"нет файла {self.models_config}")
        if not self.sources_config.exists():
            problems.append(f"нет файла {self.sources_config}")
        if not self.prompts_dir.exists():
            problems.append(f"нет папки промптов {self.prompts_dir}")

        return problems


_SETTINGS: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """Настройки собираются один раз за процесс."""
    global _SETTINGS
    if _SETTINGS is not None and not reload:
        return _SETTINGS

    load_dotenv(ROOT / ".env")

    _SETTINGS = Settings(
        database=DatabaseSettings(
            dsn=os.getenv("POSTGRES_DSN", ""),
        ),
        credentials=SourceCredentials(
            openalex_key=os.getenv("OPENALEX_KEY", ""),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            contact_email=os.getenv("CONTACT_EMAIL", ""),
        ),
        pipeline=PipelineSettings(
            duplicate_threshold=float(os.getenv("DUPLICATE_THRESHOLD", "0.93")),
            cluster_distance=float(os.getenv("CLUSTER_DISTANCE", "0.20")),
            llm_parallel=int(os.getenv("LLM_PARALLEL", "5")),
            top_n=int(os.getenv("TOP_N", "15")),
        ),
    )
    return _SETTINGS

"""Очистка после отладочных прогонов.

Запускается вручную, а не автоматически при закрытии окна: кнопка Stop
в редакторе часто убивает процесс жёстко, и обработчики выхода
не срабатывают.

    python -m scripts.cleanup --soft   # стереть кластеры, документы оставить
    python -m scripts.cleanup --all    # стереть всё: базу, кеш, логи
    python -m scripts.cleanup --cache  # только кеш извлечения

Кеш моделей Hugging Face (~/.cache/huggingface) не трогается никогда:
эмбеддер весит почти полгигабайта, качать его заново на каждой отладке
не нужно.
"""

import argparse
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

CACHE_DIR = Path(".cache")
LOGS_DIR = Path("logs")


def confirm(question: str) -> bool:
    answer = input(f"{question} [y/N]: ").strip().lower()
    return answer in {"y", "yes", "д", "да"}


def clear_clusters() -> None:
    """Стереть кластеры и прогоны, документы оставить.

    На отладке кластеризации это экономит часы: сбор и извлечение
    повторять не приходится.
    """
    from src.db import pool

    with pool.cursor() as cur:
        cur.execute("TRUNCATE clusters, cluster_documents, queries RESTART IDENTITY CASCADE")
    print("Кластеры и прогоны стёрты, документы сохранены.")


def clear_database() -> None:
    """Стереть все данные. Схема остаётся — пересоздавать не надо."""
    from src.db import pool

    with pool.cursor() as cur:
        cur.execute(
            "TRUNCATE documents, clusters, cluster_documents, queries "
            "RESTART IDENTITY CASCADE"
        )
    print("База очищена.")


def clear_cache() -> None:
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"Кеш удалён: {CACHE_DIR}")
    else:
        print("Кеша нет.")


def clear_logs() -> None:
    if LOGS_DIR.exists():
        shutil.rmtree(LOGS_DIR)
        print(f"Логи удалены: {LOGS_DIR}")
    else:
        print("Логов нет.")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Очистка после отладочных прогонов")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--soft", action="store_true", help="стереть кластеры, документы оставить")
    group.add_argument("--all", action="store_true", help="стереть базу, кеш и логи")
    group.add_argument("--cache", action="store_true", help="стереть только кеш извлечения")
    group.add_argument("--logs", action="store_true", help="стереть только логи")
    parser.add_argument("-y", "--yes", action="store_true", help="не спрашивать подтверждение")
    args = parser.parse_args()

    if args.cache:
        clear_cache()
        return

    if args.logs:
        clear_logs()
        return

    if args.soft:
        if not args.yes and not confirm("Стереть кластеры и прогоны?"):
            print("Отменено.")
            return
        clear_clusters()
        return

    if args.all:
        if not args.yes and not confirm("Стереть ВСЁ: базу, кеш извлечения и логи?"):
            print("Отменено.")
            return
        clear_database()
        clear_cache()
        clear_logs()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОтменено.")
        sys.exit(1)

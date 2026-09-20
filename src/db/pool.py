"""Подключение к PostgreSQL.

Пул соединений один на процесс: открывать соединение на каждый запрос
дорого, а на прогоне их сотни.

Драйвер psycopg (3-я версия) выбран вместо asyncpg, потому что весь
пайплайн синхронный — провайдеры используют синхронный httpx, и
параллельность сделана потоками.
"""

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

_POOL: Optional[ConnectionPool] = None


def dsn() -> str:
    from src.settings import get_settings
    value = get_settings().database.dsn
    if not value:
        raise RuntimeError("не задан POSTGRES_DSN в .env")
    return value


def get_pool() -> ConnectionPool:
    """Пул создаётся при первом обращении."""
    global _POOL
    if _POOL is None:
        _POOL = ConnectionPool(
            conninfo=dsn(),
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _POOL


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Соединение из пула. Транзакция фиксируется при выходе без ошибки."""
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


@contextmanager
def cursor() -> Iterator[psycopg.Cursor]:
    """Курсор из пула. Самый частый способ обращения к базе."""
    with connection() as conn:
        with conn.cursor() as cur:
            yield cur


def fetch_all(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def fetch_one(sql: str, params: tuple[Any, ...] = ()) -> Optional[dict[str, Any]]:
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def execute(sql: str, params: tuple[Any, ...] = ()) -> None:
    with cursor() as cur:
        cur.execute(sql, params)


def healthcheck() -> bool:
    """Проверка, что база жива и расширение векторов на месте.

    Зовётся перед прогоном: лучше упасть сразу с понятной ошибкой,
    чем после десяти минут сбора документов.
    """
    try:
        with cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            return cur.fetchone() is not None
    except Exception:
        return False


def close_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None

"""Ограничение нагрузки, повторы и таймауты.

Отказоустойчивость парсеров вынесена заказчиком в отдельный критерий
оценки, а демонстрация идёт вживую при жюри. Поэтому три механизма:

  RateLimiter — не чаще N запросов в секунду к одному хосту.
                Источники банят за частоту, а arXiv прямо просит
                паузу в 3 секунды.

  retry       — повтор с нарастающей задержкой. Сеть периодически
                отдаёт ошибку без причины.

  run_with_timeout — жёсткая граница на источник. Если один API завис,
                прогон продолжается без него.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 1.0
DEFAULT_TIMEOUT = 20.0


class RateLimiter:
    """Минимальный интервал между вызовами. Потокобезопасен.

    Используется как контекстный менеджер:

        limiter = RateLimiter(min_interval=3.0)
        with limiter:
            ...запрос...
    """

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            sleep_for = self.min_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call = time.monotonic()

    def __enter__(self) -> "RateLimiter":
        self.wait()
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class Semaphore:
    """Ограничитель одновременных вызовов.

    Обёртка над threading.Semaphore, чтобы в коде пайплайна было явно
    видно назначение: у провайдеров LLM есть лимит запросов в минуту,
    и без ограничения прилетает отказ по частоте.
    """

    def __init__(self, max_concurrent: int):
        self._semaphore = threading.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent

    def __enter__(self) -> "Semaphore":
        self._semaphore.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._semaphore.release()


def retry(
    attempts: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Повтор с нарастающей задержкой: backoff, 2*backoff, 4*backoff.

    Последняя неудача пробрасывается наружу — глушить ошибку здесь нельзя,
    вызывающий код должен отличать "источник пустой" от "источник упал".
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = backoff
            last_error: Optional[BaseException] = None

            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_error = exc
                    if attempt == attempts:
                        break
                    if on_retry:
                        on_retry(attempt, exc)
                    time.sleep(delay)
                    delay *= 2

            raise last_error  # type: ignore[misc]

        return wrapper

    return decorator


def run_with_timeout(
    func: Callable[..., T],
    *args: Any,
    timeout: float = DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> T:
    """Выполнить функцию с ограничением по времени.

    Оговорка: поток, в котором крутится func, остановить нельзя — он
    продолжит работать в фоне до завершения запроса. Для сетевых вызовов
    это приемлемо, потому что у самих клиентов тоже есть таймаут, и поток
    умрёт сам. Здесь важно освободить основной прогон, а не убить поток.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            raise TimeoutError(
                f"{getattr(func, '__name__', 'вызов')} не уложился в {timeout} с"
            ) from exc

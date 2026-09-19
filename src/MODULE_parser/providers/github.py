"""Провайдер GitHub через поиск репозиториев. Нужен personal access token.

Документация: https://docs.github.com/rest/search/search
Без токена — 10 поисковых запросов в минуту, с токеном 30.

Зачем нужен: дата создания репозитория и рост звёзд — ранний индикатор
коммерциализации. Технология обычно появляется в коде до того, как о ней
пишут в медиа.

Ограничения поиска GitHub: не больше 1000 результатов на запрос,
per_page максимум 100. ИЛИ между фразами не поддерживается,
поэтому по фразе на запрос.
"""

import time
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

import httpx
from pydantic import Field

from src.models import Document
from src.MODULE_parser.providers.base import BaseProvider, SourceQuery
from src.MODULE_parser.utils.lang import detect as detect_language

API_URL = "https://api.github.com/search/repositories"
PAGE_SIZE = 100          # максимум GitHub
MAX_RESULTS = 1000       # жёсткий потолок поиска GitHub
REQUEST_DELAY = 2.0      # 30 запросов в минуту с токеном
TIMEOUT = 20.0


class GitHubQuery(SourceQuery):
    """Параметры, специфичные для GitHub."""

    min_stars: int = 0                 # отсечь заброшенные однодневки
    include_forks: bool = False        # форки дублируют статистику
    languages: list[str] = Field(default_factory=list)  # ["Python", "Rust"]


class GitHubProvider(BaseProvider):
    name: ClassVar[str] = "github"
    source_type: ClassVar[str] = "code"
    query_model: ClassVar[type[SourceQuery]] = GitHubQuery

    def __init__(self, token: str = "", user_agent: str = "lct-signals/0.1"):
        headers = {
            "User-Agent": user_agent,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        self._authenticated = bool(token)
        self._client = httpx.Client(
            timeout=TIMEOUT,
            headers=headers,
            follow_redirects=True,
        )

    # --- публичный метод контракта -------------------------------------

    def parse_source(self, query: GitHubQuery) -> list[Document]:
        seen: set[str] = set()
        documents: list[Document] = []
        per_term = max(1, query.limit // len(query.terms))

        for term in query.terms:
            for item in self._search(term, query, per_term):
                repo_id = str(item.get("id", ""))
                if not repo_id or repo_id in seen:
                    continue
                seen.add(repo_id)
                documents.append(self._parse_repo(item))

            if len(documents) >= query.limit:
                break

        return documents[: query.limit]

    # --- внутреннее -----------------------------------------------------

    def _search(self, term: str, query: GitHubQuery, limit: int) -> list[dict[str, Any]]:
        q = self._build_q(term, query)
        items: list[dict[str, Any]] = []
        page = 1
        limit = min(limit, MAX_RESULTS)

        while len(items) < limit:
            params = {
                "q": q,
                "sort": "updated",
                "order": "desc",
                "per_page": min(PAGE_SIZE, limit - len(items)),
                "page": page,
            }
            response = self._client.get(API_URL, params=params)
            response.raise_for_status()
            payload = response.json()

            batch = payload.get("items", [])
            if not batch:
                break

            items.extend(batch)
            if len(batch) < params["per_page"]:
                break

            page += 1
            time.sleep(REQUEST_DELAY)

        return items[:limit]

    def _build_q(self, term: str, query: GitHubQuery) -> str:
        """Собрать строку поиска в синтаксисе GitHub."""
        parts = [f'"{term}"']

        if not query.include_forks:
            parts.append("fork:false")
        if query.min_stars > 0:
            parts.append(f"stars:>={query.min_stars}")
        for language in query.languages:
            parts.append(f"language:{language}")

        # GitHub фильтрует по дате создания репозитория.
        if query.date_from and query.date_to:
            parts.append(
                f"created:{query.date_from.isoformat()}..{query.date_to.isoformat()}"
            )
        elif query.date_from:
            parts.append(f"created:>={query.date_from.isoformat()}")
        elif query.date_to:
            parts.append(f"created:<={query.date_to.isoformat()}")

        return " ".join(parts)

    def _parse_repo(self, repo: dict[str, Any]) -> Document:
        title = repo.get("full_name") or repo.get("name") or ""
        description = repo.get("description") or ""

        return Document(
            provider=self.name,
            doc_id=str(repo.get("id", "")),
            url=repo.get("html_url", ""),
            title=title,
            abstract=description,
            # Дата создания, а не последнего изменения: нам нужен момент появления.
            published_at=self._parse_date(repo.get("created_at")),
            language=detect_language(title, description),
            source_type=self.source_type,
            authors=[(repo.get("owner") or {}).get("login", "")]
            if repo.get("owner")
            else [],
            raw={
                "stars": repo.get("stargazers_count"),
                "forks": repo.get("forks_count"),
                "open_issues": repo.get("open_issues_count"),
                "programming_language": repo.get("language"),
                "topics": repo.get("topics", []),
                "pushed_at": repo.get("pushed_at"),
                "updated_at": repo.get("updated_at"),
            },
        )

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return None

    def close(self) -> None:
        self._client.close()

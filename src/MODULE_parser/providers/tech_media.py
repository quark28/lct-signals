"""tech_media: поиск по архивам отраслевых СМИ через WordPress REST API.

Почему не RSS: лента отдаёт только последние 10–50 записей (окно от часов до недели),
а эталонные сигналы датируются 2025–2026. WP REST (/wp-json/wp/v2/posts?search=)
ищет по всему архиву сайта, бесплатно и без ключей.

Проверено 22.09.2026 (scripts/probe_media2.py): работает на techcrunch, siliconangle,
theaiinsider, pulse2, roboticsandautomationnews. geekwire, eu-startups,
datacenterdynamics отдают 403 (Cloudflare), pandaily 404, eetimes висит.

Поведение при сбоях:
- 401/403/404 или не-JSON — сайт выключается до конца прогона, остальные работают;
- 429 — пауза и один повтор, затем сайт выключается;
- таймаут/сетевая ошибка — пропускается одна фраза, после 3 подряд сайт выключается;
- если все сайты упали — исключение (коллектор пометит источник как failed, а не «0 документов»).
"""
from __future__ import annotations

import html
import re
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dtime, timezone
from typing import Optional
from urllib.parse import urlparse

import httpx
from pydantic import Field

from src.models import Document
from src.MODULE_parser.providers.base import BaseProvider, SourceQuery

try:  # сертификаты из хранилища ОС: чинит CERTIFICATE_VERIFY_FAILED на Windows
    import truststore
    _VERIFY: ssl.SSLContext | bool = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
except ImportError:
    _VERIFY = True

DEFAULT_SITES = [
    "techcrunch.com",
    "siliconangle.com",
    "theaiinsider.tech",
    "pulse2.com",
    "roboticsandautomationnews.com",
]
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
TIMEOUT_S = 30.0
DELAY_S = 1.0            # пауза между запросами к одному сайту
RATE_LIMIT_PAUSE_S = 15.0
MAX_NET_ERRORS = 3
FATAL_STATUSES = {401, 403, 404, 410}

_TAGS = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")
_WP_TAIL = re.compile(r"(\[\s*(…|&hellip;|\.\.\.)\s*\]|The post .* appeared first on .*)$", re.S)


class SiteDisabled(Exception):
    """Сайт недоступен до конца прогона."""


class TechMediaQuery(SourceQuery):
    sites: list[str] = Field(default_factory=lambda: list(DEFAULT_SITES))
    per_term: int = Field(default=20, ge=1, le=100)  # лимит WP: per_page ≤ 100


def clean_text(value: str) -> str:
    text = _TAGS.sub(" ", html.unescape(value or ""))  # unescape до чистки: бывает &lt;p&gt;
    text = _WP_TAIL.sub("", text.strip())
    return _SPACES.sub(" ", text).strip()


def parse_wp_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", ""))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)  # date_gmt приходит без зоны, но это UTC


class TechMediaProvider(BaseProvider):
    name = "tech_media"
    source_type = "media"
    query_model = TechMediaQuery

    def __init__(self, **_: object) -> None:
        self.client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=TIMEOUT_S,
            follow_redirects=True,
            verify=_VERIFY,
        )
        self.site_report: dict[str, str] = {}

    def close(self) -> None:
        self.client.close()

    # --- один запрос ---
    def _search(self, site: str, term: str, query: TechMediaQuery) -> list[dict]:
        params: dict[str, object] = {
            "search": term,
            "orderby": "relevance",  # по умолчанию WP сортирует по дате
            "per_page": query.per_term,
            "_fields": "id,date_gmt,link,title,excerpt",
        }
        # Даты передаём только если заданы: без --from/--to ищем по всему архиву.
        if query.date_from:
            params["after"] = datetime.combine(query.date_from, dtime.min).isoformat()
        if query.date_to:
            params["before"] = datetime.combine(query.date_to, dtime.max).isoformat(timespec="seconds")

        url = f"https://{site}/wp-json/wp/v2/posts"
        for attempt in (1, 2):
            r = self.client.get(url, params=params)
            time.sleep(DELAY_S)
            if r.status_code == 429 and attempt == 1:
                time.sleep(RATE_LIMIT_PAUSE_S)
                continue
            if r.status_code in FATAL_STATUSES or r.status_code == 429:
                raise SiteDisabled(f"HTTP {r.status_code}")
            if r.status_code == 400:  # WP отвергает параметр: странная фраза, не повод выключать сайт
                return []
            r.raise_for_status()
            try:
                data = r.json()
            except ValueError:
                raise SiteDisabled("не JSON (заглушка/Cloudflare)")
            if not isinstance(data, list):
                raise SiteDisabled(f"неожиданный ответ: {str(data)[:80]}")
            return data
        return []

    # --- один сайт, все фразы последовательно ---
    def _crawl_site(self, site: str, query: TechMediaQuery) -> list[tuple[int, dict]]:
        found: list[tuple[int, dict]] = []  # (позиция в выдаче, пост)
        net_errors = total_errors = 0
        for term in query.terms:
            try:
                posts = self._search(site, term, query)
                net_errors = 0
            except SiteDisabled as exc:
                self.site_report[site] = f"выключен: {exc}"
                return found
            except httpx.HTTPError as exc:
                net_errors += 1
                total_errors += 1
                if net_errors >= MAX_NET_ERRORS:
                    self.site_report[site] = f"выключен: {type(exc).__name__} ×{net_errors}"
                    return found
                continue
            for rank, post in enumerate(posts):
                post["_site"], post["_term"] = site, term
                found.append((rank, post))
        self.site_report[site] = f"ok, {len(found)} попаданий, сетевых ошибок {total_errors}"
        return found

    def _to_document(self, post: dict) -> Optional[Document]:
        link = post.get("link") or ""
        if not link:
            return None
        site = post["_site"]
        return Document(
            provider=self.name,
            doc_id=f"{site}:{post.get('id', '')}",
            url=link,
            title=clean_text((post.get("title") or {}).get("rendered", "")),
            abstract=clean_text((post.get("excerpt") or {}).get("rendered", "")),
            published_at=parse_wp_date(post.get("date_gmt", "")),
            language="en",
            source_type=self.source_type,
            raw={"site": site, "term": post["_term"], "domain": urlparse(link).netloc},
        )

    def parse_source(self, query: SourceQuery) -> list[Document]:
        if not isinstance(query, TechMediaQuery):
            query = TechMediaQuery(**query.model_dump())
        sites = query.sites or DEFAULT_SITES
        with ThreadPoolExecutor(max_workers=len(sites)) as pool:
            per_site = list(pool.map(lambda s: self._crawl_site(s, query), sites))

        if all(self.site_report.get(s, "").startswith("выключен") for s in sites):
            raise RuntimeError(f"tech_media: все сайты недоступны: {self.site_report}")

        # Общий лимит (стоимость LLM-извлечения!) делим честно: сначала первые места
        # выдачи всех сайтов и фраз, потом вторые и т.д. Дубли по URL отбрасываем.
        hits = sorted((h for site_hits in per_site for h in site_hits), key=lambda h: h[0])
        seen: set[str] = set()
        docs: list[Document] = []
        for _, post in hits:
            doc = self._to_document(post)
            if doc is None or doc.url in seen:
                continue
            seen.add(doc.url)
            docs.append(doc)
            if len(docs) >= query.limit:
                break
        return docs

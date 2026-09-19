"""Планировщик: запрос пользователя на естественном языке -> план поисковых запросов.

Единственное место, где LLM решает, ЧТО искать. Дальше модель на поиск
не влияет: она только извлекает факты из найденного.

План показывается пользователю с возможностью правки — он видит,
во что развернулся его запрос.
"""

from typing import Optional

from pydantic import BaseModel, Field

from src.llms.base import BaseLLM, load_prompt, log_call

PROMPT_NAME = "plan_queries"


class AdjacentArea(BaseModel):
    area: str
    why: str


class QueryPlan(BaseModel):
    """Что и где искать. Технические настройки источников сюда не входят —
    они в config/sources.yaml."""

    terms_ru: list[str] = Field(default_factory=list)
    terms_en: list[str] = Field(default_factory=list)
    arxiv_categories: list[str] = Field(default_factory=list)
    cpc_codes: list[str] = Field(default_factory=list)
    github_terms: list[str] = Field(default_factory=list)
    adjacent_areas: list[AdjacentArea] = Field(default_factory=list)

    def terms_for(self, languages: list[str]) -> list[str]:
        """Фразы под конкретный источник: русские, английские или оба."""
        terms: list[str] = []
        if "ru" in languages:
            terms.extend(self.terms_ru)
        if "en" in languages:
            terms.extend(self.terms_en)
        return terms or self.terms_en or self.terms_ru


def fallback_plan(user_query: str) -> QueryPlan:
    """План на случай, когда LLM недоступна.

    Без расширения и без смежных областей, зато демонстрация не падает.
    """
    return QueryPlan(terms_ru=[user_query], terms_en=[user_query])


def build_plan(llm: BaseLLM, user_query: str) -> QueryPlan:
    """Развернуть запрос пользователя в план. При ошибке LLM — запасной план."""
    system = load_prompt(PROMPT_NAME)
    user = f"Запрос пользователя: {user_query}"

    try:
        response = llm.complete(system, user)
    except Exception:
        return fallback_plan(user_query)

    log_call(
        response,
        purpose="plan_queries",
        prompt_name=PROMPT_NAME,
        doc_keys=[],
        reason="планирование запроса, модель задана ролью plan в config/models.yaml",
    )

    try:
        return llm._parse(response.text, QueryPlan)
    except Exception:
        return fallback_plan(user_query)

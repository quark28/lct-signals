"""Планировщик: запрос пользователя на естественном языке -> план поисковых запросов.

Единственное место, где LLM решает, ЧТО искать. Дальше модель на поиск
не влияет: она только извлекает факты из найденного.

Два шага (2 вызова LLM на весь прогон):
1. plan_subtopics — направление раскладывается на 8–15 конкретных подтехнологий.
   Причина: по фразе «edge computing» находится академический MEC, а слабые сигналы
   лежат в соседних конкретных технологиях (сжатие моделей, NPU, фотоника…),
   где слова «edge» нет. Замер 22.09: 0 из 16 эталонов Edge при старом плане.
2. plan_queries — на каждую подтехнологию короткие поисковые фразы,
   плюс коды arXiv/CPC и смежные области.

Фразы в плане перемешаны по кругу (1-я фраза каждой подтехнологии, потом 2-я…),
чтобы источник, который берёт только первые N фраз (crossref — 8), всё равно
покрывал разные подтехнологии, а не одну.

План показывается пользователю с возможностью правки.
"""

from typing import Optional

from pydantic import BaseModel, Field

from src.llms.base import BaseLLM, load_prompt, log_call

SUBTOPICS_PROMPT = "plan_subtopics"
QUERIES_PROMPT = "plan_queries"

MAX_SUBTOPICS = 15
MAX_EN_PER_SUBTOPIC = 3
MAX_RU_PER_SUBTOPIC = 1   # русские фразы идут в платный Яндекс — держим их мало


class AdjacentArea(BaseModel):
    area: str
    why: str


class Subtopic(BaseModel):
    name_en: str
    name_ru: str = ""
    why: str = ""


class SubtopicList(BaseModel):
    core_terms: list[str] = Field(default_factory=list)
    subtopics: list[Subtopic] = Field(default_factory=list)


class SubtopicTerms(BaseModel):
    subtopic: str
    terms_en: list[str] = Field(default_factory=list)
    terms_ru: list[str] = Field(default_factory=list)


class QueriesAnswer(BaseModel):
    by_subtopic: list[SubtopicTerms] = Field(default_factory=list)
    arxiv_categories: list[str] = Field(default_factory=list)
    cpc_codes: list[str] = Field(default_factory=list)
    adjacent_areas: list[AdjacentArea] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """Что и где искать. Технические настройки источников — в config/sources.yaml."""

    terms_ru: list[str] = Field(default_factory=list)
    terms_en: list[str] = Field(default_factory=list)
    arxiv_categories: list[str] = Field(default_factory=list)
    cpc_codes: list[str] = Field(default_factory=list)
    github_terms: list[str] = Field(default_factory=list)  # GitHub выключен, поле для совместимости
    adjacent_areas: list[AdjacentArea] = Field(default_factory=list)
    subtopics: list[Subtopic] = Field(default_factory=list)
    # Синонимы самого направления («edge computing», «периферийные вычисления»).
    # В поиск НЕ идут: только якорь для отсева зонтичных кластеров в run_parse.
    core_terms: list[str] = Field(default_factory=list)

    def terms_for(self, languages: list[str]) -> list[str]:
        """Фразы под конкретный источник: русские, английские или оба."""
        terms: list[str] = []
        if "ru" in languages:
            terms.extend(self.terms_ru)
        if "en" in languages:
            terms.extend(self.terms_en)
        return terms or self.terms_en or self.terms_ru


def fallback_plan(user_query: str) -> QueryPlan:
    """План на случай, когда LLM недоступна: демонстрация не падает."""
    return QueryPlan(terms_ru=[user_query], terms_en=[user_query], core_terms=[user_query])


def _clean(term: str) -> str:
    return " ".join(term.replace('"', " ").split())


def _round_robin(groups: list[list[str]], per_group: int) -> list[str]:
    """1-я фраза каждой группы, потом 2-я… Дубли (без учёта регистра) выкидываются."""
    seen: set[str] = set()
    result: list[str] = []
    for position in range(per_group):
        for group in groups:
            if position < len(group):
                term = _clean(group[position])
                key = term.lower()
                if term and key not in seen:
                    seen.add(key)
                    result.append(term)
    return result


def _ask(llm: BaseLLM, prompt_name: str, user: str, model: type[BaseModel]) -> Optional[BaseModel]:
    """Один вызов LLM с логом. None — если модель недоступна или ответ не разобрался."""
    try:
        response = llm.complete(load_prompt(prompt_name), user)
    except Exception:
        return None
    log_call(
        response,
        purpose=prompt_name,
        prompt_name=prompt_name,
        doc_keys=[],
        reason="планирование запроса, модель задана ролью plan в config/models.yaml",
    )
    try:
        return llm._parse(response.text, model)
    except Exception:
        return None


def build_plan(llm: BaseLLM, user_query: str) -> QueryPlan:
    """Развернуть запрос пользователя в план. При сбоях — деградация, а не падение."""
    # Шаг 1: подтехнологии
    step1 = _ask(llm, SUBTOPICS_PROMPT, f"Запрос пользователя: {user_query}", SubtopicList)
    subtopics = [s for s in (step1.subtopics if step1 else []) if s.name_en.strip()]
    subtopics = subtopics[:MAX_SUBTOPICS]
    if not subtopics:
        return fallback_plan(user_query)

    # Шаг 2: фразы под каждую подтехнологию
    listing = "\n".join(f"- {s.name_en} ({s.name_ru})" for s in subtopics)
    step2 = _ask(
        llm,
        QUERIES_PROMPT,
        f"Запрос пользователя: {user_query}\n\nПодтехнологии:\n{listing}",
        QueriesAnswer,
    )

    by_name = {t.subtopic.strip().lower(): t for t in (step2.by_subtopic if step2 else [])}
    groups_en: list[list[str]] = []
    groups_ru: list[list[str]] = []
    for s in subtopics:
        terms = by_name.get(s.name_en.strip().lower())
        # Если шаг 2 упал или пропустил подтехнологию — ищем по её названию.
        groups_en.append((terms.terms_en if terms and terms.terms_en else [s.name_en]))
        groups_ru.append((terms.terms_ru if terms and terms.terms_ru else [s.name_ru] if s.name_ru else []))

    return QueryPlan(
        terms_en=_round_robin(groups_en, MAX_EN_PER_SUBTOPIC),
        terms_ru=_round_robin(groups_ru, MAX_RU_PER_SUBTOPIC),
        arxiv_categories=(step2.arxiv_categories[:6] if step2 else []),
        cpc_codes=(step2.cpc_codes[:6] if step2 else []),
        adjacent_areas=(step2.adjacent_areas if step2 else []),
        subtopics=subtopics,
        core_terms=[_clean(t) for t in step1.core_terms if _clean(t)][:4] or [user_query],
    )
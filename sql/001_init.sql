-- Схема хранилища. Накатывается один раз:
--   psql -U lct_app -d lct -f sql/001_init.sql
--
-- Три сущности с разным жизненным циклом:
--   documents — накапливаются и переиспользуются между запросами
--   clusters  — пересобираются при каждом новом запросе
--   queries   — журнал прогонов, отсюда счётчики для интерфейса
--
-- ПРИНЦИП ХРАНЕНИЯ ПРИЗНАКОВ.
-- В базе лежат только СЫРЫЕ агрегаты: счётчики, даты, временные ряды.
-- Производные признаки модели (доли, отношения, всплеск по Клейнбергу,
-- нормированные коэффициенты) считаются на лету в src/features/.
-- Причина: формулы будут переписываться десятки раз при подборе модели,
-- а часть из них нормируется на объём прогона, то есть зависит от
-- контекста, а не от самого кластера.
--
-- Состав сырых полей подобран так, чтобы из них вычислялся весь
-- развёрнутый набор на 59 признаков.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- queries

CREATE TABLE IF NOT EXISTS queries (
    id              BIGSERIAL PRIMARY KEY,
    user_query      TEXT        NOT NULL,
    plan            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    date_from       DATE,
    date_to         DATE,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,

    -- Счётчики для трёх плашек интерфейса (пункт "будет плюсом" в ТЗ)
    sources_processed   INTEGER NOT NULL DEFAULT 0,
    documents_collected INTEGER NOT NULL DEFAULT 0,
    candidates_found    INTEGER NOT NULL DEFAULT 0,
    signals_confident   INTEGER NOT NULL DEFAULT 0,  -- уверенность выше 75%

    -- Итоги по прогону в целом. Нужны для нормировки признаков:
    -- коэффициент изобретательской активности и число публикаций
    -- считаются относительно объёма прогона, иначе масштабы разных
    -- предметных областей несопоставимы.
    -- {"documents": 550, "patents": 10, "by_type": {...}, "by_provider": {...}}
    run_totals      JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Какие источники отвалились и почему
    source_report   JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS queries_started_idx ON queries (started_at DESC);

-- -------------------------------------------------------------- documents

CREATE TABLE IF NOT EXISTS documents (
    doc_key         TEXT PRIMARY KEY,          -- provider:doc_id
    provider        TEXT        NOT NULL,
    source_type     TEXT        NOT NULL,      -- science | patent | media | social | code | government
    trust           TEXT        NOT NULL DEFAULT 'medium',  -- high | medium | low
    url             TEXT        NOT NULL,
    domain          TEXT,
    title           TEXT        NOT NULL,
    abstract        TEXT        NOT NULL DEFAULT '',
    published_at    TIMESTAMPTZ,
    language        TEXT,                      -- ISO 639-1, NULL = не определён
    machine_translated BOOLEAN  NOT NULL DEFAULT false,
    authors         TEXT[]      NOT NULL DEFAULT '{}',

    -- Перепечатка: ссылка на первоисточник. Дубли не удаляются — они
    -- доказательство подтверждения и вход признака "доля перепечаток".
    duplicate_of    TEXT REFERENCES documents (doc_key) ON DELETE SET NULL,

    extracted       JSONB,                     -- что вытащила LLM
    raw             JSONB       NOT NULL DEFAULT '{}'::jsonb,

    embedding       vector(384),               -- multilingual-e5-small
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_provider_idx    ON documents (provider);
CREATE INDEX IF NOT EXISTS documents_published_idx   ON documents (published_at);
CREATE INDEX IF NOT EXISTS documents_duplicate_idx   ON documents (duplicate_of);
CREATE INDEX IF NOT EXISTS documents_source_type_idx ON documents (source_type);

-- --------------------------------------------------------------- clusters

CREATE TABLE IF NOT EXISTS clusters (
    id              BIGSERIAL PRIMARY KEY,
    query_id        BIGINT NOT NULL REFERENCES queries (id) ON DELETE CASCADE,

    -- Название
    name_ru         TEXT NOT NULL,
    name_original   TEXT,
    name_source_key TEXT,                      -- документ, подтверждающий название
    aliases         TEXT[] NOT NULL DEFAULT '{}',

    -- ---- Вердикт оценщика (результат, не признак) ----
    status          TEXT,                      -- weak_signal | mature | hype | noise
    confidence      REAL,                      -- калиброванная вероятность, 0..1
    rank_position   INTEGER,                   -- место в выдаче
    exclusion_reason TEXT,

    -- ---- СЫРЫЕ АГРЕГАТЫ: объём ----
    doc_count           INTEGER NOT NULL DEFAULT 0,  -- первоисточников
    mention_count       INTEGER NOT NULL DEFAULT 0,  -- упоминаний технологии
    duplicate_count     INTEGER NOT NULL DEFAULT 0,  -- перепечаток

    -- Счётчики по каждому провайдеру: {"arxiv": 6, "openalex": 40, ...}
    -- Отсюда 13 признаков "счётчик по провайдеру".
    provider_counts     JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- По типам источника: {"science": 46, "patent": 3, "media": 0, ...}
    -- Отсюда доли по типам, конвергенция доменов, попарные отношения.
    type_counts         JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Перепечатки по типам: {"media": 4, "social": 1}
    type_duplicate_counts JSONB NOT NULL DEFAULT '{}'::jsonb,

    unique_domains      INTEGER NOT NULL DEFAULT 0,
    unique_companies    INTEGER NOT NULL DEFAULT 0,
    company_mentions    INTEGER NOT NULL DEFAULT 0,  -- с повторами

    -- ---- СЫРЫЕ АГРЕГАТЫ: время ----
    first_published_at  TIMESTAMPTZ,
    last_published_at   TIMESTAMPTZ,
    mean_published_at   TIMESTAMPTZ,

    -- Временные ряды по кварталам. Отсюда считаются всплеск по Клейнбергу
    -- (общий и по каждому типу), давность всплеска и рост последнего
    -- квартала:
    -- {"all": {"2024Q1": 3, "2024Q2": 7},
    --  "science": {...}, "patent": {...}, "media": {...}, "social": {...}}
    timeline            JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- ---- СЫРЫЕ АГРЕГАТЫ: патенты ----
    patent_count        INTEGER NOT NULL DEFAULT 0,
    patent_applicants   INTEGER NOT NULL DEFAULT 0,  -- уникальных
    patent_countries    INTEGER NOT NULL DEFAULT 0,  -- уникальных
    -- Патенты топ-заявителя: отсюда индекс технологической специализации.
    patent_top_applicant_count INTEGER NOT NULL DEFAULT 0,
    -- Патенты с зарубежным заявителем: отсюда коэффициент
    -- технологической зависимости.
    patent_foreign_count       INTEGER NOT NULL DEFAULT 0,
    -- Распределение по странам: {"US": 6, "CN": 3, "RU": 1}
    patent_country_counts      JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- ---- СЫРЫЕ АГРЕГАТЫ: финансирование ----
    funding_events      INTEGER NOT NULL DEFAULT 0,
    max_round_stage     SMALLINT,              -- seed=1, A=2, B=3, C+=4
    max_round_amount    NUMERIC,
    total_round_amount  NUMERIC,
    -- Распределение по стадиям: {"seed": 3, "series_a": 1}
    round_stage_counts  JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- ---- СЫРЫЕ АГРЕГАТЫ: зрелость ----
    -- Максимальная стадия: концепция=1, прототип=2, пилот=3,
    -- раннее внедрение=4. Это и есть УГТ.
    readiness_level     SMALLINT,
    -- Распределение стадий: {"прототип": 4, "пилот": 1}
    stage_counts        JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Число маркеров внедрения. Отсюда уровень рыночной зрелости.
    deployment_markers  INTEGER NOT NULL DEFAULT 0,

    -- ---- СЫРЫЕ АГРЕГАТЫ: код ----
    github_repos        INTEGER NOT NULL DEFAULT 0,
    github_stars_sum    INTEGER NOT NULL DEFAULT 0,
    github_stars_max    INTEGER NOT NULL DEFAULT 0,

    -- ---- Доверенность источников ----
    -- {"high": 30, "medium": 12, "low": 4}. Нужно для правила ТЗ:
    -- источник с низкой доверенностью не может быть единственным
    -- основанием для включения технологии в выдачу.
    trust_counts        JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Готовая карточка от LLM. Генерируется лениво, по клику.
    card            JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS clusters_query_idx  ON clusters (query_id);
CREATE INDEX IF NOT EXISTS clusters_status_idx ON clusters (query_id, status);
CREATE INDEX IF NOT EXISTS clusters_rank_idx   ON clusters (query_id, rank_position);

-- ------------------------------------------------------ cluster_documents

-- Связь многие-ко-многим: один документ может упоминать несколько
-- технологий и попадать в несколько кластеров.
CREATE TABLE IF NOT EXISTS cluster_documents (
    cluster_id   BIGINT NOT NULL REFERENCES clusters (id) ON DELETE CASCADE,
    doc_key      TEXT   NOT NULL REFERENCES documents (doc_key) ON DELETE CASCADE,
    name_variant TEXT,                         -- как технология названа здесь
    PRIMARY KEY (cluster_id, doc_key)
);

CREATE INDEX IF NOT EXISTS cluster_documents_doc_idx ON cluster_documents (doc_key);
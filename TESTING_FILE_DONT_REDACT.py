from dotenv import load_dotenv
from src.MODULE_parser.providers.yandex_search import YandexSearchProvider
import os

load_dotenv()
p = YandexSearchProvider(
    api_key=os.environ["YANDEX_API_KEY"],
    folder_id=os.environ["YANDEX_FOLDER_ID"],
)
docs = p.parse_source(p.build_query({
    "terms": ["neuromorphic computing funding"],
    "limit": 10,
    "deferred": False,  # sync-режим для быстрой проверки
}))
for d in docs:
    print(f"{d.raw.get('domain', ''):30} | {d.title[:60]}")

import time
start = time.monotonic()
docs = p.parse_source(p.build_query({
    "terms": ["neuromorphic computing funding"],
    "limit": 10,
    "deferred": True,
}))
print(f"{len(docs)} документов за {time.monotonic() - start:.0f} с")
p.close()

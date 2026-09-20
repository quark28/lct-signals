from dotenv import load_dotenv
from src.db import pool, repo
from src.models import Document
load_dotenv()

qid = repo.create_query("проверка", {"terms_ru": ["тест"]})

doc = Document(
    provider="arxiv", doc_id="test123",
    url="https://arxiv.org/abs/test123",
    title="Тестовый документ", abstract="Текст",
    source_type="science", language="ru",
)
repo.save_documents([doc], {"arxiv": "high"})

ids = repo.save_clusters(qid, [{
    "name_ru": "тестовая технология",
    "document_keys": [doc.key],
    "name_variants": ["тестовая технология"],
    "doc_count": 1,
    "provider_counts": {"arxiv": 1},
    "type_counts": {"science": 1},
    "timeline": {"all": {"2026Q3": 1}},
}])

print("кластер:", repo.get_clusters(qid)[0]["name_ru"])
print("документы:", repo.get_cluster_documents(ids[0]))
pool.close_pool()
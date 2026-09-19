from src.llms.base import log_call

r = llm.complete("Отвечай одним словом.", "Столица Франции?")
print(r.text)
log_call(r, purpose="smoke_test", prompt_name="inline", doc_keys=[], reason="проверка журнала")
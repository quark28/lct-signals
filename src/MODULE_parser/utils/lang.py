"""Определение языка документа по тексту.

API источников язык почти нигде не сообщают, поэтому определяем локально.
py3langid детерминирован и работает без сети.

norm_probs=True обязателен: без него classify возвращает логарифм
правдоподобия (отрицательное число), а не вероятность.
"""

from py3langid.langid import LanguageIdentifier, MODEL_FILE

MIN_CHARS = 20          # на коротких строках детектор ненадёжен
MIN_CONFIDENCE = 0.90   # ниже порога считаем язык неопределённым

# Ограничиваем набор языков: все наши источники русско- или англоязычные.
# Без ограничения детектор путает короткие технические тексты с
# голландским, африкаанс и латынью.
_identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
_identifier.set_languages(["en", "ru"])


def detect(*parts: str) -> str | None:
    """Вернуть код языка ISO 639-1 или None, если уверенности не хватает.

    Принимает несколько кусков текста (заголовок, аннотация) и склеивает их:
    чем длиннее образец, тем надёжнее ответ.
    """
    text = " ".join(p for p in parts if p).strip()
    if len(text) < MIN_CHARS:
        return None

    code, probability = _identifier.classify(text)
    return code if probability >= MIN_CONFIDENCE else None
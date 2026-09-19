import re


CRISIS_PATTERNS = [
    r"আত্মহত্যা",
    r"মরে যেতে চাই",
    r"বাঁচতে চাই না",
    r"নিজেকে শেষ",
    r"নিজেকে মেরে",
    r"suicid",
    r"kill myself",
    r"end my life",
]


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_crisis_language(text: str) -> bool:
    cleaned_text = clean_text(text).lower()
    return any(
        re.search(pattern, cleaned_text, flags=re.IGNORECASE)
        for pattern in CRISIS_PATTERNS
    )
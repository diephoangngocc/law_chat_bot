import re
import unicodedata
from collections import Counter


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [tok for tok in _TOKEN_RE.findall(normalized) if len(tok) > 1]


def ascii_tokens(text: str) -> list[str]:
    normalized = strip_accents(text.lower()).replace("đ", "d")
    return [tok for tok in _TOKEN_RE.findall(normalized) if len(tok) > 1]


def term_counts(text: str) -> Counter:
    tokens = tokenize(text)
    folded_tokens = ascii_tokens(text)
    combined = tokens + [tok for tok in folded_tokens if tok not in tokens]
    return Counter(combined)


def compact(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."

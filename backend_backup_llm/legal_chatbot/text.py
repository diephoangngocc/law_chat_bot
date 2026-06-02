from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, List, Optional

# Giữ stopword vừa đủ. Không loại quá mạnh vì tiếng Việt pháp lý có nhiều cụm quan trọng.
VI_STOPWORDS = {
    "là", "và", "hoặc", "thì", "có", "bị", "được", "của", "cho", "với", "trong",
    "khi", "nếu", "như", "một", "các", "những", "này", "kia", "đó", "tôi", "bạn",
    "a", "b", "c", "hỏi", "về", "theo", "tại", "đến", "ra", "vào", "nào", "ạ",
    "không", "xin", "cho", "biết", "vậy", "thì", "sẽ", "phải", "làm", "sao",
}

LEGAL_KEEP_TOKENS = {
    "người", "tội", "điều", "khoản", "điểm", "phạt", "tù", "tiền", "chết", "dao", "súng",
    "tài", "sản", "ma", "túy", "hành", "vi", "hậu", "quả",
}


POINT_ALIASES = {
    "a", "b", "c", "d", "đ", "e", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
}


def normalize_text(text: str) -> str:
    """Chuẩn hóa nhẹ tiếng Việt nhưng vẫn giữ dấu để match chính xác."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("Đ", "đ")
    text = text.lower()
    # Chuẩn hóa một số cách viết tắt thường gặp.
    text = re.sub(r"\bđ\s*\.?\s*(\d+)", r"điều \1", text)
    text = re.sub(r"\bk\s*\.?\s*(\d+)", r"khoản \1", text)
    text = re.sub(r"\bd\s*(\d{2,3})\b", r"điều \1", text)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text or ""))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    text = normalize_text(text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ_%\s]", " ", text)
    tokens = [tok for tok in text.split() if tok]
    if not keep_stopwords:
        out = []
        for tok in tokens:
            if len(tok) <= 1 and not tok.isdigit():
                continue
            if tok in LEGAL_KEEP_TOKENS or tok not in VI_STOPWORDS:
                out.append(tok)
        tokens = out
    return tokens


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        raw = str(item or "").strip()
        norm = normalize_text(raw)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(raw)
    return out


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    norm = normalize_text(text)
    norm_no_acc = strip_accents(norm)
    for kw in keywords:
        k = normalize_text(kw)
        if k and (k in norm or strip_accents(k) in norm_no_acc):
            return True
    return False


def phrase_count(text: str, phrases: Iterable[str]) -> int:
    norm = normalize_text(text)
    no_acc = strip_accents(norm)
    count = 0
    for phrase in phrases:
        p = normalize_text(phrase)
        if not p:
            continue
        if p in norm or strip_accents(p) in no_acc:
            count += 1
    return count


def article_number(text: str) -> Optional[str]:
    norm = normalize_text(text)
    m = re.search(r"điều\s*(\d{1,3}[a-z]?)", norm)
    if m:
        return m.group(1)
    # Một số KG dùng mã node kiểu D123 hoặc Dieu_123.
    m = re.search(r"\bd(?:ieu)?[_\- ]?(\d{1,3}[a-z]?)\b", strip_accents(norm))
    if m:
        return m.group(1)
    return None


def clause_number(text: str) -> Optional[str]:
    norm = normalize_text(text)
    m = re.search(r"khoản\s*(\d{1,2}[a-z]?)", norm)
    if m:
        return m.group(1)
    # Một số mã node dùng K123_1 hoặc Khoan_1.
    no_acc = strip_accents(norm)
    m = re.search(r"\bk(?:hoan)?[_\- ]?\d{1,3}[a-z]?[_\- ]?(\d{1,2}[a-z]?)\b", no_acc)
    if m:
        return m.group(1)
    m = re.search(r"\bk(?:hoan)?[_\- ]?(\d{1,2}[a-z]?)\b", no_acc)
    if m:
        return m.group(1)
    return None


def point_letter(text: str) -> Optional[str]:
    norm = normalize_text(text)
    m = re.search(r"điểm\s*([a-zđ])\b", norm)
    if m:
        return m.group(1)
    # Một số mã node dùng P123_1_a / Diem_a.
    no_acc = strip_accents(norm)
    m = re.search(r"\b(?:p|diem)[_\- ]?(?:\d{1,3}[a-z]?[_\- ]?)?(?:\d{1,2}[a-z]?[_\- ]?)?([a-z])\b", no_acc)
    if m and m.group(1) in POINT_ALIASES:
        return m.group(1)
    return None


def extract_article_mentions(text: str) -> List[str]:
    mentions = []
    norm = normalize_text(text)
    for m in re.finditer(r"điều\s*(\d{1,3}[a-z]?)", norm):
        mentions.append(m.group(1))
    return unique_keep_order(mentions)


def extract_clause_mentions(text: str) -> List[str]:
    mentions = []
    norm = normalize_text(text)
    for m in re.finditer(r"khoản\s*(\d{1,2}[a-z]?)", norm):
        mentions.append(m.group(1))
    return unique_keep_order(mentions)


def extract_point_mentions(text: str) -> List[str]:
    mentions = []
    norm = normalize_text(text)
    for m in re.finditer(r"điểm\s*([a-zđ])\b", norm):
        mentions.append(m.group(1))
    return unique_keep_order(mentions)


def legal_reference_display(article: str | None = None, clause: str | None = None, point: str | None = None) -> str:
    parts = []
    if point:
        parts.append(f"Điểm {point}")
    if clause:
        parts.append(f"Khoản {clause}")
    if article:
        parts.append(f"Điều {article}")
    return " ".join(parts)


def short_text(text: str, max_chars: int = 650) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 3]
    # Cắt đẹp ở dấu câu/gần khoảng trắng.
    for sep in [". ", "; ", ", ", " "]:
        idx = cut.rfind(sep)
        if idx > max_chars * 0.55:
            cut = cut[: idx + len(sep)].strip()
            break
    return cut.rstrip() + "..."


def lexical_similarity(a: str, b: str) -> float:
    """Điểm giống nhau nhẹ, không cần dependency ngoài."""
    a_norm = strip_accents(normalize_text(a))
    b_norm = strip_accents(normalize_text(b))
    if not a_norm or not b_norm:
        return 0.0
    at = set(tokenize(a_norm))
    bt = set(tokenize(b_norm))
    jaccard = len(at & bt) / max(len(at | bt), 1)
    seq = SequenceMatcher(None, a_norm[:300], b_norm[:300]).ratio()
    return 0.65 * jaccard + 0.35 * seq

"""Load and query law article content for tooltip display."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_CONTENT: dict[str, str] = {}
_loaded = False


def _load() -> None:
    global _loaded
    if _loaded:
        return
    path = _DATA_DIR / "law_content.json"
    if path.exists():
        _CONTENT.update(json.loads(path.read_text(encoding="utf-8")))
    _loaded = True


def _normalize_ref(display: str) -> str:
    """Chuẩn hóa về dạng key "Điều X Khoản Y Điểm Z" (thứ tự chuẩn).

    Input có thể là bất kỳ thứ tự nào: "Điểm c Khoản 1 Điều 123" hoặc "Điều 123 Khoản 1".
    """
    s = display.strip()
    m_article = re.search(r"[Đđ]i[eề]u\s+(\d+[a-zA-Z]?)", s, re.IGNORECASE)
    m_clause  = re.search(r"[Kk]ho[aả]n\s+(\d+)", s, re.IGNORECASE)
    m_point   = re.search(r"[Đđ]i[eể]m\s+([a-zđ])", s, re.IGNORECASE)

    if not m_article:
        return s

    parts = [f"Điều {m_article.group(1)}"]
    if m_clause:
        parts.append(f"Khoản {m_clause.group(1)}")
    if m_point:
        parts.append(f"Điểm {m_point.group(1)}")
    return " ".join(parts)


def _extract_parts(display: str):
    """Trả về (article, clause, point) tuple từ display string."""
    s = display.strip()
    m_a = re.search(r"[Đđ]i[eề]u\s+(\d+[a-zA-Z]?)", s, re.IGNORECASE)
    m_c = re.search(r"[Kk]ho[aả]n\s+(\d+)", s, re.IGNORECASE)
    m_p = re.search(r"[Đđ]i[eể]m\s+([a-zđ])", s, re.IGNORECASE)
    return (
        m_a.group(1) if m_a else None,
        m_c.group(1) if m_c else None,
        m_p.group(1) if m_p else None,
    )


def lookup(display: str) -> Optional[str]:
    """Tìm nội dung theo display string. Trả về full text, không truncate."""
    _load()
    key = _normalize_ref(display)
    text = _CONTENT.get(key)
    if not text:
        # Fallback: thử chỉ phần Điều
        parts = key.split(" Khoản ")
        if len(parts) > 1:
            text = _CONTENT.get(parts[0])
    return text.strip() if text else None


def lookup_hierarchy(display: str) -> List[dict]:
    """Trả về danh sách theo thứ tự Điều → Khoản → Điểm, mỗi cấp có label + content.

    Ví dụ input "Điểm a Khoản 1 Điều 123" trả về:
    [
      {"label": "Điều 123", "content": "Tội giết người"},
      {"label": "Khoản 1", "content": "Người nào giết người..."},
      {"label": "Điểm a", "content": "Giết 02 người trở lên"}
    ]
    """
    _load()
    article, clause, point = _extract_parts(display)
    if not article:
        return []

    result = []

    art_content = _CONTENT.get(f"Điều {article}")
    if art_content:
        result.append({"label": f"Điều {article}", "content": art_content.strip()})

    if clause:
        cl_content = _CONTENT.get(f"Điều {article} Khoản {clause}")
        if cl_content:
            result.append({"label": f"Khoản {clause}", "content": cl_content.strip()})

    if clause and point:
        pt_content = _CONTENT.get(f"Điều {article} Khoản {clause} Điểm {point}")
        if pt_content:
            result.append({"label": f"Điểm {point}", "content": pt_content.strip()})

    return result


def lookup_article(article_num: str | int) -> Optional[str]:
    return lookup(f"Điều {article_num}")


def lookup_clause(article_num: str | int, clause_num: str | int) -> Optional[str]:
    return lookup(f"Điều {article_num} Khoản {clause_num}")


def lookup_point(article_num: str | int, clause_num: str | int, point_letter: str) -> Optional[str]:
    return lookup(f"Điều {article_num} Khoản {clause_num} Điểm {point_letter}")

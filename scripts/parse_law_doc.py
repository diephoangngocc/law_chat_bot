"""
Parse Bộ luật Hình sự từ file .doc (dùng Word COM trên Windows).
Output: data/law_content.json
  {
    "Điều 1": "Nhiệm vụ của Bộ luật Hình sự\n...",
    "Điều 2 Khoản 1": "...",
    "Điều 3 Khoản 1 Điểm a": "..."
  }
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def extract_text_via_com(doc_path: str) -> list[str]:
    """Dùng Word COM để lấy text, trả về list dòng."""
    try:
        import win32com.client
    except ImportError:
        print("Cần pywin32: pip install pywin32")
        sys.exit(1)

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(doc_path)
        text = doc.Content.Text
        doc.Close(False)
    finally:
        word.Quit()

    lines = [ln.strip() for ln in re.split(r"\r\n|\n|\r|\x0b|\x0c", text)]
    return lines


# ── Regex patterns ──────────────────────────────────────────────────────────

RE_DIEU = re.compile(r"^Điều\s+(\d+[a-z]?)\.\s*(.*)", re.IGNORECASE)
# Khoản: dòng bắt đầu bằng số + dấu chấm + tuỳ chọn footnote [n] + khoảng trắng
# Ví dụ: "1. Nội dung" hoặc "1.[94] Nội dung"
RE_KHOAN = re.compile(r"^(\d+)\.\s*(?:\[\d+\])?\s+(.*)")
# Điểm: dòng bắt đầu bằng chữ thường + dấu ) + khoảng trắng
RE_DIEM = re.compile(r"^([a-zđ])\)\s+(.*)")


def _key(article: str, clause: str | None = None, point: str | None = None) -> str:
    k = f"Điều {article}"
    if clause:
        k += f" Khoản {clause}"
    if point:
        k += f" Điểm {point}"
    return k


def parse(lines: list[str]) -> dict[str, str]:
    content: dict[str, list[str]] = {}

    cur_article: str | None = None
    cur_clause: str | None = None
    cur_point: str | None = None

    def flush(buf: list[str], article: str | None, clause: str | None, point: str | None) -> None:
        if not article or not buf:
            return
        k = _key(article, clause, point)
        text = " ".join(buf).strip()
        if text:
            content.setdefault(k, []).append(text)

    buf: list[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m_dieu = RE_DIEU.match(line)
        m_khoan = RE_KHOAN.match(line)
        m_diem = RE_DIEM.match(line)

        if m_dieu:
            flush(buf, cur_article, cur_clause, cur_point)
            buf = []
            cur_article = m_dieu.group(1)
            cur_clause = None
            cur_point = None
            title = m_dieu.group(2).strip()
            if title:
                buf.append(title)

        elif m_khoan and cur_article:
            flush(buf, cur_article, cur_clause, cur_point)
            buf = []
            cur_clause = m_khoan.group(1)
            cur_point = None
            rest = m_khoan.group(2).strip()
            if rest:
                buf.append(rest)

        elif m_diem and cur_article and cur_clause:
            flush(buf, cur_article, cur_clause, cur_point)
            buf = []
            cur_point = m_diem.group(1)
            rest = m_diem.group(2).strip()
            if rest:
                buf.append(rest)

        elif cur_article:
            # Tiếp tục đoạn hiện tại — bỏ các dòng header Chương/Phần
            if not re.match(r"^(Chương|Phần|MỤC|Mục)\s", line, re.IGNORECASE):
                buf.append(line)

    flush(buf, cur_article, cur_clause, cur_point)

    # Join lists → string
    result: dict[str, str] = {k: " ".join(v) for k, v in content.items()}

    # Thêm nội dung tổng hợp cho mỗi Điều (gộp tất cả khoản/điểm)
    article_texts: dict[str, list[str]] = {}
    for k, v in result.items():
        parts = k.split(" Khoản ")
        article_key = parts[0]  # "Điều X"
        article_texts.setdefault(article_key, []).append(v)

    for art_key, texts in article_texts.items():
        if art_key not in result:
            result[art_key] = " ".join(texts[:3])  # tóm tắt 3 đoạn đầu

    return result


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python parse_law_doc.py <path_to_.doc>")
        sys.exit(1)

    doc_path = sys.argv[1]
    print(f"Đang đọc: {doc_path}")
    lines = extract_text_via_com(doc_path)
    print(f"Tổng số dòng: {len(lines)}")

    print("Đang parse...")
    data = parse(lines)
    print(f"Số mục đã parse: {len(data)}")

    out = ROOT / "data" / "law_content.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu: {out}")

    # In sample
    samples = list(data.items())[:5]
    for k, v in samples:
        print(f"\n[{k}]\n{v[:120]}...")


if __name__ == "__main__":
    main()

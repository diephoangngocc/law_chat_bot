from __future__ import annotations

import json
import sys
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_kg.llm import HuggingFaceLLM
from legal_kg.pipeline import LegalReasoningPipeline


@lru_cache(maxsize=2)
def get_pipeline(use_llm: bool = False) -> LegalReasoningPipeline:
    llm = HuggingFaceLLM.from_env() if use_llm else None
    return LegalReasoningPipeline(ROOT / "data", llm=llm)


def build_reply(output: dict[str, Any], use_llm: bool = False) -> str:
    facts = output.get("facts") if isinstance(output.get("facts"), dict) else {}
    result = output.get("result") if isinstance(output.get("result"), dict) else {}
    candidates = output.get("candidates") if isinstance(output.get("candidates"), list) else []
    laws = result.get("dieu_luat") if isinstance(result.get("dieu_luat"), list) else []
    best_law = laws[0] if laws and isinstance(laws[0], dict) else {}

    crime = result.get("toi_danh_de_xuat") or "chưa xác định được tội danh phù hợp"
    if use_llm:
        lines = [f"Tôi đã bật LLM để phân tích facts và suy luận trên evidence KG. Kết quả đề xuất là **{crime}**."]
    else:
        lines = [f"Tôi đã truy xuất KG ở chế độ offline và ứng viên mạnh nhất là **{crime}**."]

    if best_law:
        article = best_law.get("article_id") or ""
        title = best_law.get("title") or ""
        clause = best_law.get("clause") or ""
        lines.append(f"Căn cứ chính: **{article} - {title}**, {clause}.")

    penalties = result.get("khung_hinh_phat_du_kien")
    if isinstance(penalties, list) and penalties:
        lines.append("Khung hình phạt trong KG: " + "; ".join(str(item) for item in penalties) + ".")

    query_terms = facts.get("tu_khoa_truy_van")
    if isinstance(query_terms, list) and query_terms:
        lines.append("Từ khóa truy xuất: " + ", ".join(str(item) for item in query_terms[:8]) + ".")

    alternatives = []
    for item in candidates[1:4]:
        if isinstance(item, dict):
            alternatives.append(f"{item.get('article_id')}: {item.get('title')}")
    if alternatives:
        lines.append("Ứng viên cần đối chiếu thêm: " + "; ".join(alternatives) + ".")

    if use_llm:
        lines.append("Lưu ý: LLM giúp lập luận tốt hơn nhưng vẫn cần kiểm tra thủ công theo hồ sơ và quy định pháp luật.")
    else:
        lines.append(
            "Lưu ý: kết quả này chỉ hỗ trợ truy xuất điều luật ở chế độ offline; cần kiểm tra chuyên môn hoặc bật LLM để chốt khoản, điểm và lập luận pháp lý cuối cùng."
        )
    return "\n\n".join(lines)


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_common_headers("application/json; charset=utf-8")
        self.end_headers()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            message = str(payload.get("message") or "").strip()
            top_k = int(payload.get("top_k") or 5)
            use_llm = bool(payload.get("use_llm"))
            top_k = max(1, min(10, top_k))
            if not message:
                raise ValueError("Cần nhập tóm tắt vụ án.")

            data = get_pipeline(use_llm).run(message, top_k=top_k).to_json_dict(include_candidates=True)
            self._send_json({"reply": build_reply(data, use_llm=use_llm), "data": data, "mode": "LLM" if use_llm else "Offline"})
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_GET(self) -> None:
        self._send_json({"status": "ok", "service": "Legal KG Chat API"})

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_common_headers("application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_common_headers(self, content_type: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

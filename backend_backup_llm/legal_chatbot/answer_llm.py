from __future__ import annotations

import json
from typing import Dict, List

from .answer_template import TemplateAnswerer
from .llm_client import LocalLLMError, OllamaClient
from .text import short_text


class LocalLLMAnswerer:
    """Generate a fluent answer from retrieved legal evidence only."""

    def __init__(self, client: OllamaClient):
        self.client = client
        self.fallback = TemplateAnswerer()

    def generate(self, question: str, evidence: Dict[str, object]) -> str:
        if not self._has_useful_evidence(evidence):
            return self.fallback.generate(question, evidence)

        system_prompt = (
            "Bạn là trợ lý tra cứu pháp luật Việt Nam. "
            "Bạn KHÔNG phải luật sư và KHÔNG được tự kết luận thay cơ quan có thẩm quyền. "
            "Nhiệm vụ của bạn là diễn đạt lại kết quả truy xuất KG/RAG thành câu trả lời dễ hiểu. "
            "Chỉ được dùng thông tin trong EVIDENCE. "
            "Không được tự bịa Điều, Khoản, Điểm, mức phạt, tình tiết hoặc căn cứ pháp lý. "
            "Nếu evidence không đủ, hãy nói rõ thiếu thông tin nào. "
            "Không trình bày quá dài. Không để lộ suy luận nội bộ."
        )
        user_prompt = self._build_prompt(question, evidence)
        try:
            answer = self.client.chat(system_prompt=system_prompt, user_prompt=user_prompt)
            return self._postprocess(answer)
        except LocalLLMError as exc:
            fallback_answer = self.fallback.generate(question, evidence)
            return (
                "Không gọi được LLM local nên hệ thống tự chuyển sang chế độ không LLM.\n\n"
                f"Lý do: {exc}\n\n"
                f"{fallback_answer}"
            )

    def _has_useful_evidence(self, evidence: Dict[str, object]) -> bool:
        return bool(evidence.get("article") or evidence.get("contexts") or evidence.get("kg_paths"))

    def _build_prompt(self, question: str, evidence: Dict[str, object]) -> str:
        compact_evidence = self._compact_evidence(evidence)
        evidence_text = json.dumps(compact_evidence, ensure_ascii=False, indent=2)
        return f"""
QUESTION:
{question}

EVIDENCE:
{evidence_text}

QUY TẮC BẮT BUỘC:
- Chỉ trả lời dựa trên EVIDENCE.
- Không thêm điều luật/khoản/điểm/mức phạt nếu evidence không có.
- Nếu có nhiều Khoản/Điểm, nêu rõ hệ thống đang nghiêng về căn cứ nào và vì sao.
- Nếu chỉ xác định được Điều nhưng chưa xác định được Khoản/Điểm, phải nói rõ.
- Không dùng markdown bảng.

ĐỊNH DẠNG TRẢ LỜI:
1. Kết luận hỗ trợ tra cứu: nêu Điều/Khoản/Điểm phù hợp nhất nếu có.
2. Đối chiếu dữ kiện: gạch đầu dòng các dữ kiện khớp.
3. Hình phạt/căn cứ: chỉ nêu nếu evidence có.
4. Thông tin còn thiếu: nêu các thông tin cần bổ sung để kết luận chắc hơn.
5. Lưu ý pháp lý ngắn.
""".strip()

    def _compact_evidence(self, evidence: Dict[str, object]) -> Dict[str, object]:
        out = dict(evidence)

        contexts = out.get("contexts") or []
        compact_contexts: List[Dict[str, object]] = []
        for c in contexts[:6]:
            if not isinstance(c, dict):
                continue
            compact_contexts.append(
                {
                    "title": c.get("title"),
                    "content": short_text(str(c.get("content", "")), 700),
                    "score": c.get("score"),
                    "source": c.get("source"),
                    "reference": c.get("reference") or c.get("scope_display") or c.get("title"),
                }
            )
        out["contexts"] = compact_contexts

        if "candidates" in out:
            out.pop("candidates", None)
        return out

    def _postprocess(self, answer: str) -> str:
        answer = (answer or "").strip()
        if not answer:
            return "LLM local không trả về nội dung. Bạn hãy thử lại hoặc chuyển sang chế độ không LLM."
        if "không thay thế" not in answer.lower() and "tư vấn pháp lý" not in answer.lower():
            answer += "\n\nLưu ý: Nội dung này chỉ hỗ trợ tra cứu, không thay thế tư vấn pháp lý chính thức."
        return answer

from __future__ import annotations

import json
from typing import Dict, List

from .answer_template import TemplateAnswerer
from .llm_client import ChatClient, LocalLLMError
from .text import short_text


class LocalLLMAnswerer:
    """Use local LLM only to rewrite retrieved KG/RAG evidence into a smoother answer."""

    def __init__(self, client: ChatClient):
        self.client = client
        self.fallback = TemplateAnswerer()

    def generate(self, question: str, evidence: Dict[str, object]) -> str:
        if not self._has_useful_evidence(evidence):
            return self.fallback.generate(question, evidence)

        system_prompt = (
            "Bạn là trợ lý tra cứu pháp luật Việt Nam. "
            "Bạn chỉ được trả lời dựa trên EVIDENCE được cung cấp. "
            "Không tự bịa Điều, Khoản, Điểm, mức phạt, tình tiết hoặc căn cứ pháp lý. "
            "Nếu evidence thiếu thông tin, hãy nói rõ phần còn thiếu. "
            "Không khẳng định chắc chắn tội danh nếu dữ kiện chưa đủ. "
            "Trả lời ngắn gọn, mạch lạc, không để lộ suy luận nội bộ."
        )
        user_prompt = self._build_prompt(question, evidence)
        try:
            answer = self.client.chat(system_prompt=system_prompt, user_prompt=user_prompt)
            return self._postprocess(answer)
        except LocalLLMError as exc:
            fallback_answer = self.fallback.generate(question, evidence)
            return (
                "Không gọi được LLM local nên hệ thống chuyển sang chế độ không LLM.\n\n"
                f"Lý do: {exc}\n\n"
                f"{fallback_answer}"
            )

    def _has_useful_evidence(self, evidence: Dict[str, object]) -> bool:
        return bool(evidence.get("article") or evidence.get("contexts") or evidence.get("kg_paths"))

    def _build_prompt(self, question: str, evidence: Dict[str, object]) -> str:
        # Rút gọn thật mạnh để model nhỏ như Qwen 1.7B / context 4096 không bị lỗi n_keep > n_ctx.
        evidence_text = json.dumps(self._compact_evidence(evidence), ensure_ascii=False, indent=2)
        return f"""
CÂU HỎI:
{question}

EVIDENCE ĐÃ RÚT GỌN:
{evidence_text}

QUY TẮC BẮT BUỘC:
- Chỉ trả lời dựa trên EVIDENCE.
- Không thêm điều luật/khoản/điểm/mức phạt nếu evidence không có.
- Nếu chỉ xác định được Điều nhưng chưa xác định được Khoản/Điểm, phải nói rõ.
- Nếu evidence thiếu thông tin, nêu thông tin còn thiếu.
- Không dùng markdown bảng.

ĐỊNH DẠNG TRẢ LỜI NGẮN:
1. Kết luận hỗ trợ tra cứu.
2. Căn cứ/evidence chính.
3. Hình phạt hoặc thông tin còn thiếu nếu có.
4. Lưu ý pháp lý ngắn.
""".strip()

    def _compact_evidence(self, evidence: Dict[str, object]) -> Dict[str, object]:
        """Keep only the fields that the LLM needs.

        The full evidence is still returned to the frontend in data.evidence. This compact
        version is only for the local LLM prompt.
        """
        ref = evidence.get("reference") or {}
        article = evidence.get("article") or {}
        target = evidence.get("target_node") or {}

        out: Dict[str, object] = {
            "intent": evidence.get("intent"),
            "reference": {
                "display": ref.get("display"),
                "article": ref.get("article"),
                "clause": ref.get("clause"),
                "point": ref.get("point"),
            },
            "article": article.get("name") if isinstance(article, dict) else None,
            "target_node": target.get("name") if isinstance(target, dict) else None,
            "confidence_label": evidence.get("confidence_label"),
        }

        penalties = evidence.get("penalties") or []
        compact_penalties: List[str] = []
        for p in penalties[:3]:
            if isinstance(p, dict):
                compact_penalties.append(short_text(str(p.get("name") or p.get("label") or ""), 260))
        out["penalties"] = [p for p in compact_penalties if p]

        matched = evidence.get("matched_facts") or []
        out["matched_facts"] = [short_text(str(x), 220) for x in matched[:6]]

        missing = evidence.get("missing_info") or []
        out["missing_info"] = [short_text(str(x), 220) for x in missing[:4]]

        contexts = evidence.get("contexts") or []
        compact_contexts: List[Dict[str, object]] = []
        for c in contexts[:3]:
            if not isinstance(c, dict):
                continue
            cref = c.get("reference") or {}
            ref_display = cref.get("display") if isinstance(cref, dict) else None
            compact_contexts.append(
                {
                    "reference": ref_display,
                    "title": short_text(str(c.get("title") or ""), 180),
                    "content": short_text(str(c.get("content") or ""), 360),
                    "score": c.get("score"),
                    "kind": c.get("node_kind") or c.get("label") or c.get("source"),
                }
            )
        out["contexts"] = compact_contexts

        kg_paths = evidence.get("kg_paths") or []
        out["kg_paths"] = [short_text(str(x), 220) for x in kg_paths[:3]]

        return out

    def _postprocess(self, answer: str) -> str:
        answer = (answer or "").strip()
        if not answer:
            return "LLM local không trả về nội dung. Bạn hãy thử lại hoặc chuyển sang chế độ không LLM."
        lowered = answer.lower()
        if "không thay thế" not in lowered and "tư vấn pháp lý" not in lowered:
            answer += "\n\nLưu ý: Nội dung này chỉ hỗ trợ tra cứu, không thay thế tư vấn pháp lý chính thức."
        return answer

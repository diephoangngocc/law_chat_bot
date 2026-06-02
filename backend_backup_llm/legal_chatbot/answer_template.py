from __future__ import annotations

from typing import Dict, List

from .text import short_text


class TemplateAnswerer:
    """Sinh câu trả lời không LLM: rõ ràng, ít bịa, bám evidence."""

    def generate(self, question: str, evidence: Dict[str, object]) -> str:
        article = evidence.get("article") or {}
        reference = evidence.get("reference") or {}
        target_node = evidence.get("target_node") or {}
        clauses: List[Dict[str, object]] = evidence.get("clauses") or []
        points: List[Dict[str, object]] = evidence.get("points") or []
        penalties: List[Dict[str, object]] = evidence.get("penalties") or []
        matched_facts: List[str] = evidence.get("matched_facts") or []
        missing_info: List[str] = evidence.get("missing_info") or []
        contexts: List[Dict[str, object]] = evidence.get("contexts") or []
        confidence = evidence.get("confidence", 0.0)
        confidence_label = evidence.get("confidence_label", "thấp")
        intent = evidence.get("intent", "UNKNOWN")

        lines: List[str] = []

        # 1. Câu trả lời chính ở mức Điểm/Khoản/Điều.
        ref_display = reference.get("display") or ""
        if ref_display:
            if target_node and target_node.get("name"):
                lines.append(f"Căn cứ phù hợp nhất hệ thống tìm thấy: {ref_display} — {target_node.get('name')}.")
            elif article:
                lines.append(f"Căn cứ phù hợp nhất hệ thống tìm thấy: {ref_display} — {article.get('name', 'Không rõ')}.")
            else:
                lines.append(f"Căn cứ phù hợp nhất hệ thống tìm thấy: {ref_display}.")
        elif article:
            lines.append(f"Căn cứ phù hợp nhất hệ thống tìm thấy: {article.get('name', 'Không rõ')}.")
        else:
            lines.append("Hệ thống chưa xác định chắc chắn điều/khoản/điểm phù hợp từ dữ liệu KG/RAG hiện có.")

        if reference:
            lines.append(
                f"Mức truy xuất hiện tại: Điều {reference.get('article') or '?'}"
                + (f" → Khoản {reference.get('clause')}" if reference.get("clause") else "")
                + (f" → Điểm {reference.get('point')}" if reference.get("point") else "")
                + "."
            )

        # 2. Trả lời theo intent.
        if intent == "LOOKUP_ARTICLE" and article:
            lines.append("\nNội dung liên quan được truy xuất:")
            self._append_context_summary(lines, contexts, max_items=4)
        elif intent == "LOOKUP_PENALTY":
            if penalties:
                lines.append("\nThông tin hình phạt/hậu quả pháp lý tìm thấy:")
                display_penalties = self._filter_generic_penalties(penalties)
                for item in display_penalties[:5]:
                    lines.append(f"- {short_text(item.get('name', ''), 280)}")
            else:
                lines.append("\nChưa tìm thấy node hình phạt đủ rõ trong KG cho câu hỏi này.")
        elif intent == "CLASSIFY_CASE":
            lines.append("\nĐối chiếu nhanh với dữ kiện bạn nêu:")
            if matched_facts:
                for fact in matched_facts[:7]:
                    lines.append(f"- {fact}")
            else:
                lines.append("- Chưa nhận diện được đủ hành vi/hậu quả cụ thể từ câu hỏi.")
            if points:
                lines.append("\nĐiểm/tình tiết gần nhất trong KG:")
                for item in points[:4]:
                    lines.append(f"- {short_text(item.get('name', ''), 280)}")
            if penalties:
                lines.append("\nThông tin hình phạt liên quan trong KG:")
                for item in penalties[:4]:
                    lines.append(f"- {short_text(item.get('name', ''), 280)}")
        elif intent == "LOOKUP_CONDITIONS":
            if points:
                lines.append("\nĐiểm/tình tiết liên quan được tìm thấy:")
                for item in points[:6]:
                    lines.append(f"- {short_text(item.get('name', ''), 280)}")
            elif clauses:
                lines.append("\nKhoản/tình tiết liên quan được tìm thấy:")
                for item in clauses[:5]:
                    lines.append(f"- {short_text(item.get('name', ''), 280)}")
            self._append_context_summary(lines, contexts, max_items=3)
        else:
            if points:
                lines.append("\nĐiểm/tình tiết liên quan được truy xuất:")
                for item in points[:4]:
                    lines.append(f"- {short_text(item.get('name', ''), 280)}")
            lines.append("\nThông tin liên quan được truy xuất:")
            self._append_context_summary(lines, contexts, max_items=3)

        # 3. Căn cứ/evidence ngắn.
        if contexts:
            lines.append("\nEvidence dùng để suy ra kết quả:")
            display_contexts = self._display_contexts(contexts)
            for ctx in display_contexts[:4]:
                title = ctx.get("title", "Nguồn")
                score = ctx.get("score", 0)
                ctx_ref = ctx.get("reference") or {}
                ctx_display = ctx_ref.get("display") or ""
                prefix = f"{ctx_display} | " if ctx_display else ""
                lines.append(f"- {prefix}{title} | điểm: {score}")

        # 4. Cảnh báo khi thiếu dữ kiện.
        if missing_info:
            lines.append("\nThông tin còn thiếu để trả lời chắc hơn:")
            for item in missing_info[:5]:
                lines.append(f"- {item}")

        lines.append(f"\nĐộ tin cậy truy xuất: {confidence} ({confidence_label}).")
        if confidence < 0.45:
            lines.append("Vì độ tin cậy còn thấp, bạn nên hỏi cụ thể hơn, ví dụ nêu tên tội danh, điều luật, khoản, điểm, hành vi, hậu quả và tình tiết định khung.")

        if intent == "CLASSIFY_CASE":
            lines.append("\nLưu ý: với câu hỏi tình huống, kết quả chỉ là gợi ý căn cứ điều/khoản/điểm liên quan, không phải kết luận tội danh cuối cùng.")
        lines.append("Nội dung này chỉ hỗ trợ tra cứu, không thay thế tư vấn pháp lý chính thức.")
        return "\n".join(lines)


    def _display_contexts(self, contexts: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Ưu tiên hiển thị evidence cụ thể.

        Node Điều tổng quát vẫn có thể xuất hiện để làm bối cảnh, nhưng không được
        đứng trên Khoản/Điểm/Hình phạt khi các node cụ thể đã tồn tại.
        """
        if not contexts:
            return []
        specific = [
            c for c in contexts
            if c.get("node_kind") in {"point", "clause", "penalty", "other"}
        ]
        articles = [c for c in contexts if c.get("node_kind") == "article"]
        ordered = specific + articles
        return ordered or contexts

    def _filter_generic_penalties(self, penalties: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if len(penalties) <= 1:
            return penalties
        generic_phrases = ["tùy khoản", "tùy từng khoản", "tuỳ khoản", "tuỳ từng khoản"]
        filtered = []
        for item in penalties:
            name = str(item.get("name", "")).lower()
            if any(p in name for p in generic_phrases):
                continue
            filtered.append(item)
        return filtered or penalties

    def _append_context_summary(self, lines: List[str], contexts: List[Dict[str, object]], max_items: int = 3) -> None:
        if not contexts:
            lines.append("- Không có context phù hợp để tóm tắt.")
            return
        added = 0
        for ctx in contexts:
            text = ctx.get("content") or ctx.get("title") or ""
            if not text:
                continue
            ref = ctx.get("reference") or {}
            prefix = f"{ref.get('display')} — " if ref.get("display") else ""
            lines.append(f"- {prefix}{short_text(str(text), 300)}")
            added += 1
            if added >= max_items:
                break
        if added == 0:
            lines.append("- Không có context phù hợp để tóm tắt.")

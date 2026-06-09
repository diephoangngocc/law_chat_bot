from __future__ import annotations

from typing import Dict, List, Optional

from .text import short_text
from . import law_content as lc


class TemplateAnswerer:
    """Sinh câu trả lời không LLM theo cấu trúc cố định."""

    def generate(self, question: str, evidence: Dict[str, object], top_k: int = 10) -> str:
        article = evidence.get("article") or {}
        reference = evidence.get("reference") or {}
        target_node = evidence.get("target_node") or {}
        clauses: List[Dict] = evidence.get("clauses") or []
        points: List[Dict] = evidence.get("points") or []
        penalties: List[Dict] = evidence.get("penalties") or []
        matched_facts: List[str] = evidence.get("matched_facts") or []
        missing_info: List[str] = evidence.get("missing_info") or []
        contexts: List[Dict] = evidence.get("contexts") or []
        confidence = evidence.get("confidence", 0.0)
        confidence_label = evidence.get("confidence_label", "thấp")

        lines: List[str] = []

        # --- 1. Tình tiết truy xuất ---
        lines.append("## 🔍 Tình tiết truy xuất được")
        facts_shown = 0
        if matched_facts:
            for fact in matched_facts[:7]:
                lines.append(f"- {fact}")
                facts_shown += 1
        if facts_shown == 0 and contexts:
            for ctx in contexts[:4]:
                text = ctx.get("content") or ctx.get("title") or ""
                if text:
                    lines.append(f"- {short_text(str(text), 220)}")
                    facts_shown += 1
        if facts_shown == 0:
            lines.append("- Chưa nhận diện được tình tiết cụ thể từ câu hỏi.")

        # --- 2. Nhận định vi phạm (inline, không phải bullet) ---
        lines.append("\n## ⚖️ Nhận định vi phạm")
        violation_refs = self._build_violation_refs(
            reference, target_node, article, contexts, points, clauses, top_k=top_k
        )
        if violation_refs:
            lines.append(
                "Dựa vào các tình tiết trên, nhận thấy hành vi của chủ thể "
                "**có thể** đang vi phạm:"
            )
            for ref in violation_refs:
                lines.append(f"- {ref}")
        else:
            lines.append(
                "Chưa xác định rõ điều/khoản/điểm vi phạm từ dữ liệu hiện có."
            )

        # --- 3. Hình phạt dự kiến ---
        lines.append("\n## 🏛️ Hình phạt dự kiến")
        display_penalties = self._filter_generic_penalties(penalties)
        if display_penalties:
            lines.append("Hình phạt dự kiến của chủ thể là:")
            for p in display_penalties[:4]:
                name = p.get("name", "").strip()
                if name:
                    lines.append(f"- {short_text(name, 300)}")
        else:
            lines.append("- Chưa tìm thấy thông tin hình phạt cụ thể trong KG.")

        # --- 4. Thông tin còn thiếu ---
        if missing_info:
            lines.append("\n## ❗ Thông tin còn thiếu")
            for item in missing_info[:5]:
                lines.append(f"- {item}")

        # --- 5. Độ tin cậy & lưu ý ---
        lines.append("\n## 📌 Độ tin cậy & Lưu ý")
        pct = int(confidence * 100)
        lines.append(f"- Độ tin cậy: **{confidence_label}** ({pct}%)")
        if confidence < 0.45:
            lines.append(
                "- ⚠️ Độ tin cậy thấp — hãy cung cấp thêm thông tin: "
                "hành vi cụ thể, hậu quả, tên tội danh, tình tiết định khung."
            )
        lines.append(
            "- _Nội dung chỉ hỗ trợ tra cứu, không thay thế tư vấn pháp lý chính thức._"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_violation_refs(
        self,
        reference: Dict,
        target_node: Dict,
        article: Dict,
        contexts: List[Dict],
        points: List[Dict],
        clauses: List[Dict],
        top_k: int = 10,
    ) -> List[str]:
        """Trả về danh sách ref vi phạm.

        Quy tắc:
        - Nếu khoản có điểm → chỉ trích xuất tới điểm (bỏ qua ref cấp khoản của cùng khoản đó)
        - Nếu khoản không có điểm → trích xuất tới khoản
        """
        import re as _re

        def _parse(display: str):
            s = display or ""
            ma = _re.search(r"[Đđ]i[eề]u\s+(\d+[a-zA-Z]?)", s, _re.IGNORECASE)
            mc = _re.search(r"[Kk]ho[aả]n\s+(\d+)", s, _re.IGNORECASE)
            mp = _re.search(r"[Đđ]i[eể]m\s+([a-zđ])", s, _re.IGNORECASE)
            return (
                ma.group(1) if ma else None,
                mc.group(1) if mc else None,
                mp.group(1) if mp else None,
            )

        # --- Thu thập top_k contexts có score cao nhất ---
        # Contexts đã được sort theo score trong evidence_builder
        scored: List[tuple] = []   # (score, display, art, clause, point)
        seen_d: set = set()

        def _add(display: str, score: float = 0.0) -> None:
            d = (display or "").strip()
            if not d or d in seen_d:
                return
            seen_d.add(d)
            a, c, p = _parse(d)
            if a:
                scored.append((score, d, a, c, p))

        # Primary reference (score cao nhất)
        _add(reference.get("display") or "", score=1.0)

        # Contexts theo score thực tế
        for ctx in contexts:
            _add(
                (ctx.get("reference") or {}).get("display", ""),
                score=float(ctx.get("score") or 0.0),
            )

        # Sort theo score giảm dần, lấy top_k
        scored.sort(key=lambda x: -x[0])
        candidates: List[tuple] = [(d, a, c, p) for _, d, a, c, p in scored[:top_k]]

        if not candidates:
            # Fallback raw
            for item in (points + clauses)[:4]:
                name = (item.get("name") or "").strip()
                if name:
                    return [short_text(name, 200)]
            return []

        # --- Xác định (art, clause) nào đã có điểm trong candidates ---
        ac_with_point: set = set()
        for _, a, c, p in candidates:
            if a and c and p:
                ac_with_point.add((a, c))

        # --- Với khoản chưa có điểm trong evidence, tra law_content để mở rộng ---
        lc._load()
        expanded: List[tuple] = []
        for display, a, c, p in candidates:
            if a and c and not p and (a, c) not in ac_with_point:
                # Tìm điểm của khoản này trong law_content.json
                prefix = f"Điều {a} Khoản {c} Điểm "
                diem_keys = sorted(k for k in lc._CONTENT if k.startswith(prefix))
                if diem_keys:
                    # Mở rộng: thay khoản bằng các điểm của nó
                    for dk in diem_keys[:max(top_k, 6)]:
                        letter = dk[len(prefix):]
                        diem_display = f"Điểm {letter} Khoản {c} Điều {a}"
                        expanded.append((diem_display, a, c, letter))
                    ac_with_point.add((a, c))   # đánh dấu đã có điểm
                else:
                    expanded.append((display, a, c, None))
            else:
                expanded.append((display, a, c, p))

        # --- Lọc cuối: bỏ ref khoản nếu đã có điểm ---
        result: List[str] = []
        seen_r: set = set()
        for display, a, c, p in expanded:
            if a and c and not p and (a, c) in ac_with_point:
                continue
            if display in seen_r:
                continue
            seen_r.add(display)
            result.append(display)
            if len(result) >= top_k:
                break

        return result

    def _display_contexts(self, contexts: List[Dict]) -> List[Dict]:
        if not contexts:
            return []
        specific = [c for c in contexts if c.get("node_kind") in {"point", "clause", "penalty", "other"}]
        articles = [c for c in contexts if c.get("node_kind") == "article"]
        return (specific + articles) or contexts

    def _filter_generic_penalties(self, penalties: List[Dict]) -> List[Dict]:
        if len(penalties) <= 1:
            return penalties
        generic = ["tùy khoản", "tùy từng khoản", "tuỳ khoản", "tuỳ từng khoản"]
        filtered = [p for p in penalties if not any(g in str(p.get("name", "")).lower() for g in generic)]
        return filtered or penalties

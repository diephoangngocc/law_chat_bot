from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .graph import LegalKnowledgeGraph
from .models import Node
from .text import extract_article_mentions, normalize_text, short_text, unique_keep_order


COMPARE_KEYWORDS = ["khác nhau", "khác gì", "so sánh", "phân biệt", "giống nhau", "điểm khác", "điểm giống"]
LOOKUP_KEYWORDS = ["quy định gì", "nội dung", "là gì", "quy định", "nói gì", "gồm những gì", "trình bày"]


@dataclass
class ArticleSummary:
    number: str
    found: bool
    article: Optional[Dict[str, object]] = None
    clauses: List[Dict[str, object]] = None
    points: List[Dict[str, object]] = None
    penalties: List[Dict[str, object]] = None
    related: List[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "number": self.number,
            "found": self.found,
            "article": self.article,
            "clauses": self.clauses or [],
            "points": self.points or [],
            "penalties": self.penalties or [],
            "related": self.related or [],
        }


class ArticleAnswerer:
    """Trả lời nhanh các câu hỏi có Điều luật rõ ràng.

    Các câu như "Điều 123 quy định gì?" hoặc "Điều 123 và 124 khác nhau thế nào?"
    không nên đi qua retrieval mở rộng quá nhiều, vì rất dễ kéo evidence không liên quan
    và làm prompt LLM dài. Module này đọc trực tiếp từ KG theo số Điều.
    """

    def __init__(self, graph: LegalKnowledgeGraph):
        self.graph = graph

    def can_handle(self, question: str, semantic: Dict[str, object] | None = None) -> bool:
        numbers = self.article_numbers(question, semantic)
        if not numbers:
            return False
        q = normalize_text(question)
        if len(numbers) >= 2 and any(k in q for k in COMPARE_KEYWORDS):
            return True
        if len(numbers) == 1 and any(k in q for k in LOOKUP_KEYWORDS):
            return True
        # Chỉ hỏi "Điều 123" cũng coi như tra cứu nội dung điều.
        return len(numbers) == 1 and q.strip().startswith("điều ")

    def answer(self, question: str, semantic: Dict[str, object] | None = None) -> Tuple[str, Dict[str, object]]:
        numbers = self.article_numbers(question, semantic)
        q = normalize_text(question)
        if len(numbers) >= 2 and any(k in q for k in COMPARE_KEYWORDS):
            return self.compare(numbers[:2], question)
        if numbers:
            return self.lookup(numbers[0], question)
        return "Bạn hãy nêu rõ số điều cần tra cứu, ví dụ: Điều 123 quy định gì?", {
            "kind": "article_lookup",
            "article_numbers": [],
        }

    def article_numbers(self, question: str, semantic: Dict[str, object] | None = None) -> List[str]:
        numbers = list(extract_article_mentions(question))
        if semantic:
            entities = semantic.get("entities") or {}
            if isinstance(entities, dict):
                for raw in entities.get("ARTICLE", []) or []:
                    numbers.extend(extract_article_mentions(str(raw)))
            for raw in semantic.get("article_numbers", []) or []:
                numbers.append(str(raw))
        return unique_keep_order([str(n) for n in numbers if str(n).strip()])

    def lookup(self, number: str, question: str = "") -> Tuple[str, Dict[str, object]]:
        summary = self.summarize_article(number)
        evidence = {
            "kind": "article_lookup",
            "question": question,
            "article_numbers": [number],
            "articles": [summary.to_dict()],
        }

        if not summary.found:
            return (
                f"Mình chưa tìm thấy Điều {number} trong Knowledge Graph hiện có. "
                "Bạn kiểm tra lại số điều hoặc cập nhật dữ liệu KG rồi thử lại.",
                evidence,
            )

        article_name = summary.article.get("name", f"Điều {number}") if summary.article else f"Điều {number}"
        lines: List[str] = []
        lines.append(f"Điều {number} quy định về: {article_name}.")

        if summary.clauses:
            lines.append("\nCác khoản chính trong KG:")
            for item in summary.clauses[:8]:
                lines.append(f"- {short_text(str(item.get('name') or ''), 260)}")

        if summary.points:
            lines.append("\nCác điểm/tình tiết liên quan được ghi nhận:")
            for item in summary.points[:8]:
                lines.append(f"- {short_text(str(item.get('name') or ''), 260)}")

        if summary.penalties:
            lines.append("\nThông tin hình phạt/hậu quả pháp lý liên quan:")
            for item in self._filter_generic_penalties(summary.penalties)[:6]:
                lines.append(f"- {short_text(str(item.get('name') or ''), 260)}")

        if not summary.clauses and not summary.points and not summary.penalties:
            lines.append("\nKG hiện chỉ có node Điều, chưa có đủ Khoản/Điểm/Hình phạt để tóm tắt chi tiết hơn.")

        lines.append("\nLưu ý: Nội dung này được truy xuất từ KG hiện có, chỉ hỗ trợ tra cứu và không thay thế tư vấn pháp lý chính thức.")
        return "\n".join(lines), evidence

    def compare(self, numbers: List[str], question: str = "") -> Tuple[str, Dict[str, object]]:
        nums = unique_keep_order(numbers)[:2]
        summaries = [self.summarize_article(n) for n in nums]
        evidence = {
            "kind": "article_compare",
            "question": question,
            "article_numbers": nums,
            "articles": [s.to_dict() for s in summaries],
        }

        if len(summaries) < 2:
            return "Bạn hãy nêu đủ 2 điều luật cần so sánh, ví dụ: Điều 123 và 124 khác nhau như thế nào?", evidence

        missing = [s.number for s in summaries if not s.found]
        if missing:
            return (
                f"Mình chưa tìm thấy Điều {', Điều '.join(missing)} trong Knowledge Graph hiện có. "
                "Bạn kiểm tra lại số điều hoặc cập nhật dữ liệu KG rồi thử lại.",
                evidence,
            )

        a, b = summaries[0], summaries[1]
        lines: List[str] = []
        lines.append(f"So sánh Điều {a.number} và Điều {b.number} theo dữ liệu KG hiện có:")
        lines.append("")
        lines.append(f"1. Điều {a.number}: {a.article.get('name') if a.article else 'Không rõ tên điều'}")
        self._append_short_article_details(lines, a)
        lines.append("")
        lines.append(f"2. Điều {b.number}: {b.article.get('name') if b.article else 'Không rõ tên điều'}")
        self._append_short_article_details(lines, b)

        lines.append("\nĐiểm khác nhau chính:")
        name_a = str(a.article.get("name") if a.article else "")
        name_b = str(b.article.get("name") if b.article else "")
        if name_a and name_b and name_a != name_b:
            lines.append(f"- Tên/căn cứ: Điều {a.number} là “{short_text(name_a, 180)}”, còn Điều {b.number} là “{short_text(name_b, 180)}”.")

        clause_a = self._first_names(a.clauses, 3)
        clause_b = self._first_names(b.clauses, 3)
        if clause_a or clause_b:
            lines.append(f"- Cấu trúc khoản: Điều {a.number} có các khoản nổi bật: {self._join_or_none(clause_a)}; Điều {b.number} có các khoản nổi bật: {self._join_or_none(clause_b)}.")

        point_a = self._first_names(a.points, 3)
        point_b = self._first_names(b.points, 3)
        if point_a or point_b:
            lines.append(f"- Điểm/tình tiết: Điều {a.number}: {self._join_or_none(point_a)}; Điều {b.number}: {self._join_or_none(point_b)}.")

        penalty_a = self._first_names(self._filter_generic_penalties(a.penalties), 2)
        penalty_b = self._first_names(self._filter_generic_penalties(b.penalties), 2)
        if penalty_a or penalty_b:
            lines.append(f"- Hình phạt trong KG: Điều {a.number}: {self._join_or_none(penalty_a)}; Điều {b.number}: {self._join_or_none(penalty_b)}.")

        lines.append("\nLưu ý: Đây là so sánh theo các node/edge trong KG hiện có. Nếu KG thiếu khoản, điểm hoặc hình phạt, phần so sánh có thể chưa đầy đủ.")
        return "\n".join(lines), evidence

    def summarize_article(self, number: str) -> ArticleSummary:
        article_nodes = self.graph.get_article_nodes(str(number))
        if not article_nodes:
            return ArticleSummary(number=str(number), found=False, clauses=[], points=[], penalties=[], related=[])

        article = article_nodes[0]
        family = self.graph.article_family(article.id, depth=5)
        clauses: List[Node] = []
        points: List[Node] = []
        penalties: List[Node] = []
        related: List[Node] = []
        seen = set()

        for node in family:
            if node.id == article.id or node.id in seen:
                continue
            seen.add(node.id)
            # Kiểm tra penalty trước clause, vì một số node hình phạt có chữ "khoản"
            # trong nội dung, ví dụ "Hình phạt: tùy khoản áp dụng".
            if self.graph.is_penalty_node(node):
                penalties.append(node)
            elif self.graph.is_clause_node(node):
                clauses.append(node)
            elif self.graph.is_point_node(node):
                points.append(node)
            else:
                related.append(node)

        clauses.sort(key=lambda n: (self.graph.clause_number_of_node(n) or "99", n.name))
        points.sort(key=lambda n: (self.graph.clause_number_of_node(n) or "99", self.graph.point_letter_of_node(n) or "z", n.name))

        return ArticleSummary(
            number=str(number),
            found=True,
            article=self._node_to_dict(article),
            clauses=[self._node_to_dict(n) for n in clauses[:12]],
            points=[self._node_to_dict(n) for n in points[:16]],
            penalties=[self._node_to_dict(n) for n in penalties[:10]],
            related=[self._node_to_dict(n) for n in related[:10]],
        )

    def _append_short_article_details(self, lines: List[str], summary: ArticleSummary) -> None:
        clauses = self._first_names(summary.clauses, 3)
        points = self._first_names(summary.points, 3)
        penalties = self._first_names(self._filter_generic_penalties(summary.penalties), 2)
        if clauses:
            lines.append(f"   - Khoản chính: {self._join_or_none(clauses)}.")
        if points:
            lines.append(f"   - Điểm/tình tiết: {self._join_or_none(points)}.")
        if penalties:
            lines.append(f"   - Hình phạt: {self._join_or_none(penalties)}.")
        if not clauses and not points and not penalties:
            lines.append("   - KG chưa có đủ node Khoản/Điểm/Hình phạt để tóm tắt chi tiết.")

    def _node_to_dict(self, node: Node) -> Dict[str, object]:
        ref = self.graph.reference_of_node(node.id)
        return {
            "id": node.id,
            "name": node.name,
            "label": node.label,
            "reference": ref,
            "metadata": node.metadata,
        }

    def _first_names(self, items: List[Dict[str, object]] | None, limit: int) -> List[str]:
        out: List[str] = []
        for item in items or []:
            name = str(item.get("name") or "").strip()
            if name:
                out.append(short_text(name, 150))
            if len(out) >= limit:
                break
        return out

    def _join_or_none(self, items: List[str]) -> str:
        return "; ".join(items) if items else "chưa có dữ liệu rõ"

    def _filter_generic_penalties(self, penalties: List[Dict[str, object]] | None) -> List[Dict[str, object]]:
        penalties = penalties or []
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

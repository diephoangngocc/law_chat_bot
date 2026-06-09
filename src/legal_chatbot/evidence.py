from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from .graph import LegalKnowledgeGraph
from .models import Candidate, Node
from .text import (
    article_number,
    clause_number,
    legal_reference_display,
    normalize_text,
    point_letter,
    short_text,
    unique_keep_order,
)


class EvidenceBuilder:
    def __init__(self, graph: LegalKnowledgeGraph):
        self.graph = graph

    def build(self, question: str, semantic: Dict[str, object], candidates: List[Candidate], top_k: int = 10) -> Dict[str, object]:
        explicit_ref = self._explicit_reference(semantic, question)
        best_ref = self._best_reference(semantic, candidates, explicit_ref)

        best_article = self.graph.nodes.get(str(best_ref.get("article_id") or "")) if best_ref else None
        target_node_id = str(best_ref.get("target_node_id") or "") if best_ref else None
        target_node = self.graph.nodes.get(target_node_id) if target_node_id else None

        penalties = self._penalties(best_ref, best_article, candidates)
        clauses = self._clauses(best_article, candidates, best_ref)
        points = self._points(best_article, candidates, best_ref)
        contexts = self._contexts(best_ref, best_article, candidates, intent=str(semantic.get("intent", "UNKNOWN")), top_k=top_k)
        kg_paths = self._paths(candidates)
        facts = self._matched_facts(semantic)
        confidence = self._confidence(candidates, best_ref)
        missing_info = self._missing_info(semantic, bool(best_article), confidence)

        clause_cap = max(top_k, 8)
        point_cap = max(top_k * 2, 10)

        return {
            "question": question,
            "intent": semantic.get("intent"),
            "reference": self._reference_to_dict(best_ref, target_node),
            "article": self._node_to_dict(best_article),
            "target_node": self._node_to_dict(target_node),
            "clauses": [self._node_to_dict(c) for c in clauses[:clause_cap]],
            "points": [self._node_to_dict(p) for p in points[:point_cap]],
            "penalties": [self._node_to_dict(p) for p in penalties[:8]],
            "matched_facts": facts,
            "missing_info": missing_info,
            "confidence": confidence,
            "confidence_label": self._confidence_label(confidence),
            "kg_paths": kg_paths[:10],
            "contexts": contexts,
        }

    def _explicit_reference(self, semantic: Dict[str, object], question: str) -> Dict[str, Optional[str]]:
        entities = semantic.get("entities", {}) or {}
        article = article_number(question)
        clause = clause_number(question)
        point = point_letter(question)
        if isinstance(entities, dict):
            for v in entities.get("ARTICLE", []) or []:
                article = article or article_number(str(v))
            for v in entities.get("CLAUSE", []) or []:
                clause = clause or clause_number(str(v))
            for v in entities.get("POINT", []) or []:
                point = point or point_letter(str(v))
        return {"article": article, "clause": clause, "point": point}

    def _best_reference(
        self,
        semantic: Dict[str, object],
        candidates: List[Candidate],
        explicit_ref: Dict[str, Optional[str]],
    ) -> Dict[str, object]:
        """Chọn căn cứ ở mức cụ thể nhất có thể: Điểm > Khoản > Điều.

        Bản cũ aggregate score về Điều, nên thường dừng ở Khoản. Bản này gắn mỗi
        candidate với reference gần nhất rồi aggregate theo key Điểm/Khoản/Điều.
        """
        # 1) Nếu user hỏi rõ Điều/Khoản/Điểm, ưu tiên strict matching ở đủ cấp đã hỏi.
        if explicit_ref.get("article"):
            nodes = self.graph.get_article_nodes(str(explicit_ref["article"]))
            # Hỏi đúng "Điều X quy định gì" mà không nêu khoản/điểm thì dừng ở Điều,
            # không tự kéo xuống khoản chỉ vì khoản có điểm retrieval cao hơn.
            if nodes and not explicit_ref.get("clause") and not explicit_ref.get("point") and semantic.get("intent") == "LOOKUP_ARTICLE":
                return self.graph.reference_of_node(nodes[0].id)

            strict = self._best_reference_from_candidates(candidates, explicit_ref=explicit_ref, strict=True)
            if strict:
                return strict
            # Nếu KG có Điều nhưng chưa thấy candidate cụ thể, vẫn trả Điều để không lạc.
            if nodes:
                base = self.graph.reference_of_node(nodes[0].id)
                if explicit_ref.get("clause"):
                    base["clause"] = explicit_ref.get("clause")
                    base["display"] = legal_reference_display(base.get("article"), base.get("clause"), base.get("point"))
                if explicit_ref.get("point"):
                    base["point"] = explicit_ref.get("point")
                    base["display"] = legal_reference_display(base.get("article"), base.get("clause"), base.get("point"))
                return base

        # 2) Hinted article như tử vong -> Điều 123. Nhưng vẫn cho candidate điểm/khoản cụ thể thắng nếu cùng điều.
        hinted = list(semantic.get("hinted_articles", []) or [])
        if hinted:
            hinted_ref = self._best_reference_from_candidates(
                candidates,
                explicit_ref={"article": None, "clause": None, "point": None},
                preferred_articles=set(str(x) for x in hinted),
                strict=False,
            )
            if hinted_ref:
                return hinted_ref

        # 3) Aggregate score theo reference cụ thể nhất.
        ref = self._best_reference_from_candidates(candidates, explicit_ref=explicit_ref, strict=False)
        if ref:
            return ref
        return {}

    def _best_reference_from_candidates(
        self,
        candidates: List[Candidate],
        explicit_ref: Dict[str, Optional[str]],
        preferred_articles: Optional[set[str]] = None,
        strict: bool = False,
    ) -> Dict[str, object]:
        ref_scores: Dict[str, float] = defaultdict(float)
        ref_rows: Dict[str, Dict[str, object]] = {}
        preferred_articles = preferred_articles or set()

        for rank, cand in enumerate(candidates[:20]):
            node_id = cand.document.node_id
            if not node_id:
                continue
            ref = self.graph.reference_of_node(node_id)
            if not ref.get("article"):
                continue

            # Nếu explicit có cấp nào thì cấp đó phải khớp khi strict=True.
            if strict:
                if explicit_ref.get("article") and ref.get("article") != explicit_ref.get("article"):
                    continue
                if explicit_ref.get("clause") and ref.get("clause") != explicit_ref.get("clause"):
                    continue
                if explicit_ref.get("point") and ref.get("point") != explicit_ref.get("point"):
                    continue

            # Nếu có preferred article, loại nhánh khác để tránh câu hỏi tử vong trôi sang điều khác.
            if preferred_articles and str(ref.get("article")) not in preferred_articles:
                continue

            key = str(ref.get("key") or node_id)
            node = self.graph.nodes.get(node_id)
            specificity = int(ref.get("specificity") or 0)
            score = float(cand.score) + 1.0 / (rank + 1) + 0.28 * specificity
            if node:
                if self.graph.is_point_node(node):
                    score += 0.85
                elif self.graph.is_clause_node(node):
                    score += 0.55
                elif self.graph.is_article_node(node):
                    score += 0.25
                elif self.graph.is_penalty_node(node):
                    score += 0.25
            score += 0.35 * float(cand.score_parts.get("ref_point", 0.0))
            score += 0.25 * float(cand.score_parts.get("ref_clause", 0.0))
            score += 0.15 * float(cand.score_parts.get("ref_article", 0.0))
            if str(ref.get("article")) in preferred_articles:
                score += 0.5
            ref_scores[key] += score
            # Giữ reference cụ thể hơn nếu key trùng/score tốt hơn.
            if key not in ref_rows or int(ref.get("specificity") or 0) >= int(ref_rows[key].get("specificity") or 0):
                ref_rows[key] = ref

        if not ref_scores:
            return {}
        best_key = max(ref_scores.items(), key=lambda item: item[1])[0]
        best = dict(ref_rows[best_key])
        best["aggregate_score"] = round(float(ref_scores[best_key]), 4)
        return best

    def _clauses(self, article: Optional[Node], candidates: List[Candidate], reference: Dict[str, object]) -> List[Node]:
        out: List[Node] = []
        seen = set()
        target_clause = str(reference.get("clause") or "")
        if article:
            for node in self.graph.article_family(article.id, depth=4):
                if self.graph.is_clause_node(node) and node.id not in seen:
                    seen.add(node.id)
                    out.append(node)
        for cand in candidates:
            if cand.document.node_id:
                node = self.graph.nodes.get(cand.document.node_id)
                if node and self.graph.is_clause_node(node) and node.id not in seen:
                    seen.add(node.id)
                    out.append(node)
        out.sort(key=lambda n: (self.graph.clause_number_of_node(n) != target_clause if target_clause else False, n.name))
        return out

    def _points(self, article: Optional[Node], candidates: List[Candidate], reference: Dict[str, object]) -> List[Node]:
        out: List[Node] = []
        seen = set()
        target_clause = str(reference.get("clause") or "")
        target_point = str(reference.get("point") or "")
        if article:
            for node in self.graph.article_family(article.id, depth=5):
                if self.graph.is_point_node(node) and node.id not in seen:
                    seen.add(node.id)
                    out.append(node)
        for cand in candidates:
            if cand.document.node_id:
                node = self.graph.nodes.get(cand.document.node_id)
                if node and self.graph.is_point_node(node) and node.id not in seen:
                    seen.add(node.id)
                    out.append(node)

        def sort_key(n: Node):
            ref = self.graph.reference_of_node(n.id)
            return (
                ref.get("point") != target_point if target_point else False,
                ref.get("clause") != target_clause if target_clause else False,
                n.name,
            )
        out.sort(key=sort_key)
        return out

    def _penalties(self, reference: Dict[str, object], article: Optional[Node], candidates: List[Candidate]) -> List[Node]:
        penalty_nodes: List[Node] = []
        family_ids = set()
        target = str(reference.get("target_node_id") or "") if reference else ""
        if target and target in self.graph.nodes:
            penalty_nodes.extend(self.graph.find_penalties_near(target, depth=3))
        if reference.get("clause_id"):
            penalty_nodes.extend(self.graph.find_penalties_near(str(reference["clause_id"]), depth=3))
        if article:
            family_ids = {node.id for node in self.graph.article_family(article.id, depth=5)}
            penalty_nodes.extend(self.graph.find_penalties_near(article.id, depth=5))
        for cand in candidates:
            if cand.document.node_id:
                node = self.graph.nodes.get(cand.document.node_id)
                if node and self.graph.is_penalty_node(node):
                    if not article or node.id in family_ids:
                        penalty_nodes.append(node)
        seen = set()
        out = []
        for node in penalty_nodes:
            if node.id not in seen:
                seen.add(node.id)
                out.append(node)
        return out

    def _contexts(self, reference: Dict[str, object], article: Optional[Node], candidates: List[Candidate], intent: str = "UNKNOWN", top_k: int = 10) -> List[Dict[str, object]]:
        family_ids = set()
        family_nodes: List[Node] = []
        if article:
            family_nodes = self.graph.article_family(article.id, depth=5)
            family_ids = {node.id for node in family_nodes}
            family_ids.add(article.id)

        rows: List[Dict[str, object]] = []
        seen = set()

        family_cap = max(top_k * 3, 24)
        for idx, node in enumerate(family_nodes[:family_cap]):
            seen.add(node.id)
            ref = self.graph.reference_of_node(node.id)
            same_point = bool(reference.get("point") and self.graph.same_reference(node.id, reference, level="point"))
            same_clause = bool(reference.get("clause") and self.graph.same_reference(node.id, reference, level="clause"))
            same_article = bool(reference.get("article") and ref.get("article") == reference.get("article"))
            base_score = max(0.92 - idx * 0.025, 0.35)
            if same_point:
                base_score += 0.45
            elif same_clause:
                base_score += 0.28
            elif same_article:
                base_score += 0.12
            rows.append(
                {
                    "title": node.name,
                    "content": short_text(node.text, 900),
                    "source": "kg_reference_family",
                    "score": round(base_score, 4),
                    "node_id": node.id,
                    "label": node.label,
                    "reference": ref,
                    "same_point": same_point,
                    "same_clause": same_clause,
                    "same_article_family": same_article,
                }
            )

        for cand in candidates:
            node_id = cand.document.node_id
            if node_id in seen:
                for row in rows:
                    if row.get("node_id") == node_id:
                        row["score"] = max(float(row.get("score", 0)), round(float(cand.score), 4))
                        break
                continue
            ref = self.graph.reference_of_node(node_id) if node_id else {}
            same_family = bool(node_id and node_id in family_ids) if family_ids else True
            if family_ids and not same_family and len([r for r in rows if not r.get("same_article_family")]) >= 2:
                continue
            rows.append(
                {
                    "title": cand.document.title,
                    "content": short_text(cand.document.content, 900),
                    "source": cand.document.source,
                    "score": round(float(cand.score), 4),
                    "node_id": node_id,
                    "label": cand.document.metadata.get("label", ""),
                    "reference": ref,
                    "same_point": bool(reference.get("point") and node_id and self.graph.same_reference(node_id, reference, level="point")),
                    "same_clause": bool(reference.get("clause") and node_id and self.graph.same_reference(node_id, reference, level="clause")),
                    "same_article_family": same_family,
                }
            )
        # Chuẩn hóa lại điểm evidence theo cấp pháp lý.
        # Lưu ý: điểm ở evidence KHÔNG nên dùng nguyên score retrieval, vì node Điều
        # thường match tên tội danh rất mạnh nên dễ cao hơn Khoản/Điểm. Ở đây ta dùng
        # score hiển thị theo độ cụ thể: Điểm > Khoản > Hình phạt/tình tiết > Điều.
        rows = self._normalize_context_scores(rows, reference, intent)
        rows.sort(
            key=lambda r: (
                int(r.get("evidence_priority", 99)),
                -int((r.get("reference") or {}).get("specificity") or 0),
                -float(r.get("score", 0)),
                str(r.get("title", "")),
            )
        )
        ctx_cap = max(top_k * 2, 18)
        return rows[:ctx_cap]

    def _normalize_context_scores(self, rows: List[Dict[str, object]], reference: Dict[str, object], intent: str) -> List[Dict[str, object]]:
        """Tính điểm evidence theo scope pháp lý, không để node Điều lấn Khoản/Điểm.

        Retrieval score vẫn được giữ ở `raw_score`. Field `score` dùng để hiển thị evidence.
        Mục tiêu: khi đã có các node Khoản/Điểm trong cùng Điều, chúng phải đứng trên
        node Điều tổng quát. Node Điều chỉ là context cha/supporting context.
        """
        normalized: List[Dict[str, object]] = []
        for row in rows:
            node_id = str(row.get("node_id") or "")
            node = self.graph.nodes.get(node_id)
            raw_score = float(row.get("score", 0) or 0)
            kind = self._node_kind(node)
            ref = row.get("reference") or {}
            specificity = int(ref.get("specificity") or 0)

            same_point = bool(row.get("same_point"))
            same_clause = bool(row.get("same_clause"))
            same_article = bool(row.get("same_article_family"))

            # Priority nhỏ hơn sẽ đứng trước.
            if reference.get("point") and same_point:
                priority = 0
            elif reference.get("clause") and same_clause and kind in {"point", "clause", "penalty", "other"}:
                priority = 1
            elif intent == "LOOKUP_PENALTY" and kind == "clause":
                priority = 2
            elif intent == "LOOKUP_PENALTY" and kind == "point":
                priority = 3
            elif kind == "point":
                priority = 4
            elif kind == "clause":
                priority = 5
            elif kind == "penalty":
                priority = 6
            elif kind == "article":
                priority = 9
            else:
                priority = 7

            # Base theo cấp: Điều tổng quát bị giới hạn thấp hơn Khoản/Điểm.
            if kind == "point":
                base = 1.18
                cap = 1.45
            elif kind == "clause":
                base = 1.08
                cap = 1.32
            elif kind == "penalty":
                base = 0.98
                cap = 1.16
                # Hình phạt gắn chung Điều, chưa rõ Khoản, chỉ để hỗ trợ nên không cao hơn Khoản.
                if specificity <= 1 and not reference.get("clause"):
                    cap = 0.96
            elif kind == "article":
                base = 0.72
                cap = 0.86
            else:
                base = 0.88
                cap = 1.08

            if same_point:
                base += 0.18
            elif same_clause:
                base += 0.12
            elif same_article:
                base += 0.04

            # raw_score chỉ góp một phần nhỏ để không đảo thứ tự cấp pháp lý.
            display_score = min(base + min(raw_score, 2.0) * 0.08, cap)

            row["raw_score"] = round(raw_score, 4)
            row["score"] = round(display_score, 4)
            row["node_kind"] = kind
            row["evidence_priority"] = priority
            normalized.append(row)

        # Nếu đã có context cụ thể hơn trong cùng Điều, giảm vai trò các Điều tổng quát.
        has_specific = any(r.get("node_kind") in {"point", "clause", "penalty"} for r in normalized)
        if has_specific:
            for row in normalized:
                if row.get("node_kind") == "article":
                    row["score"] = min(float(row.get("score", 0)), 0.78)
                    row["evidence_priority"] = max(int(row.get("evidence_priority", 99)), 10)
        return normalized

    def _node_kind(self, node: Optional[Node]) -> str:
        if not node:
            return "other"
        if self.graph.is_point_node(node):
            return "point"
        if self.graph.is_clause_node(node):
            return "clause"
        if self.graph.is_article_node(node):
            return "article"
        if self.graph.is_penalty_node(node):
            return "penalty"
        return "other"

    def _paths(self, candidates: List[Candidate]) -> List[str]:
        paths: List[str] = []
        for cand in candidates:
            paths.extend(cand.kg_paths)
        return unique_keep_order(paths)

    def _matched_facts(self, semantic: Dict[str, object]) -> List[str]:
        entities = semantic.get("entities", {}) or {}
        facts = []
        labels = {
            "ARTICLE": "Điều luật được hỏi",
            "CLAUSE": "Khoản",
            "POINT": "Điểm",
            "CRIME": "Tội danh/cụm tội danh",
            "ACTION": "Hành vi",
            "RESULT": "Hậu quả",
            "OBJECT": "Khách thể/đối tượng",
            "CONDITION": "Tình tiết/điều kiện",
            "PENALTY": "Yêu cầu về hình phạt",
            "AGGRAVATING": "Tình tiết tăng nặng (Điều 52)",
            "MITIGATING": "Tình tiết giảm nhẹ (Điều 51)",
            "INTENT": "Lỗi/ý thức chủ quan",
        }
        if isinstance(entities, dict):
            for key, name in labels.items():
                values = entities.get(key) or []
                if values:
                    facts.append(f"{name}: {', '.join(values)}")
        domains = semantic.get("domains", []) or []
        if domains:
            domain_vi = {
                "homicide": "nhóm xâm phạm tính mạng/giết người",
                "injury": "nhóm cố ý gây thương tích",
                "robbery": "nhóm cướp tài sản",
                "theft": "nhóm trộm cắp tài sản",
                "fraud": "nhóm lừa đảo chiếm đoạt tài sản",
                "drugs": "nhóm ma túy",
                "trust_abuse": "nhóm lạm dụng tín nhiệm chiếm đoạt tài sản (Điều 175)",
                "forest_destruction": "nhóm hủy hoại rừng (Điều 243)",
            }
            facts.append("Nhóm vấn đề pháp lý gợi ý: " + ", ".join(domain_vi.get(d, d) for d in domains))
        return facts

    def _missing_info(self, semantic: Dict[str, object], has_article: bool, confidence: float) -> List[str]:
        intent = str(semantic.get("intent", "UNKNOWN"))
        entities = semantic.get("entities", {}) or {}
        missing = []
        if not has_article or confidence < 0.35:
            missing.append(
                "Chưa xác định đủ chắc điều/khoản/điểm phù hợp trong KG; "
                "nên bổ sung thêm tình tiết hoặc tên tội danh/điều luật."
            )
        if intent == "CLASSIFY_CASE":
            # Mặt khách quan: hành vi + hậu quả.
            if not entities.get("ACTION"):
                missing.append("Hành vi cụ thể của người thực hiện (mặt khách quan).")
            if not entities.get("RESULT"):
                missing.append("Hậu quả cụ thể của hành vi (thiệt hại vật chất, tính mạng, tài sản...).")
            # Chủ thể: ai thực hiện hành vi.
            if not entities.get("OBJECT"):
                missing.append(
                    "Chủ thể thực hiện hành vi: người phạm tội là ai, độ tuổi, "
                    "năng lực trách nhiệm hình sự, có phải người có chức vụ quyền hạn không?"
                )
            # Mặt chủ quan: lỗi cố ý/vô ý, mục đích.
            if not entities.get("INTENT"):
                missing.append(
                    "Mặt chủ quan (lỗi): hành vi thực hiện với lỗi cố ý hay vô ý? "
                    "Mục đích, động cơ của người phạm tội là gì?"
                )
            # Khách thể pháp lý: quan hệ xã hội bị xâm phạm.
            if not entities.get("CONDITION"):
                missing.append(
                    "Khách thể/tình tiết định khung: công cụ phương tiện sử dụng, "
                    "số lượng/giá trị tài sản, số người bị hại, địa bàn, "
                    "có tổ chức hay không, tái phạm nguy hiểm, đối tượng tác động..."
                )
            # Tình tiết tăng nặng/giảm nhẹ.
            if not entities.get("AGGRAVATING"):
                missing.append(
                    "Tình tiết tăng nặng trách nhiệm hình sự (Điều 52): "
                    "phạm tội có tổ chức, tái phạm, lợi dụng chức vụ, "
                    "phạm tội đối với người dưới 16 tuổi, thủ đoạn tinh vi...?"
                )
            if not entities.get("MITIGATING"):
                missing.append(
                    "Tình tiết giảm nhẹ trách nhiệm hình sự (Điều 51): "
                    "thành khẩn khai báo, tự nguyện bồi thường, phạm tội lần đầu, "
                    "hoàn cảnh đặc biệt khó khăn, đầu thú...?"
                )
        if intent == "LOOKUP_PENALTY" and not entities.get("CRIME") and not entities.get("ARTICLE") and not semantic.get("hinted_articles"):
            missing.append("Tên tội danh hoặc điều luật cụ thể để xác định khung hình phạt.")
        return unique_keep_order(missing)

    def _confidence(self, candidates: List[Candidate], reference: Dict[str, object]) -> float:
        if not candidates:
            return 0.0
        top = max(float(c.score) for c in candidates[:3])
        base = min(top / 2.4, 1.0)
        if reference.get("article"):
            base = min(base + 0.05, 1.0)
        if reference.get("clause"):
            base = min(base + 0.05, 1.0)
        if reference.get("point"):
            base = min(base + 0.06, 1.0)
        return round(base, 3)

    def _confidence_label(self, confidence: float) -> str:
        if confidence >= 0.75:
            return "cao"
        if confidence >= 0.45:
            return "trung bình"
        return "thấp"

    def _reference_to_dict(self, reference: Dict[str, object], target_node: Optional[Node]) -> Dict[str, object]:
        if not reference:
            return {}
        return {
            "article": reference.get("article"),
            "clause": reference.get("clause"),
            "point": reference.get("point"),
            "display": reference.get("display") or legal_reference_display(reference.get("article"), reference.get("clause"), reference.get("point")),
            "key": reference.get("key"),
            "specificity": reference.get("specificity"),
            "aggregate_score": reference.get("aggregate_score"),
            "target_node_id": reference.get("target_node_id"),
            "target_node_name": target_node.name if target_node else None,
        }

    def _node_to_dict(self, node: Optional[Node]) -> Optional[Dict[str, object]]:
        if node is None:
            return None
        ref = self.graph.reference_of_node(node.id)
        return {
            "id": node.id,
            "name": node.name,
            "label": node.label,
            "article_number": self.graph.article_number_of_node(node),
            "clause_number": self.graph.clause_number_of_node(node),
            "point_letter": self.graph.point_letter_of_node(node),
            "reference": ref,
            "metadata": node.metadata,
        }

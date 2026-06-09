from __future__ import annotations

import math
from typing import Dict, List, Optional, TYPE_CHECKING

from .models import Candidate
from .text import (
    article_number,
    clause_number,
    lexical_similarity,
    normalize_text,
    phrase_count,
    point_letter,
    strip_accents,
    tokenize,
)

if TYPE_CHECKING:  # pragma: no cover
    from .graph import LegalKnowledgeGraph

DOMAIN_POSITIVE = {
    "homicide": ["giết người", "tước đoạt", "tính mạng", "chết", "làm chết người", "điều 123"],
    "injury": ["cố ý gây thương tích", "thương tích", "tổn hại sức khỏe", "điều 134"],
    "robbery": ["cướp tài sản", "dùng vũ lực", "đe dọa dùng vũ lực", "chiếm đoạt tài sản", "điều 168"],
    "theft": ["trộm cắp tài sản", "trộm", "chiếm đoạt tài sản", "điều 173"],
    "fraud": ["lừa đảo chiếm đoạt tài sản", "gian dối", "chiếm đoạt tài sản", "điều 174"],
    "drugs": ["ma túy", "ma tuý", "chất ma túy", "tàng trữ", "mua bán", "vận chuyển"],
}

# Một số cụm dễ làm retrieval đi lệch khi chỉ match "hậu quả nghiêm trọng".
DOMAIN_NEGATIVE = {
    "homicide": ["lật đổ chính quyền", "an ninh quốc gia", "hoạt động nhằm", "gián điệp", "khủng bố"],
    "robbery": ["giết người", "ma túy", "an ninh quốc gia"],
}


class LegalReranker:
    def __init__(self, graph: Optional["LegalKnowledgeGraph"] = None):
        self.graph = graph

    def rank(self, question: str, semantic: Dict[str, object], candidates: List[Candidate], top_k: int = 5) -> List[Candidate]:
        intent = str(semantic.get("intent", "UNKNOWN"))
        entities = semantic.get("entities", {}) or {}
        query_terms = list(semantic.get("query_terms", []) or [])
        domains = list(semantic.get("domains", []) or [])
        hinted_articles = set(str(x) for x in (semantic.get("hinted_articles", []) or []))
        q_norm = normalize_text(question)
        q_tokens = set(tokenize(question))
        explicit_article = article_number(q_norm)
        explicit_clause = self._first_entity_number(entities, "CLAUSE", fallback=clause_number(q_norm))
        explicit_point = self._first_entity_point(entities, fallback=point_letter(q_norm))

        entity_values: List[str] = []
        if isinstance(entities, dict):
            for values in entities.values():
                entity_values.extend(values)

        phrases = list(dict.fromkeys([*entity_values, *query_terms]))

        for cand in candidates:
            doc_text = normalize_text(f"{cand.document.title} {cand.document.content}")
            doc_no_acc = strip_accents(doc_text)
            doc_tokens = set(tokenize(doc_text))

            bm25_score = float(cand.score_parts.get("bm25", 0.0))
            kg_score = float(cand.score_parts.get("kg", 0.0))

            overlap = len(q_tokens & doc_tokens) / max(len(q_tokens), 1) if q_tokens else 0.0
            entity_score = self._entity_score(entity_values, doc_text, doc_no_acc)
            phrase_score = min(phrase_count(doc_text, phrases) / max(len(phrases), 1), 1.0) if phrases else 0.0
            article_score = self._article_score(explicit_article, hinted_articles, cand, doc_text)
            reference_score, reference_parts = self._reference_score(explicit_article, explicit_clause, explicit_point, cand, doc_text)
            intent_score = self._intent_score(intent, doc_text)
            domain_score = self._domain_score(domains, doc_text)
            similarity = lexical_similarity(q_norm, doc_text)

            label = normalize_text(str(cand.document.metadata.get("label", "")))
            label_boost = 0.0
            if "điều" in label:
                label_boost += 0.18
            if "khoản" in label:
                label_boost += 0.22
            if "điểm" in label:
                label_boost += 0.30
            if intent == "LOOKUP_PENALTY" and any(k in doc_text for k in ["hình phạt", "phạt tù", "tử hình", "chung thân", "phạt tiền"]):
                label_boost += 0.35
            if intent == "LOOKUP_ARTICLE" and "điều" in doc_text:
                label_boost += 0.22

            final = (
                0.18 * self._normalize_bm25(bm25_score)
                + 0.14 * min(kg_score / 8.0, 1.0)
                + 0.15 * entity_score
                + 0.12 * phrase_score
                + 0.08 * overlap
                + 0.08 * max(intent_score, article_score)
                + 0.18 * reference_score
                + 0.12 * domain_score
                + 0.05 * min(similarity, 1.0)
                + label_boost
            )

            # Nếu user hỏi rõ Điểm/Khoản/Điều thì reference_score phải thắng mạnh.
            if reference_score >= 1.0:
                final += 0.75
            elif reference_score >= 0.75:
                final += 0.45
            elif article_score >= 1.0:
                final += 0.35
            elif article_score >= 0.6:
                final += 0.18

            # Penalize tài liệu lệch miền rõ ràng.
            final -= self._negative_domain_penalty(domains, doc_text)
            final = max(final, 0.0)

            cand.score = final
            cand.score_parts.update(
                {
                    "bm25_norm": self._normalize_bm25(bm25_score),
                    "kg_norm": min(kg_score / 8.0, 1.0),
                    "entity": entity_score,
                    "phrase": phrase_score,
                    "token_overlap": overlap,
                    "intent": intent_score,
                    "article": article_score,
                    "reference": reference_score,
                    "ref_article": reference_parts.get("article", 0.0),
                    "ref_clause": reference_parts.get("clause", 0.0),
                    "ref_point": reference_parts.get("point", 0.0),
                    "domain": domain_score,
                    "similarity": min(similarity, 1.0),
                    "final": final,
                }
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def _first_entity_number(self, entities: object, key: str, fallback: Optional[str] = None) -> Optional[str]:
        if isinstance(entities, dict):
            for value in entities.get(key, []) or []:
                if key == "ARTICLE":
                    num = article_number(str(value))
                else:
                    num = clause_number(str(value))
                if num:
                    return num
        return fallback

    def _first_entity_point(self, entities: object, fallback: Optional[str] = None) -> Optional[str]:
        if isinstance(entities, dict):
            for value in entities.get("POINT", []) or []:
                pt = point_letter(str(value))
                if pt:
                    return pt
        return fallback

    def _normalize_bm25(self, score: float) -> float:
        if score <= 0:
            return 0.0
        return min(math.log1p(score) / math.log1p(25.0), 1.0)

    def _entity_score(self, entity_values: List[str], doc_text: str, doc_no_acc: str) -> float:
        if not entity_values:
            return 0.0
        score = 0.0
        for value in entity_values:
            v = normalize_text(value)
            if not v:
                continue
            if v in doc_text:
                score += 1.0
            elif strip_accents(v) in doc_no_acc:
                score += 0.8
        return min(score / max(len(entity_values), 1), 1.0)

    def _article_score(self, explicit_article: str | None, hinted_articles: set[str], cand: Candidate, doc_text: str) -> float:
        ref_article = None
        if self.graph and cand.document.node_id:
            ref_article = self.graph.reference_of_node(cand.document.node_id).get("article")
        doc_art = ref_article or article_number(f"{doc_text} {cand.document.node_id or ''}")
        if explicit_article and doc_art == explicit_article:
            return 1.0
        if explicit_article and f"điều {explicit_article}" in doc_text:
            return 1.0
        if doc_art and doc_art in hinted_articles:
            return 0.85
        for art in hinted_articles:
            if f"điều {art}" in doc_text:
                return 0.75
        return 0.0

    def _reference_score(
        self,
        explicit_article: Optional[str],
        explicit_clause: Optional[str],
        explicit_point: Optional[str],
        cand: Candidate,
        doc_text: str,
    ) -> tuple[float, Dict[str, float]]:
        """Score tới đúng Điểm/Khoản/Điều.

        Nếu câu hỏi có "điểm a khoản 1 điều 123", candidate chỉ đạt tối đa khi khớp cả 3 cấp.
        Nếu không có điểm/khoản rõ ràng, điểm này vẫn ưu tiên candidate cụ thể hơn trong cùng điều.
        """
        if self.graph and cand.document.node_id:
            ref = self.graph.reference_of_node(cand.document.node_id)
            doc_article = str(ref.get("article") or "") or article_number(doc_text)
            doc_clause = str(ref.get("clause") or "") or clause_number(doc_text)
            doc_point = str(ref.get("point") or "") or point_letter(doc_text)
            specificity = int(ref.get("specificity") or 0)
        else:
            doc_article = article_number(doc_text)
            doc_clause = clause_number(doc_text)
            doc_point = point_letter(doc_text)
            specificity = sum(1 for x in [doc_article, doc_clause, doc_point] if x)

        parts = {"article": 0.0, "clause": 0.0, "point": 0.0}
        required = 0
        score = 0.0

        if explicit_article:
            required += 1
            if doc_article == explicit_article or f"điều {explicit_article}" in doc_text:
                parts["article"] = 1.0
                score += 1.0
        if explicit_clause:
            required += 1
            if doc_clause == explicit_clause or f"khoản {explicit_clause}" in doc_text:
                parts["clause"] = 1.0
                score += 1.0
        if explicit_point:
            required += 1
            if doc_point == explicit_point or f"điểm {explicit_point}" in doc_text:
                parts["point"] = 1.0
                score += 1.0

        if required:
            # Phạt nếu hỏi điểm nhưng candidate chỉ khớp khoản/điều.
            return score / required, parts

        # Không hỏi rõ điểm/khoản: vẫn cộng nhẹ cho node cụ thể hơn để hệ thống có thể trả về Điểm khi evidence match.
        return min(0.15 * specificity, 0.45), parts

    def _domain_score(self, domains: List[str], doc_text: str) -> float:
        if not domains:
            return 0.0
        score = 0.0
        doc_no_acc = strip_accents(doc_text)
        for domain in domains:
            positives = DOMAIN_POSITIVE.get(domain, [])
            hits = sum(1 for kw in positives if normalize_text(kw) in doc_text or strip_accents(normalize_text(kw)) in doc_no_acc)
            if hits:
                score += min(hits / 3.0, 1.0)
        return min(score / max(len(domains), 1), 1.0)

    def _negative_domain_penalty(self, domains: List[str], doc_text: str) -> float:
        penalty = 0.0
        for domain in domains:
            for kw in DOMAIN_NEGATIVE.get(domain, []):
                if normalize_text(kw) in doc_text:
                    penalty += 0.25
        return min(penalty, 0.6)

    def _intent_score(self, intent: str, doc_text: str) -> float:
        if intent == "LOOKUP_PENALTY":
            return 1.0 if any(k in doc_text for k in ["hình phạt", "phạt tù", "phạt tiền", "tử hình", "chung thân", "khung hình phạt"]) else 0.0
        if intent == "LOOKUP_ARTICLE":
            return 1.0 if "điều" in doc_text else 0.0
        if intent == "LOOKUP_CONDITIONS":
            return 1.0 if any(k in doc_text for k in ["hành vi", "dấu hiệu", "điều kiện", "cần điều kiện", "cấu thành", "tình tiết", "điểm"] ) else 0.0
        if intent == "CLASSIFY_CASE":
            return 1.0 if any(k in doc_text for k in ["tội", "hành vi", "điều", "khoản", "điểm", "tính mạng", "tài sản"]) else 0.0
        if intent == "COMPARE_CRIMES":
            return 0.8 if any(k in doc_text for k in ["tội", "hành vi", "điều", "khoản", "điểm"]) else 0.0
        return 0.2

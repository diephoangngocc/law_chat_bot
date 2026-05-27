from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph import ArticleEvidence, LegalKnowledgeGraph
from .llm import ChatLLM
from .prompts import (
    FACT_EXTRACTION_SYSTEM,
    FINAL_REASONING_SYSTEM,
    fact_extraction_user,
    final_reasoning_user,
)
from .retrieval import LegalRetriever, RetrievalResult
from .text import compact
from .text import normalize_text
from .text import strip_accents


@dataclass(slots=True)
class PipelineOutput:
    facts: dict[str, Any]
    candidates: list[RetrievalResult]
    result: dict[str, Any]

    def to_json_dict(self, include_candidates: bool = True) -> dict[str, Any]:
        data = {
            "facts": self.facts,
            "result": self.result,
        }
        if include_candidates:
            data["candidates"] = [candidate_to_dict(item) for item in self.candidates]
        return data


class LegalReasoningPipeline:
    def __init__(self, data_dir: str | Path, llm: ChatLLM | None = None) -> None:
        self.graph = LegalKnowledgeGraph.from_csv_dir(data_dir)
        self.articles = self.graph.build_article_evidence()
        self.retriever = LegalRetriever(self.articles)
        self.llm = llm

    def run(
        self,
        case_summary: str,
        top_k: int = 8,
        point_k: int = 48,
        point_article_weight: float = 0.15,
        point_clause_weight: float = 0.5,
    ) -> PipelineOutput:
        facts = self.extract_facts(case_summary)
        query = self._build_query(case_summary, facts)
        candidates = self.retriever.retrieve(
            query,
            top_k=top_k,
            point_k=point_k,
            point_article_weight=point_article_weight,
            point_clause_weight=point_clause_weight,
        )
        evidence = self._format_evidence(candidates)
        result = self.reason(case_summary, facts, evidence, candidates)
        return PipelineOutput(facts=facts, candidates=candidates, result=result)

    def extract_facts(self, case_summary: str) -> dict[str, Any]:
        if not self.llm:
            return heuristic_extract_facts(case_summary)

        return self.llm.chat_json(
            [
                {"role": "system", "content": FACT_EXTRACTION_SYSTEM},
                {"role": "user", "content": fact_extraction_user(case_summary)},
            ]
        )

    def reason(
        self,
        case_summary: str,
        facts: dict[str, Any],
        evidence: str,
        candidates: list[RetrievalResult],
    ) -> dict[str, Any]:
        if not self.llm:
            return heuristic_result(candidates)

        facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
        return self.llm.chat_json(
            [
                {"role": "system", "content": FINAL_REASONING_SYSTEM},
                {"role": "user", "content": final_reasoning_user(case_summary, facts_json, evidence)},
            ]
        )

    def _build_query(self, case_summary: str, facts: dict[str, Any]) -> str:
        pieces = [case_summary]
        for key in ("hanh_vi", "doi_tuong_bi_xam_hai", "hau_qua", "loi", "chu_the", "tinh_tiet_dinh_khung", "tu_khoa_truy_van"):
            value = facts.get(key)
            if isinstance(value, list):
                pieces.extend(str(item) for item in value)
            elif value:
                pieces.append(str(value))
        return "\n".join(pieces)

    def _format_evidence(self, candidates: list[RetrievalResult]) -> str:
        blocks = []
        for idx, item in enumerate(candidates, start=1):
            article = item.article
            lines = [
                f"[Ứng viên {idx}] score={item.score:.3f}",
                f"article_id: {article.article_id}",
                f"title: {article.title}",
            ]
            if article.chapter:
                lines.append(f"chapter: {article.chapter}")
            for scored_clause in item.matched_clauses:
                clause = scored_clause.clause
                clause_score = scored_clause.score
                lines.append(f"- clause_id: {clause.clause_id}; score={clause_score:.3f}")
                lines.append(compact(clause.as_text(), max_chars=1600))
                for scored_point in scored_clause.matched_points:
                    point = scored_point.point
                    lines.append(f"  - point_id: {point.point_id}; score={scored_point.score:.3f}")
                    lines.append("    " + compact(point.as_text(), max_chars=600))
                if clause.graph_paths:
                    lines.append("  graph_paths: " + " | ".join(clause.graph_paths[:4]))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


def heuristic_result(candidates: list[RetrievalResult]) -> dict[str, Any]:
    if not candidates:
        return {
            "toi_danh_de_xuat": None,
            "dieu_luat": [],
            "khung_hinh_phat_du_kien": None,
            "phan_tich_vu_an": "Không tìm thấy ứng viên điều luật trong KG.",
            "doi_chieu_dieu_kien": [],
            "ung_vien_khac": [],
            "thieu_thong_tin": ["Cần bật LLM để phân tích pháp lý chi tiết."],
            "do_tin_cay": 0.0,
        }

    best = candidates[0]
    best_clause = best.matched_clauses[0].clause if best.matched_clauses else None
    alternatives = [
        {"article_id": item.article.article_id, "title": item.article.title, "score": round(item.score, 3)}
        for item in candidates[1:4]
    ]
    return {
        "toi_danh_de_xuat": best.article.title,
        "dieu_luat": [
            {
                "article_id": best.article.article_id,
                "title": best.article.title,
                "clause_id": best_clause.clause_id if best_clause else None,
                "clause": best_clause.clause_name if best_clause else None,
            }
        ],
        "khung_hinh_phat_du_kien": best_clause.penalties if best_clause else [],
        "phan_tich_vu_an": (
            "Kết quả này mới là truy xuất KG bằng BM25. Hãy bật LLM để đối chiếu đầy đủ "
            "các yếu tố cấu thành tội phạm và tình tiết định khung."
        ),
        "doi_chieu_dieu_kien": best_clause.conditions if best_clause else [],
        "ung_vien_khac": alternatives,
        "thieu_thong_tin": ["Chưa chạy LLM suy luận cuối."],
        "do_tin_cay": min(0.75, max(0.2, best.score / 25)),
    }


def heuristic_extract_facts(case_summary: str) -> dict[str, Any]:
    text = normalize_text(case_summary)
    folded_text = strip_accents(text).replace("đ", "d")
    query_terms: list[str] = []
    hau_qua: list[str] = []
    tinh_tiet: list[str] = []
    doi_tuong: list[str] = []

    death_terms = ("tử vong", "chết", "thiệt mạng")
    violence_terms = ("đâm", "chém", "bắn", "đánh", "dao", "súng", "hung khí")
    vehicle_terms = ("xe ", "ô tô", "xe máy")
    traffic_context_terms = ("giao thông", "điều khiển", "lái", "vượt đèn đỏ", "tông", "va chạm", "tai nạn")

    def contains_term(haystack: str, needle: str) -> bool:
        needle = needle.strip()
        if not needle:
            return False
        if " " in needle:
            return needle in haystack
        return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None

    def has(term: str) -> bool:
        folded_term = strip_accents(term.lower()).replace("đ", "d")
        return contains_term(text, term.lower()) or contains_term(folded_text, folded_term)

    def has_any(terms: tuple[str, ...]) -> bool:
        return any(has(term) for term in terms)

    if has_any(death_terms):
        hau_qua.append("làm chết người")
        query_terms.extend(["chết người", "làm chết người"])
    if has_any(death_terms) and has_any(violence_terms):
        query_terms.extend(["giết người", "tước đoạt tính mạng", "xâm phạm tính mạng"])
        doi_tuong.append("tính mạng con người")
    traffic_hit = has_any(vehicle_terms) and has_any(traffic_context_terms)
    if has("dao") or has("súng") or has("hung khí"):
        tinh_tiet.append("dùng hung khí nguy hiểm")
        query_terms.append("hung khí nguy hiểm")
    if has("gậy"):
        tinh_tiet.append("dùng công cụ nguy hiểm")
    if has("thương tích") or has("tổn thương cơ thể") or has("tổn hại sức khỏe"):
        doi_tuong.append("sức khỏe con người")
        if not traffic_hit and (has_any(violence_terms) or has("cố ý")):
            query_terms.extend(["cố ý gây thương tích", "gây thương tích", "tổn hại sức khỏe", "tỷ lệ tổn thương cơ thể"])
    if traffic_hit:
        query_terms.extend(["vi phạm quy định về tham gia giao thông đường bộ", "an toàn giao thông đường bộ"])
    if (has("ma túy") or has("ma tuý")) and not traffic_hit:
        query_terms.extend(["ma túy", "chất ma túy"])
    if has("đánh bạc") or has("cá độ"):
        query_terms.extend(["đánh bạc", "tổ chức đánh bạc"])
    if has("chiếm đoạt") or has("lừa đảo"):
        query_terms.extend(["chiếm đoạt tài sản", "lừa đảo chiếm đoạt tài sản"])
    if has("trộm") or has("lén lút"):
        query_terms.extend(["trộm cắp tài sản", "lén lút chiếm đoạt tài sản"])
        doi_tuong.append("tài sản")

    return {
        "hanh_vi": [case_summary],
        "doi_tuong_bi_xam_hai": doi_tuong,
        "hau_qua": hau_qua,
        "loi": "chưa rõ",
        "chu_the": [],
        "tinh_tiet_dinh_khung": tinh_tiet,
        "tu_khoa_truy_van": query_terms,
        "thieu_thong_tin": ["Chưa chạy LLM trích xuất tình tiết; đây là trích xuất heuristic để truy xuất KG."],
    }


def candidate_to_dict(item: RetrievalResult) -> dict[str, Any]:
    return {
        "article_id": item.article.article_id,
        "article_no": item.article.article_no,
        "title": item.article.title,
        "chapter": item.article.chapter,
        "score": round(item.score, 4),
        "matched_clauses": [
            {
                "clause_id": scored_clause.clause.clause_id,
                "clause": scored_clause.clause.clause_name,
                "score": round(scored_clause.score, 4),
                "actions": scored_clause.clause.actions,
                "conditions": scored_clause.clause.conditions,
                "points": [
                    {
                        "point_id": point.point_id,
                        "point": point.point_name,
                        "parent_logic_id": point.parent_logic_id,
                        "parent_logic": point.parent_logic_name,
                        "graph_paths": point.graph_paths,
                    }
                    for point in scored_clause.clause.points
                ],
                "matched_points": [
                    {
                        "point_id": scored_point.point.point_id,
                        "point": scored_point.point.point_name,
                        "score": round(scored_point.score, 4),
                        "parent_logic_id": scored_point.point.parent_logic_id,
                        "parent_logic": scored_point.point.parent_logic_name,
                        "graph_paths": scored_point.point.graph_paths,
                    }
                    for scored_point in scored_clause.matched_points
                ],
                "penalties": scored_clause.clause.penalties,
            }
            for scored_clause in item.matched_clauses
        ],
    }

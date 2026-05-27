from __future__ import annotations

import math
from dataclasses import dataclass

from .graph import ArticleEvidence, ClauseEvidence, PointEvidence
from .text import normalize_text, term_counts


@dataclass(slots=True)
class ScoredPoint:
    point: PointEvidence
    score: float


@dataclass(slots=True)
class ScoredClause:
    clause: ClauseEvidence
    score: float
    matched_points: list[ScoredPoint]

    def __iter__(self):
        yield self.clause
        yield self.score

    def __getitem__(self, index: int):
        if index == 0:
            return self.clause
        if index == 1:
            return self.score
        raise IndexError(index)


@dataclass(slots=True)
class RetrievalResult:
    article: ArticleEvidence
    score: float
    matched_clauses: list[ScoredClause]


class BM25Index:
    def __init__(self, docs: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_terms = [term_counts(text) for _, text in docs]
        self.doc_lens = [sum(counter.values()) for counter in self.doc_terms]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.df: dict[str, int] = {}
        for counter in self.doc_terms:
            for term in counter:
                self.df[term] = self.df.get(term, 0) + 1

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        query_terms = list(term_counts(query))
        scores: list[tuple[str, float]] = []
        total_docs = len(self.docs)
        for idx, (doc_id, _text) in enumerate(self.docs):
            score = 0.0
            doc_counter = self.doc_terms[idx]
            doc_len = self.doc_lens[idx] or 1
            for term in query_terms:
                freq = doc_counter.get(term, 0)
                if not freq:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * (freq * (self.k1 + 1) / denom)
            if score:
                scores.append((doc_id, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]


class LegalRetriever:
    def __init__(self, articles: list[ArticleEvidence]) -> None:
        self.articles = articles
        self.article_by_id = {article.article_id: article for article in articles}
        self.clause_to_article: dict[str, ArticleEvidence] = {}
        self.clause_by_id: dict[str, ClauseEvidence] = {}
        self.point_to_clause: dict[str, tuple[ArticleEvidence, ClauseEvidence, PointEvidence]] = {}
        clause_docs: list[tuple[str, str]] = []
        point_docs: list[tuple[str, str]] = []
        article_docs: list[tuple[str, str]] = []

        for article in articles:
            article_docs.append((article.article_id, article.as_text()))
            for clause in article.clauses:
                self.clause_to_article[clause.clause_id] = article
                self.clause_by_id[clause.clause_id] = clause
                clause_docs.append((clause.clause_id, article.title + "\n" + clause.as_text()))
                for point in clause.points:
                    self.point_to_clause[point.point_id] = (article, clause, point)
                    point_docs.append(
                        (
                            point.point_id,
                            "\n".join([article.title, clause.clause_name, point.as_text()]),
                        )
                    )

        self.article_index = BM25Index(article_docs)
        self.clause_index = BM25Index(clause_docs)
        self.point_index = BM25Index(point_docs)

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        clause_k: int = 24,
        point_k: int = 48,
        point_article_weight: float = 0.15,
        point_clause_weight: float = 0.5,
    ) -> list[RetrievalResult]:
        article_scores: dict[str, float] = {}
        clause_score_by_id: dict[str, float] = {}
        clause_scores_by_article: dict[str, list[ScoredClause]] = {}
        point_scores_by_clause: dict[str, list[ScoredPoint]] = {}
        normalized_query = normalize_text(query)

        for article_id, score in self.article_index.search(query, top_k=top_k * 3):
            article_scores[article_id] = max(article_scores.get(article_id, 0.0), score)

        for clause_id, score in self.clause_index.search(query, top_k=clause_k):
            article = self.clause_to_article.get(clause_id)
            if not article:
                continue
            clause_score = score + clause_phrase_boost(self.clause_by_id[clause_id], normalized_query)
            clause_score_by_id[clause_id] = max(clause_score_by_id.get(clause_id, 0.0), clause_score)
            article_scores[article.article_id] = article_scores.get(article.article_id, 0.0) + clause_score * 1.25

        if point_k > 0:
            for point_id, score in self.point_index.search(query, top_k=point_k):
                resolved = self.point_to_clause.get(point_id)
                if not resolved:
                    continue
                article, clause, point = resolved
                point_score = score + point_phrase_boost(point, normalized_query)
                weighted_point_score = point_score * point_clause_weight
                point_scores_by_clause.setdefault(clause.clause_id, []).append(ScoredPoint(point=point, score=point_score))
                clause_score_by_id[clause.clause_id] = max(clause_score_by_id.get(clause.clause_id, 0.0), weighted_point_score)
                article_scores[article.article_id] = article_scores.get(article.article_id, 0.0) + point_score * point_article_weight

        for clause_id, clause_score in clause_score_by_id.items():
            article = self.clause_to_article.get(clause_id)
            clause = self.clause_by_id.get(clause_id)
            if clause:
                matched_points = sorted(
                    point_scores_by_clause.get(clause.clause_id, []),
                    key=lambda item: item.score,
                    reverse=True,
                )
                clause_scores_by_article.setdefault(article.article_id, []).append(
                    ScoredClause(clause=clause, score=clause_score, matched_points=matched_points[:5])
                )

        for article in self.articles:
            boost = article_phrase_boost(article, normalized_query)
            if boost:
                article_scores[article.article_id] = article_scores.get(article.article_id, 0.0) + boost

        results: list[RetrievalResult] = []
        for article_id, score in article_scores.items():
            article = self.article_by_id[article_id]
            clauses = sorted(
                clause_scores_by_article.get(article_id, []),
                key=lambda item: item.score,
                reverse=True,
            )
            seen_clause_ids = {item.clause.clause_id for item in clauses}
            for clause in article.clauses:
                if len(clauses) >= 6:
                    break
                if clause.clause_id not in seen_clause_ids:
                    fallback_points = sorted(
                        point_scores_by_clause.get(clause.clause_id, []),
                        key=lambda item: item.score,
                        reverse=True,
                    )
                    fallback_score = clause_phrase_boost(clause, normalized_query)
                    if fallback_points:
                        fallback_score = max(fallback_score, fallback_points[0].score * point_clause_weight)
                    clauses.append(
                        ScoredClause(
                            clause=clause,
                            score=fallback_score,
                            matched_points=fallback_points[:5],
                        )
                    )
                    seen_clause_ids.add(clause.clause_id)
            if not clauses:
                clauses = [
                    ScoredClause(clause=clause, score=0.0, matched_points=[])
                    for clause in article.clauses[:6]
                ]
            results.append(RetrievalResult(article=article, score=score, matched_clauses=clauses[:5]))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]


def article_phrase_boost(article: ArticleEvidence, normalized_query: str) -> float:
    title = normalize_text(article.title)
    crime_name = title.split("tội ", 1)[1] if "tội " in title else title
    crime_name = crime_name.strip(" .:")
    boost = 0.0
    if crime_name and crime_name in normalized_query:
        boost += 250.0
    if "giết người" in normalized_query and "tội giết người" in title:
        boost += 350.0
    if "tham gia giao thông đường bộ" in normalized_query and "tham gia giao thông đường bộ" in title:
        boost += 250.0
    if "đánh bạc" in normalized_query and "tội đánh bạc" in title:
        boost += 250.0
    if "lừa đảo chiếm đoạt tài sản" in normalized_query and "lừa đảo chiếm đoạt tài sản" in title:
        boost += 250.0
    if "cố ý gây thương tích" in normalized_query and "tội cố ý gây thương tích" in title:
        boost += 250.0
    if "trộm cắp tài sản" in normalized_query and "tội trộm cắp tài sản" in title:
        boost += 250.0
    return boost


def clause_phrase_boost(clause: ClauseEvidence, normalized_query: str) -> float:
    text = normalize_text(clause.as_text())
    boost = 0.0
    if "một người chết" in normalized_query or "1 người chết" in normalized_query or "làm chết người" in normalized_query:
        if "làm chết người" in text and "02 người" not in text and "03 người" not in text:
            boost += 40.0
        if "làm chết 02 người" in text or "làm chết 03 người" in text:
            boost -= 25.0
    if "dùng hung khí nguy hiểm" in normalized_query and "hung khí nguy hiểm" in text:
        boost += 30.0
    return boost


def point_phrase_boost(point: PointEvidence, normalized_query: str) -> float:
    text = normalize_text(point.as_text())
    boost = 0.0
    if "làm chết người" in normalized_query and "làm chết người" in text:
        boost += 25.0
    if "hung khí nguy hiểm" in normalized_query and "hung khí nguy hiểm" in text:
        boost += 25.0
    if "ma túy" in normalized_query and "ma túy" in text:
        boost += 20.0
    if "chiếm đoạt tài sản" in normalized_query and "chiếm đoạt tài sản" in text:
        boost += 20.0
    return boost

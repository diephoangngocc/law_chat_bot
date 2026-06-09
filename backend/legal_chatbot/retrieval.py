from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from .graph import LegalKnowledgeGraph
from .models import Candidate, Document
from .text import article_number, normalize_text, phrase_count, short_text, strip_accents, tokenize, unique_keep_order


class BM25Index:
    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        # Index cả bản có dấu và không dấu để câu hỏi thiếu dấu vẫn match.
        self.doc_tokens: List[List[str]] = []
        for doc in documents:
            tokens = tokenize(doc.content)
            tokens += tokenize(strip_accents(doc.content))
            self.doc_tokens.append(tokens)
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freq[token] += 1

    def score(self, query: str, doc_index: int) -> float:
        q_tokens = tokenize(query) + tokenize(strip_accents(query))
        if not q_tokens or doc_index >= len(self.documents):
            return 0.0
        freqs = Counter(self.doc_tokens[doc_index])
        score = 0.0
        total_docs = max(len(self.documents), 1)
        dl = self.doc_len[doc_index] or 1
        for token in q_tokens:
            if freqs[token] == 0:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            tf = freqs[token]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, phrase_terms: List[str] | None = None, limit: int = 10) -> List[Candidate]:
        phrase_terms = phrase_terms or []
        scores = []
        for i, doc in enumerate(self.documents):
            s = self.score(query, i)
            # Phrase boost giúp cụm "tội giết người", "điều 123", "cướp tài sản" thắng token chung.
            pc = phrase_count(f"{doc.title} {doc.content}", phrase_terms)
            if pc:
                s += min(pc * 2.0, 10.0)
            # Exact article trong doc.
            q_art = article_number(query)
            if q_art and q_art == article_number(f"{doc.title} {doc.content} {doc.node_id or ''}"):
                s += 25.0
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [Candidate(document=self.documents[i], score=s, score_parts={"bm25": s}) for i, s in scores[:limit]]


def _load_text_documents(doc_dir: Path) -> List[Document]:
    docs: List[Document] = []
    if not doc_dir.exists():
        return docs

    for path in sorted(doc_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="utf-8-sig", errors="ignore")
            docs.append(Document(id=f"file:{path.relative_to(doc_dir)}", title=path.name, content=content, source="document"))
        elif suffix == ".csv":
            try:
                rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
            except Exception:
                continue
            for idx, row in enumerate(rows):
                content = " | ".join(str(v).strip() for v in row.values() if str(v).strip())
                if content:
                    docs.append(Document(id=f"file:{path.name}:{idx}", title=f"{path.name} #{idx+1}", content=content, source="document_csv"))
    return docs


class HybridRetriever:
    def __init__(self, graph: LegalKnowledgeGraph, data_dir: Path):
        self.graph = graph
        self.data_dir = Path(data_dir)
        self.documents: List[Document] = graph.to_documents() + _load_text_documents(self.data_dir / "documents")
        self.index = BM25Index(self.documents)

    def rebuild(self) -> None:
        self.documents = self.graph.to_documents() + _load_text_documents(self.data_dir / "documents")
        self.index = BM25Index(self.documents)

    def retrieve(self, question: str, semantic: Dict[str, object], top_k: int = 5, expand_depth: int = 1) -> List[Candidate]:
        entities = semantic.get("entities", {}) or {}
        query_terms = list(semantic.get("query_terms", []) or [])
        hinted_articles = list(semantic.get("hinted_articles", []) or [])

        # Query chính: câu hỏi + query_terms. Không quá dài để BM25 không bị nhiễu.
        expanded_query = " ".join(unique_keep_order([question, *query_terms[:20]]))
        phrase_terms = unique_keep_order(query_terms + [question])

        bm25_candidates = self.index.search(expanded_query, phrase_terms=phrase_terms, limit=max(top_k * 6, 30))

        # KG search: entity/query terms + hinted articles, rồi mở rộng neighbor.
        kg_seed_terms: List[str] = []
        if isinstance(entities, dict):
            for values in entities.values():
                kg_seed_terms.extend(values)
        kg_seed_terms.extend(query_terms)
        kg_hits = self.graph.search_nodes(
            question,
            extra_terms=kg_seed_terms,
            limit=40,
            preferred_articles=hinted_articles,
        )
        seed_ids = [node.id for node, _score in kg_hits]

        # Nếu có hinted article, luôn thêm cả article node vào seed.
        for art_num in hinted_articles:
            for node in self.graph.get_article_nodes(art_num):
                seed_ids.append(node.id)

        expanded_nodes = self.graph.expand(seed_ids, depth=max(expand_depth, 2), max_nodes=140)

        kg_candidates: List[Candidate] = []
        hit_score_map = {node.id: score for node, score in kg_hits}
        for node in expanded_nodes:
            doc = Document(
                id=f"node:{node.id}",
                title=node.name or node.id,
                content=node.text,
                source="kg_expanded",
                node_id=node.id,
                metadata={"label": node.label, **node.metadata},
            )
            kg_score = 1.0
            if node.id in hit_score_map:
                kg_score += min(hit_score_map[node.id] / 5.0, 8.0)
            if self.graph.is_article_node(node):
                kg_score += 2.5
            article = self.graph.nearest_article(node.id)
            if article:
                kg_score += 0.8
                art_num = self.graph.article_number_of_node(article)
                if art_num in hinted_articles:
                    # Tang tu 5.0 len 15.0: dam bao dieu luat duoc hint boi DOMAIN_HINTS
                    # luon thang dieu luat khac du BM25 co loi the token (vd "trong cay"
                    # trong D247 khi query co "trong cay" nhung y nghia khac ngu canh).
                    # Max KG score hinted: 1.0+8.0+2.5+0.8+15.0 = 27.3
                    # Max KG score unrelated: 1.0+8.0+2.5+0.8 = 12.3
                    # → margin 15pt du bu BM25 gap thong thuong (~8-10pt).
                    kg_score += 15.0
            kg_candidates.append(Candidate(document=doc, score=kg_score, score_parts={"kg": kg_score}))

        merged: Dict[str, Candidate] = {}
        for cand in bm25_candidates + kg_candidates:
            key = cand.document.id
            if key not in merged:
                merged[key] = cand
            else:
                merged[key].score += cand.score
                for k, v in cand.score_parts.items():
                    merged[key].score_parts[k] = merged[key].score_parts.get(k, 0.0) + v

        candidates = list(merged.values())
        for cand in candidates:
            cand.document.content = short_text(cand.document.content, 1400)
            if cand.document.node_id:
                cand.kg_paths = self.graph.paths_from_node(cand.document.node_id, depth=2, max_paths=5)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: max(top_k * 6, top_k)]

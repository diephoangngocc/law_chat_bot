from __future__ import annotations

from typing import Dict, Optional

from .answer_llm import LocalLLMAnswerer
from .answer_template import TemplateAnswerer
from .config import Settings
from .evidence import EvidenceBuilder
from .graph import LegalKnowledgeGraph
from .llm_client import OllamaClient
from .rerank import LegalReranker
from .retrieval import HybridRetriever
from .semantic import SemanticParser
from .text import normalize_text


class LegalChatbotPipeline:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.from_env()
        self.graph = LegalKnowledgeGraph(self.settings.data_dir)
        self.semantic_parser = SemanticParser()
        self.retriever = HybridRetriever(self.graph, self.settings.data_dir)
        self.reranker = LegalReranker(self.graph)
        self.evidence_builder = EvidenceBuilder(self.graph)
        self.template_answerer = TemplateAnswerer()
        self.local_llm_answerer = LocalLLMAnswerer(
            OllamaClient(
                base_url=self.settings.ollama_base_url,
                model=self.settings.local_llm_model,
                timeout=self.settings.local_llm_timeout,
            )
        )

    def run(self, question: str, top_k: Optional[int] = None, mode: Optional[str] = None, use_llm: Optional[bool] = None) -> Dict[str, object]:
        question = (question or "").strip()
        if not question:
            return {
                "reply": "Bạn hãy nhập câu hỏi pháp luật cần tra cứu.",
                "mode": "empty",
                "data": {},
            }

        top_k = int(top_k or self.settings.top_k)
        mode = self._resolve_mode(mode=mode, use_llm=use_llm)

        semantic = self.semantic_parser.parse(question)
        normalized_question = normalize_text(question)

        candidates = self.retriever.retrieve(
            question=normalized_question,
            semantic=semantic,
            top_k=max(top_k, 1),
            expand_depth=self.settings.kg_expand_depth,
        )
        ranked = self.reranker.rank(
            question=normalized_question,
            semantic=semantic,
            candidates=candidates,
            top_k=max(top_k, 1),
        )
        evidence = self.evidence_builder.build(question=question, semantic=semantic, candidates=ranked)

        if mode == "local_llm":
            reply = self.local_llm_answerer.generate(question=question, evidence=evidence)
        else:
            reply = self.template_answerer.generate(question=question, evidence=evidence)

        return {
            "reply": reply,
            "mode": mode,
            "data": {
                "semantic": semantic,
                "evidence": evidence,
                "candidates": [c.to_dict() for c in ranked],
                "graph_stats": self.graph.stats(),
            },
        }

    def _resolve_mode(self, mode: Optional[str], use_llm: Optional[bool]) -> str:
        if use_llm is True:
            return "local_llm"
        if use_llm is False and mode is None:
            return "no_llm"
        mode = (mode or self.settings.answer_mode or "no_llm").strip().lower()
        if mode not in {"no_llm", "local_llm"}:
            mode = "no_llm"
        return mode

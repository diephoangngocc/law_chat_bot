from __future__ import annotations

from typing import Dict, Optional

from .answer_llm import LocalLLMAnswerer
from .article_answer import ArticleAnswerer
from .answer_template import TemplateAnswerer
from .config import Settings
from .evidence import EvidenceBuilder
from .graph import LegalKnowledgeGraph
from .llm_client import build_llm_client
from .rerank import LegalReranker
from .retrieval import HybridRetriever
from .scope_guard import classify_basic_message, is_legal_question, out_of_scope_reply
from .semantic import SemanticParser
from .text import normalize_text


class LegalChatbotPipeline:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.from_env()
        self.graph = LegalKnowledgeGraph(self.settings.data_dir)
        self.semantic_parser = SemanticParser()
        self.retriever = HybridRetriever(self.graph, self.settings.data_dir)
        # Scope-scoring version needs graph in reranker.
        self.reranker = LegalReranker(self.graph)
        self.evidence_builder = EvidenceBuilder(self.graph)
        self.article_answerer = ArticleAnswerer(self.graph)
        self.template_answerer = TemplateAnswerer()
        self.local_llm_answerer = LocalLLMAnswerer(
            build_llm_client(
                provider=self.settings.llm_provider,
                model=self.settings.local_llm_model,
                timeout=self.settings.local_llm_timeout,
                ollama_base_url=self.settings.ollama_base_url,
                openai_compatible_base_url=self.settings.openai_compatible_base_url,
                api_key=self.settings.local_llm_api_key,
            )
        )

    def run(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        use_llm: Optional[bool] = None,
    ) -> Dict[str, object]:
        question = (question or "").strip()
        if not question:
            return {
                "reply": "Bạn hãy nhập câu hỏi pháp luật cần tra cứu.",
                "mode": "empty",
                "data": {},
            }

        resolved_mode = self._resolve_mode(mode=mode, use_llm=use_llm)

        # 0) Small-talk/help router. Không đưa những câu như "chào" vào KG/RAG,
        # nếu không evidence pháp lý sẽ làm prompt LLM dài hàng nghìn token.
        basic = classify_basic_message(question)
        if basic is not None:
            route, reply = basic
            return {
                "reply": reply,
                "mode": "basic",
                "data": {
                    "route": route,
                    "answer_mode_requested": resolved_mode,
                    "graph_stats": self.graph.stats(),
                },
            }

        semantic = self.semantic_parser.parse(question)

        # 1) Scope guard. Nếu không phải câu hỏi cơ bản và cũng không phải câu hỏi pháp lý,
        # trả lời ngoài phạm vi thay vì retrieval lạc sang một điều luật ngẫu nhiên.
        if not is_legal_question(question, semantic):
            return {
                "reply": out_of_scope_reply(),
                "mode": "out_of_scope",
                "data": {
                    "semantic": semantic,
                    "answer_mode_requested": resolved_mode,
                    "graph_stats": self.graph.stats(),
                },
            }

        # 2) Câu hỏi nêu rõ Điều luật: xử lý trực tiếp từ KG để trả lời ổn định và tránh
        # gửi cả một đống evidence sang LLM. Ví dụ: "Điều 123 quy định gì?" hoặc
        # "Điều 123 và 124 khác nhau như thế nào?".
        if self.article_answerer.can_handle(question, semantic):
            reply, article_evidence = self.article_answerer.answer(question, semantic)
            return {
                "reply": reply,
                "mode": "article_direct",
                "data": {
                    "semantic": semantic,
                    "evidence": article_evidence,
                    "answer_mode_requested": resolved_mode,
                    "graph_stats": self.graph.stats(),
                },
            }

        top_k = int(top_k or self.settings.top_k)
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
        evidence = self.evidence_builder.build(question=question, semantic=semantic, candidates=ranked, top_k=top_k)

        if resolved_mode == "local_llm":
            reply = self.local_llm_answerer.generate(question=question, evidence=evidence)
        else:
            reply = self.template_answerer.generate(question=question, evidence=evidence, top_k=top_k)

        return {
            "reply": reply,
            "mode": resolved_mode,
            "data": {
                "semantic": semantic,
                "evidence": evidence,
                "candidates": [c.to_dict() for c in ranked],
                "graph_stats": self.graph.stats(),
                "llm_provider": self.settings.llm_provider,
                "llm_model": self.settings.local_llm_model,
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

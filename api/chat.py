from __future__ import annotations

import json
import sys
import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.legal_chatbot import LegalChatbotPipeline  # noqa: E402

_PIPELINE = None


def get_pipeline() -> LegalChatbotPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        # Ensure data directory path works in Vercel
        os.environ.setdefault("DATA_DIR", str(Path(__file__).parents[1] / "data"))
        _PIPELINE = LegalChatbotPipeline()
    return _PIPELINE


def _json_response(handler: BaseHTTPRequestHandler, payload: Dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        _json_response(self, {"ok": True})

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        from backend.legal_chatbot import law_content as lc

        try:
            parsed = urlparse(self.path)
            if "law" in parsed.path:
                params = parse_qs(parsed.query)
                ref = (params.get("ref") or [""])[0].strip()
                hierarchy = lc.lookup_hierarchy(ref) if ref else []
                if hierarchy:
                    _json_response(self, {"ref": ref, "hierarchy": hierarchy})
                else:
                    _json_response(self, {"ref": ref, "hierarchy": []}, status=404)
            elif "chat" in parsed.path or parsed.path == "/api/":
                pipeline = get_pipeline()
                _json_response(self, {"ok": True, "graph": pipeline.graph.stats()})
            else:
                _json_response(self, {"error": "Not found"}, status=404)
        except Exception as exc:
            _json_response(self, {"ok": False, "error": str(exc)}, status=500)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw or "{}")
            message = payload.get("message") or payload.get("question") or ""
            top_k = payload.get("top_k")
            mode = payload.get("mode")
            use_llm = payload.get("use_llm")
            result = get_pipeline().run(message, top_k=top_k, mode=mode, use_llm=use_llm)
            _json_response(self, result)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _json_response(self, {"reply": "API gặp lỗi khi xử lý câu hỏi.", "error": str(exc)}, status=500)

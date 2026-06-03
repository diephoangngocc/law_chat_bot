from __future__ import annotations

import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.legal_chatbot import LegalChatbotPipeline  # noqa: E402

PIPELINE = LegalChatbotPipeline()
PUBLIC_DIR = ROOT / "frontend" / "public"


class LocalHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        from urllib.parse import parse_qs
        from backend.legal_chatbot import law_content as lc

        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            self._send_json({"ok": True, "graph": PIPELINE.graph.stats()})
            return
        if parsed.path == "/api/law":
            params = parse_qs(parsed.query)
            ref = (params.get("ref") or [""])[0].strip()
            hierarchy = lc.lookup_hierarchy(ref) if ref else []
            if hierarchy:
                self._send_json({"ref": ref, "hierarchy": hierarchy})
            else:
                self._send_json({"ref": ref, "hierarchy": []}, status=404)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self._send_json({"error": "Not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw or "{}")
            result = PIPELINE.run(
                question=payload.get("message") or payload.get("question") or "",
                top_k=payload.get("top_k"),
                mode=payload.get("mode"),
                use_llm=payload.get("use_llm"),
            )
            self._send_json(result)
        except Exception as exc:
            self._send_json({"reply": "Server gặp lỗi khi xử lý câu hỏi.", "error": str(exc)}, status=500)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), LocalHandler)
    print(f"Law Chatbot V2 đang chạy: http://localhost:{port}")
    print(f"Graph stats: {PIPELINE.graph.stats()}")
    print(f"Answer mode: {PIPELINE.settings.answer_mode}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")

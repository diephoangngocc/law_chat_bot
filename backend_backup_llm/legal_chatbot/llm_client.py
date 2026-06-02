from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List


class LocalLLMError(RuntimeError):
    """Raised when the local LLM backend cannot be called safely."""


def strip_thinking_blocks(text: str) -> str:
    """Remove Qwen/Ollama thinking traces if a thinking model emits them."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*(suy nghĩ|thinking)\s*:\s*.*?(?=\n\s*(trả lời|answer)\s*:|$)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:4b"
    timeout: int = 180
    temperature: float = 0.1
    top_p: float = 0.85
    num_ctx: int = 8192

    def _request_json(self, path: str, payload: Dict[str, Any] | None = None, method: str = "POST") -> Dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise LocalLLMError(
                f"Không gọi được Ollama tại {url}. Hãy kiểm tra Ollama đã chạy chưa và OLLAMA_BASE_URL có đúng không. Chi tiết: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise LocalLLMError("Ollama phản hồi quá lâu. Hãy thử model nhỏ hơn hoặc tăng LOCAL_LLM_TIMEOUT.") from exc

        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise LocalLLMError(f"Phản hồi Ollama không phải JSON hợp lệ: {raw[:300]}") from exc

    def list_models(self) -> List[str]:
        data = self._request_json("/api/tags", method="GET")
        models = data.get("models") or []
        names = []
        for item in models:
            name = item.get("name")
            if name:
                names.append(str(name))
        return names

    def ensure_model_available(self) -> None:
        names = self.list_models()
        if self.model not in names:
            hint = "ollama pull " + self.model
            available = ", ".join(names[:8]) if names else "chưa có model nào"
            raise LocalLLMError(
                f"Model '{self.model}' chưa có trong Ollama. Hãy chạy: {hint}. Model hiện có: {available}."
            )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.ensure_model_available()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_ctx": self.num_ctx,
            },
        }
        result = self._request_json("/api/chat", payload=payload)
        message = result.get("message") or {}
        content = message.get("content") or result.get("response")
        if not content:
            raise LocalLLMError(f"Phản hồi Ollama không có nội dung trả lời: {json.dumps(result, ensure_ascii=False)[:300]}")
        return strip_thinking_blocks(str(content))

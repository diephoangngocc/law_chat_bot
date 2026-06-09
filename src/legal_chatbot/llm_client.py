from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


class LocalLLMError(RuntimeError):
    pass


class ChatClient(Protocol):
    def chat(self, system_prompt: str, user_prompt: str) -> str: ...


def strip_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks returned by some reasoning models."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:4b"
    timeout: int = 120

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        url = self.base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.15, "top_p": 0.85, "num_ctx": 8192},
        }
        result = self._request_json(url, payload)
        message = result.get("message") or {}
        content = message.get("content") or result.get("response")
        if not content:
            raise LocalLLMError(f"Phản hồi Ollama không có nội dung: {json.dumps(result, ensure_ascii=False)[:300]}")
        return strip_thinking_blocks(str(content))

    def _request_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise LocalLLMError(f"Không gọi được Ollama tại {url}: {exc}") from exc
        except TimeoutError as exc:
            raise LocalLLMError("Ollama phản hồi quá lâu.") from exc
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise LocalLLMError(f"Phản hồi Ollama không phải JSON hợp lệ: {raw[:300]}") from exc


@dataclass
class OpenAICompatibleClient:
    """Client for LM Studio OpenAI-compatible local server."""

    base_url: str = "http://localhost:1234/v1"
    model: str = "auto"
    timeout: int = 180
    api_key: str = "lm-studio"
    temperature: float = 0.1
    top_p: float = 0.85
    max_tokens: int = 1400

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        model_name = self._resolve_model()
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        result = self._request_json("/chat/completions", payload=payload, method="POST")
        choices = result.get("choices") or []
        if not choices:
            raise LocalLLMError(f"LM Studio không trả về choices: {json.dumps(result, ensure_ascii=False)[:300]}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = (message or {}).get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "".join(str(part.get("text") or part.get("content") or part) for part in content)
        if not content:
            raise LocalLLMError(f"LM Studio không trả về nội dung: {json.dumps(result, ensure_ascii=False)[:300]}")
        return strip_thinking_blocks(str(content))

    def _resolve_model(self) -> str:
        requested = (self.model or "auto").strip()
        if requested.lower() not in {"auto", "local-model", "lmstudio", "lm_studio"}:
            return requested
        models = self.list_models()
        if not models:
            raise LocalLLMError("LM Studio chưa load model nào. Hãy bấm Load Model trong tab Developer/Local Server.")
        return models[0]

    def list_models(self) -> List[str]:
        result = self._request_json("/models", method="GET")
        data = result.get("data") or []
        names: List[str] = []
        for item in data:
            if isinstance(item, dict):
                name = item.get("id") or item.get("name")
                if name:
                    names.append(str(name))
        return names

    def _request_json(self, path: str, payload: Optional[Dict[str, Any]] = None, method: str = "POST") -> Dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise LocalLLMError(f"LM Studio trả lỗi HTTP {exc.code} tại {url}. Chi tiết: {body}") from exc
        except urllib.error.URLError as exc:
            raise LocalLLMError(
                f"Không gọi được LM Studio tại {url}. Hãy kiểm tra Local Server đang Running và base URL đúng chưa. Chi tiết: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise LocalLLMError("LM Studio phản hồi quá lâu. Hãy thử model nhỏ hơn hoặc tăng LOCAL_LLM_TIMEOUT.") from exc
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise LocalLLMError(f"Phản hồi LM Studio không phải JSON hợp lệ: {raw[:300]}") from exc


def build_llm_client(
    provider: str = "ollama",
    model: str = "qwen3:4b",
    timeout: int = 180,
    ollama_base_url: str = "http://localhost:11434",
    openai_compatible_base_url: str = "http://localhost:1234/v1",
    api_key: str = "lm-studio",
) -> ChatClient:
    provider_key = (provider or "ollama").strip().lower().replace("-", "_")
    if provider_key in {"lmstudio", "lm_studio", "openai_compatible", "openai", "compatible"}:
        return OpenAICompatibleClient(
            base_url=openai_compatible_base_url,
            model=model or "auto",
            timeout=timeout,
            api_key=api_key or "lm-studio",
        )
    return OllamaClient(
        base_url=ollama_base_url,
        model=model or "qwen3:4b",
        timeout=timeout,
    )

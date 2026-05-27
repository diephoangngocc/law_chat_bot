from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class ChatLLM(Protocol):
    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.0) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class OpenAICompatibleLLM:
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 90

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLM":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return cls(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        )

    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.0) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Only add response_format for OpenAI API (not compatible with LM Studio, Ollama, etc.)
        if "openai.com" in self.base_url:
            payload["response_format"] = {"type": "json_object"}
        
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {detail}") from exc

        content = body["choices"][0]["message"]["content"]
        return parse_json_object(content)


def parse_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(content[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM response does not contain a JSON object")

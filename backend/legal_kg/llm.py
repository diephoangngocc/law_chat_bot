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
class HuggingFaceLLM:
    model: str
    api_key: str
    base_url: str = "https://api-inference.huggingface.co/v1"
    timeout: int = 120

    @classmethod
    def from_env(cls) -> "HuggingFaceLLM":
        api_key = os.environ.get("HF_TOKEN", "")
        if not api_key:
            raise RuntimeError(
                "HF_TOKEN is not set. Please set your Hugging Face API token:\n"
                "   Windows: $env:HF_TOKEN='your-token-here'\n"
                "   Linux/Mac: export HF_TOKEN='your-token-here'"
            )
        return cls(
            model=os.environ.get("HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
            api_key=api_key,
            base_url="https://api-inference.huggingface.co/v1",
        )

    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.0) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        }
        
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


# Alias for backward compatibility
OpenAICompatibleLLM = HuggingFaceLLM

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import requests

logger = logging.getLogger(__name__)


class ChatLLM(Protocol):
    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.0) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class HuggingFaceLLM:
    model: str
    api_key: str
    base_url: str = "https://api-inference.huggingface.co/v1"
    timeout: int = 60

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
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        }

        try:
            with requests.post(url, headers=headers, json=payload, timeout=self.timeout) as response:
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                return parse_json_object(content)
                
        except requests.exceptions.Timeout:
            logger.error("Hugging Face API timeout after %d seconds", self.timeout)
            return {
                "error": "Hugging Face API timeout. Please try again.",
                "toi_danh_de_xuat": None,
                "dieu_luat": [],
                "khung_hinh_phat_du_kien": None,
                "phan_tich_vu_an": "API timeout - vui lòng thử lại.",
            }
        except requests.exceptions.ConnectionError as exc:
            logger.error("Connection error: %s", str(exc))
            return {
                "error": f"Connection error: {str(exc)}",
                "toi_danh_de_xuat": None,
                "dieu_luat": [],
                "khung_hinh_phat_du_kien": None,
                "phan_tich_vu_an": "Lỗi kết nối API - vui lòng thử lại sau.",
            }
        except requests.exceptions.RequestException as exc:
            logger.error("API request error: %s", str(exc))
            return {
                "error": f"API error: {str(exc)}",
                "toi_danh_de_xuat": None,
                "dieu_luat": [],
                "khung_hinh_phat_du_kien": None,
                "phan_tich_vu_an": f"Lỗi API - vui lòng thử lại sau.",
            }


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

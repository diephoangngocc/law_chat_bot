from __future__ import annotations

import json
import os
import logging
import re
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
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            
            # Check HTTP status and handle errors
            if response.status_code == 503:
                try:
                    error_detail = response.json().get("error", "")
                except:
                    error_detail = response.text[:200]
                if "loading" in str(error_detail).lower():
                    raise RuntimeError(
                        f"Model '{self.model}' đang được load trên Hugging Face. Vui lòng chờ vài phút rồi thử lại."
                    )
                raise RuntimeError(f"Model không khả dụng: {error_detail}")
            
            if response.status_code == 401:
                raise RuntimeError("API token không hợp lệ. Vui lòng kiểm tra HF_TOKEN.")
            
            if response.status_code >= 400:
                try:
                    error_detail = response.json().get("error", response.text[:200])
                except:
                    error_detail = response.text[:200]
                raise RuntimeError(f"Lỗi HTTP {response.status_code}: {error_detail}")
            
            # Try to parse JSON response
            try:
                body = response.json()
            except json.JSONDecodeError:
                logger.warning("Non-JSON response from API: %s", response.text[:200])
                raise RuntimeError(f"API trả về phản hồi không hợp lệ: {response.text[:100]}")
            
            # Validate response structure
            if not isinstance(body, dict):
                raise RuntimeError(f"API trả về định dạng không đúng: {type(body)}")
            
            if "choices" not in body or not body["choices"]:
                logger.warning("Invalid API response: no choices in body")
                return self._fallback_response("Response không hợp lệ từ API")
            
            first_choice = body["choices"][0]
            if not isinstance(first_choice, dict):
                return self._fallback_response("Response structure invalid")
            
            message = first_choice.get("message", {})
            if not isinstance(message, dict):
                return self._fallback_response("Message structure invalid")
            
            content = message.get("content")
            if not content:
                logger.warning("Empty content in response")
                return self._fallback_response("API trả về nội dung trống")
            
            # Parse the content - this is where JSON parsing can fail
            return parse_json_object(content)
                
        except requests.exceptions.Timeout:
            logger.error("Hugging Face API timeout after %d seconds", self.timeout)
            return self._fallback_response("API timeout - vui lòng thử lại sau")
        except requests.exceptions.ConnectionError as exc:
            logger.error("Connection error: %s", str(exc))
            return self._fallback_response("Lỗi kết nối - vui lòng kiểm tra mạng")
        except RuntimeError:
            # Re-raise RuntimeError for model loading issues
            raise
        except Exception as exc:
            logger.error("Unexpected error: %s", str(exc))
            return self._fallback_response(f"Lỗi không xác định: {str(exc)}")

    def _fallback_response(self, error_msg: str) -> dict[str, Any]:
        """Return a fallback response when API fails."""
        return {
            "error": error_msg,
            "toi_danh_de_xuat": None,
            "dieu_luat": [],
            "khung_hinh_phat_du_kien": [],
            "phan_tich_vu_an": error_msg,
            "doi_chieu_dieu_kien": [],
            "thieu_thong_tin": ["LLM không khả dụng - sử dụng kết quả truy xuất KG"],
        }


def clean_markdown_json(text: str) -> str:
    """Remove markdown code blocks from JSON text."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def parse_json_object(content: str) -> dict[str, Any]:
    """
    Parse LLM response to JSON. 
    If LLM returns plain text instead of JSON, wrap it into the expected format.
    """
    if not content or not isinstance(content, str):
        return _wrap_raw_text(str(content) if content else "")
    
    content = content.strip()
    if not content:
        return _wrap_raw_text("")
    
    # Step 1: Try direct JSON parse
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return _ensure_valid_schema(parsed)
    except json.JSONDecodeError:
        pass
    
    # Step 2: Clean markdown and try again
    cleaned = clean_markdown_json(content)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return _ensure_valid_schema(parsed)
    except json.JSONDecodeError:
        pass
    
    # Step 3: Try to find JSON object in content
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        json_part = content[start:end + 1]
        try:
            parsed = json.loads(json_part)
            if isinstance(parsed, dict):
                return _ensure_valid_schema(parsed)
        except json.JSONDecodeError:
            pass
    
    # Step 4: LLM returned plain text - wrap it into expected format
    logger.warning("LLM returned plain text instead of JSON. Wrapping into schema.")
    return _wrap_raw_text(content)


def _wrap_raw_text(raw_text: str) -> dict[str, Any]:
    """Wrap raw text into the expected schema format."""
    return {
        "phan_tich_vu_an": raw_text,
        "toi_danh_de_xuat": None,
        "dieu_luat": [],
        "khung_hinh_phat_du_kien": [],
        "doi_chieu_dieu_kien": [],
        "ung_vien_khac": [],
        "thieu_thong_tin": ["LLM trả về phản hồi không đúng định dạng JSON"],
        "do_tin_cay": 0.0,
        "_raw_text": True,
    }


def _ensure_valid_schema(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure the parsed data has all required keys with proper types."""
    required_keys = {
        "phan_tich_vu_an": "",
        "toi_danh_de_xuat": None,
        "dieu_luat": [],
        "khung_hinh_phat_du_kien": [],
        "doi_chieu_dieu_kien": [],
        "ung_vien_khac": [],
        "thieu_thong_tin": [],
        "do_tin_cay": 0.0,
    }
    
    result = {}
    for key, default in required_keys.items():
        if key in data:
            result[key] = data[key]
        else:
            result[key] = default
    
    # Preserve any extra keys (like error, facts from extraction)
    for key, value in data.items():
        if key not in required_keys:
            result[key] = value
    
    return result


# Alias for backward compatibility
OpenAICompatibleLLM = HuggingFaceLLM

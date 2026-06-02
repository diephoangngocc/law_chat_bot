from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    answer_mode: str = "no_llm"
    top_k: int = 5
    kg_expand_depth: int = 1
    ollama_base_url: str = "http://localhost:11434"
    local_llm_model: str = "qwen3:4b"
    local_llm_timeout: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        root = project_root()
        raw_data_dir = os.getenv("DATA_DIR", str(root / "data"))
        data_dir = Path(raw_data_dir)
        if not data_dir.is_absolute():
            data_dir = root / data_dir

        def env_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        return cls(
            data_dir=data_dir,
            answer_mode=os.getenv("ANSWER_MODE", "no_llm").strip() or "no_llm",
            top_k=env_int("TOP_K", 5),
            kg_expand_depth=env_int("KG_EXPAND_DEPTH", 1),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            local_llm_model=os.getenv("LOCAL_LLM_MODEL", "qwen3:4b"),
            local_llm_timeout=env_int("LOCAL_LLM_TIMEOUT", 120),
        )

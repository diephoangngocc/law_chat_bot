from backend.legal_chatbot.config import Settings
from backend.legal_chatbot.llm_client import build_llm_client, LocalLLMError


if __name__ == "__main__":
    settings = Settings.from_env()
    client = build_llm_client(
        provider=settings.llm_provider,
        model=settings.local_llm_model,
        timeout=settings.local_llm_timeout,
        ollama_base_url=settings.ollama_base_url,
        openai_compatible_base_url=settings.openai_compatible_base_url,
        api_key=settings.local_llm_api_key,
    )
    try:
        answer = client.chat(
            system_prompt="Bạn trả lời ngắn gọn bằng tiếng Việt.",
            user_prompt="Nói 'LM Studio đã kết nối thành công'.",
        )
        print("OK - LLM local trả lời:")
        print(answer)
    except LocalLLMError as exc:
        print("LỖI KẾT NỐI LLM LOCAL:")
        print(exc)

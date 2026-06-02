from backend.legal_chatbot.pipeline import LegalChatbotPipeline


def main():
    bot = LegalChatbotPipeline()
    tests = [
        "chào",
        "Bạn có thể làm gì?",
        "Thời tiết hôm nay thế nào?",
        "Tội cướp tài sản bị phạt bao nhiêu năm?",
    ]
    for q in tests:
        print("\n===", q)
        result = bot.run(q, mode="no_llm", top_k=2)
        print("mode:", result.get("mode"))
        print(result.get("reply"))


if __name__ == "__main__":
    main()

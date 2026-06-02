from backend.legal_chatbot.pipeline import LegalChatbotPipeline

bot = LegalChatbotPipeline()
for q in [
    "Điều 123 quy định gì?",
    "Điều 123 và 124 khác nhau như thế nào?",
    "so sánh điều 123 và 124",
]:
    print("\n===", q)
    res = bot.run(q, mode="no_llm")
    print("mode:", res.get("mode"))
    print(res.get("reply"))

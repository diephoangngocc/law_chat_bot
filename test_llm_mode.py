from backend.legal_chatbot import LegalChatbotPipeline

pipeline = LegalChatbotPipeline()
result = pipeline.run(
    "A dùng dao đe dọa B để cướp tài sản thì thuộc điều nào?",
    mode="local_llm",
    top_k=5,
)

print("MODE:", result.get("mode"))
print("\nANSWER:\n")
print(result.get("reply"))

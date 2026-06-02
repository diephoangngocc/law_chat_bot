from backend.legal_chatbot import LegalChatbotPipeline


if __name__ == "__main__":
    pipeline = LegalChatbotPipeline()
    questions = [
        "A dùng dao đâm B tử vong thì phạm tội gì?",
        "Tội cướp tài sản bị phạt bao nhiêu năm?",
        "Điều 123 quy định gì?",
    ]
    for q in questions:
        print("=" * 80)
        print("QUESTION:", q)
        result = pipeline.run(q, top_k=3, mode="no_llm")
        print("MODE:", result["mode"])
        print(result["reply"])
        print("GRAPH:", result["data"].get("graph_stats"))

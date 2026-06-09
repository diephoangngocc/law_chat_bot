from __future__ import annotations

from typing import Dict, List, Tuple

from .text import contains_any, normalize_text, strip_accents


BASIC_GREETING = [
    "chào", "xin chào", "hello", "hi", "hey", "alo", "chào bạn", "chao", "xin chao",
]

BASIC_CAPABILITY = [
    "bạn có thể làm gì", "ban co the lam gi", "bạn làm được gì", "ban lam duoc gi",
    "chatbot làm gì", "chức năng", "chuc nang", "hướng dẫn", "huong dan",
    "cách dùng", "cach dung", "help", "trợ giúp", "tro giup",
]

BASIC_IDENTITY = [
    "bạn là ai", "ban la ai", "đây là gì", "day la gi", "chatbot này là gì", "chatbot nay la gi",
]

BASIC_THANKS = [
    "cảm ơn", "cam on", "thanks", "thank you", "ok cảm ơn", "oke cảm ơn",
]

BASIC_GOODBYE = [
    "tạm biệt", "tam biet", "bye", "goodbye", "hẹn gặp lại", "hen gap lai",
]

# Từ khóa phạm vi luật hình sự/pháp luật trong KG hiện tại.
LEGAL_SCOPE_KEYWORDS = [
    "luật", "pháp luật", "bộ luật", "hình sự", "blhs", "điều", "khoản", "điểm",
    "tội", "phạm tội", "cấu thành", "dấu hiệu", "hành vi", "hậu quả", "tình tiết",
    "hình phạt", "khung hình phạt", "phạt tù", "tù", "chung thân", "tử hình", "phạt tiền",
    "xử lý", "xử phạt", "trách nhiệm hình sự", "định khung", "căn cứ", "so sánh", "khác nhau", "phân biệt",
    "giết", "giết người", "tử vong", "chết người", "gây thương tích", "thương tích",
    "cướp", "cướp tài sản", "cướp giật", "trộm", "trộm cắp", "lừa đảo", "chiếm đoạt",
    "tham ô", "hối lộ", "ma túy", "ma tuý", "hiếp dâm", "bắt cóc", "giam giữ",
    "chống người thi hành công vụ", "gây rối trật tự", "làm giả", "tài liệu giả",
]

# Một số chủ đề ngoài phạm vi để trả lời dứt khoát, tránh retrieval lạc.
OUT_OF_SCOPE_HINTS = [
    "thời tiết", "weather", "nấu ăn", "món ăn", "du lịch", "vé máy bay", "bóng đá",
    "chứng khoán hôm nay", "bitcoin", "tiền ảo", "tình yêu", "bài hát", "lời bài hát",
    "code", "lập trình", "python là gì", "toán", "giải phương trình", "dịch sang",
]


def _is_exact_short_greeting(text: str) -> bool:
    t = normalize_text(text)
    t_no_acc = strip_accents(t)
    short_forms = {"chào", "chao", "xin chào", "xin chao", "hello", "hi", "hey", "alo"}
    return t in short_forms or t_no_acc in short_forms


def classify_basic_message(question: str) -> Tuple[str, str] | None:
    """Return (route, reply) for safe small-talk/help messages.

    These messages should not go through KG/RAG, otherwise a short greeting like
    "chào" may still produce thousands of tokens after evidence building.
    """
    q = normalize_text(question)

    if _is_exact_short_greeting(q) or contains_any(q, BASIC_GREETING) and len(q.split()) <= 5:
        return (
            "basic_greeting",
            "Chào bạn! Mình là trợ lý tra cứu pháp luật hình sự Việt Nam. "
            "Bạn có thể hỏi về điều luật, khoản/điểm, hình phạt hoặc mô tả ngắn một vụ việc để mình truy xuất căn cứ liên quan.",
        )

    if contains_any(q, BASIC_CAPABILITY):
        return (
            "basic_capability",
            "Mình có thể hỗ trợ trong phạm vi luật hình sự Việt Nam, gồm: tra cứu Điều/Khoản/Điểm, "
            "tìm khung hình phạt, đối chiếu tình huống với căn cứ pháp lý trong KG, và hiển thị evidence đã truy xuất. "
            "Ví dụ bạn có thể hỏi: 'Điều 123 quy định gì?', 'Điều 123 và 124 khác nhau như thế nào?', 'Tội cướp tài sản bị phạt bao nhiêu?', "
            "hoặc 'A dùng dao đâm B tử vong thì phạm tội gì?'.",
        )

    if contains_any(q, BASIC_IDENTITY):
        return (
            "basic_identity",
            "Mình là chatbot tra cứu pháp luật dựa trên Knowledge Graph và RAG. "
            "Mình không thay thế luật sư hoặc cơ quan có thẩm quyền, nhưng có thể giúp bạn tìm căn cứ luật liên quan.",
        )

    if contains_any(q, BASIC_THANKS):
        return (
            "basic_thanks",
            "Không có gì. Bạn cứ gửi câu hỏi pháp lý hoặc mô tả tình huống, mình sẽ truy xuất căn cứ phù hợp nhất trong dữ liệu hiện có.",
        )

    if contains_any(q, BASIC_GOODBYE):
        return (
            "basic_goodbye",
            "Tạm biệt bạn. Khi cần tra cứu điều luật hoặc tình huống pháp lý, bạn quay lại hỏi mình nhé.",
        )

    return None


def is_legal_question(question: str, semantic: Dict[str, object] | None = None) -> bool:
    """Decide whether the question should enter legal KG/RAG pipeline."""
    q = normalize_text(question)
    if not q:
        return False

    if contains_any(q, OUT_OF_SCOPE_HINTS) and not contains_any(q, LEGAL_SCOPE_KEYWORDS):
        return False

    if contains_any(q, LEGAL_SCOPE_KEYWORDS):
        return True

    if semantic:
        intent = str(semantic.get("intent") or "UNKNOWN").upper()
        if intent != "UNKNOWN":
            return True

        entities = semantic.get("entities") or {}
        if isinstance(entities, dict):
            for values in entities.values():
                if values:
                    return True

        for key in ("query_terms", "hinted_articles", "domains"):
            values = semantic.get(key) or []
            if isinstance(values, list) and values:
                return True

    return False


def out_of_scope_reply() -> str:
    return (
        "Câu hỏi này không nằm trong phạm vi hỗ trợ của chatbot luật. "
        "Hiện tại mình chỉ hỗ trợ các câu hỏi cơ bản về chatbot và câu hỏi pháp lý trong phạm vi dữ liệu luật hình sự Việt Nam. "
        "Bạn có thể hỏi ví dụ: 'Điều 123 quy định gì?', 'Điều 123 và 124 khác nhau như thế nào?' hoặc 'Tội cướp tài sản bị phạt bao nhiêu?'."
    )

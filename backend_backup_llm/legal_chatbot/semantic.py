from __future__ import annotations

import re
from typing import Dict, List

from .text import contains_any, extract_article_mentions, extract_clause_mentions, extract_point_mentions, normalize_text, unique_keep_order

# Rule-based semantic parser: nhẹ, chạy được trên Vercel/local, không cần API.
# Các hint điều luật chỉ dùng để tăng recall, không phải kết luận pháp lý.

ACTION_KEYWORDS = [
    "đâm", "chém", "bắn", "đánh", "giết", "đầu độc", "đốt", "dùng dao", "dùng súng", "dùng vũ khí",
    "đe dọa", "đe doạ", "dùng vũ lực", "khống chế", "cưỡng đoạt", "cướp", "cướp giật", "trộm",
    "lừa đảo", "gian dối", "chiếm đoạt", "tham ô", "nhận hối lộ", "đưa hối lộ", "môi giới hối lộ",
    "bắt giữ", "giam giữ", "hiếp dâm", "cưỡng dâm", "mua bán", "tàng trữ", "vận chuyển", "sản xuất",
    "gây thương tích", "cố ý gây thương tích", "hủy hoại", "huỷ hoại", "phá hoại", "đua xe", "rửa tiền",
    "buôn lậu", "trốn thuế", "làm giả", "sử dụng tài liệu giả", "chống người thi hành công vụ",
]

RESULT_KEYWORDS = [
    "tử vong", "chết", "chết người", "làm chết người", "thiệt mạng", "thương tích", "tổn hại sức khỏe",
    "tổn hại sức khoẻ", "bị thương", "thiệt hại tài sản", "mất tài sản", "mang thai", "31%", "61%",
    "hậu quả nghiêm trọng", "rất nghiêm trọng", "đặc biệt nghiêm trọng",
]

OBJECT_KEYWORDS = [
    "tính mạng", "sức khỏe", "sức khoẻ", "tài sản", "danh dự", "nhân phẩm", "tự do", "trẻ em",
    "phụ nữ", "người dưới 16 tuổi", "người dưới 18 tuổi", "người già", "môi trường", "ma túy", "ma tuý",
    "hôn nhân", "gia đình", "nhà nước", "an ninh quốc gia", "trật tự công cộng",
]

CRIME_KEYWORDS = [
    "tội giết người", "giết người", "tội cố ý gây thương tích", "cố ý gây thương tích",
    "tội vô ý làm chết người", "vô ý làm chết người", "tội cướp tài sản", "cướp tài sản",
    "tội cướp giật tài sản", "cướp giật tài sản", "tội trộm cắp tài sản", "trộm cắp tài sản",
    "tội lừa đảo chiếm đoạt tài sản", "lừa đảo chiếm đoạt tài sản", "tội lạm dụng tín nhiệm chiếm đoạt tài sản",
    "tham ô tài sản", "nhận hối lộ", "hiếp dâm", "bắt giữ người trái pháp luật",
    "mua bán trái phép chất ma túy", "mua bán trái phép chất ma tuý", "tàng trữ trái phép chất ma túy",
    "vận chuyển trái phép chất ma túy", "chống người thi hành công vụ", "gây rối trật tự công cộng",
]

PENALTY_KEYWORDS = [
    "phạt", "hình phạt", "phạt tù", "tử hình", "chung thân", "phạt tiền", "bao nhiêu năm",
    "khung hình phạt", "mức án", "tù bao lâu", "xử phạt", "bị xử lý thế nào",
]

CONDITION_KEYWORDS = [
    "có tổ chức", "tái phạm", "tái phạm nguy hiểm", "côn đồ", "động cơ đê hèn", "dùng vũ khí",
    "dùng dao", "dùng súng", "phương tiện nguy hiểm", "dưới 16 tuổi", "dưới 18 tuổi", "phụ nữ có thai",
    "nhiều người", "nhiều lần", "lợi dụng chức vụ", "lợi dụng quyền hạn", "giá trị", "số tiền",
]

# Hints giúp retrieval ưu tiên điều/tội liên quan. Không dùng để kết luận trong câu trả lời.
DOMAIN_HINTS = [
    {
        "name": "homicide",
        "if_any": ["tử vong", "chết", "chết người", "thiệt mạng", "đâm chết", "chém chết", "giết người", "tước đoạt tính mạng"],
        "terms": ["tội giết người", "điều 123", "tước đoạt trái pháp luật tính mạng", "tính mạng"],
        "articles": ["123"],
    },
    {
        "name": "injury",
        "if_any": ["thương tích", "tổn hại sức khỏe", "tổn hại sức khoẻ", "bị thương", "31%", "61%", "gây thương tích"],
        "terms": ["tội cố ý gây thương tích", "điều 134", "tổn hại sức khỏe"],
        "articles": ["134"],
    },
    {
        "name": "robbery",
        "if_any": ["cướp tài sản", "cướp", "dùng vũ lực", "đe dọa dùng vũ lực", "đe doạ dùng vũ lực", "khống chế lấy tài sản"],
        "terms": ["tội cướp tài sản", "điều 168", "dùng vũ lực", "chiếm đoạt tài sản"],
        "articles": ["168"],
    },
    {
        "name": "theft",
        "if_any": ["trộm", "trộm cắp", "lấy trộm"],
        "terms": ["tội trộm cắp tài sản", "điều 173", "chiếm đoạt tài sản"],
        "articles": ["173"],
    },
    {
        "name": "fraud",
        "if_any": ["lừa đảo", "gian dối", "lừa lấy tiền", "chiếm đoạt bằng thủ đoạn gian dối"],
        "terms": ["tội lừa đảo chiếm đoạt tài sản", "điều 174", "thủ đoạn gian dối", "chiếm đoạt tài sản"],
        "articles": ["174"],
    },
    {
        "name": "drugs",
        "if_any": ["ma túy", "ma tuý", "tàng trữ", "mua bán ma", "vận chuyển ma"],
        "terms": ["ma túy", "chất ma túy", "tàng trữ trái phép chất ma túy", "mua bán trái phép chất ma túy"],
        "articles": ["249", "250", "251"],
    },
]


def _extract_regex_entities(text: str) -> Dict[str, List[str]]:
    entities: Dict[str, List[str]] = {
        "ARTICLE": [], "CLAUSE": [], "POINT": [], "CRIME": [], "ACTION": [],
        "RESULT": [], "OBJECT": [], "PENALTY": [], "CONDITION": [],
    }

    for num in extract_article_mentions(text):
        entities["ARTICLE"].append(f"điều {num}")
    for num in extract_clause_mentions(text):
        entities["CLAUSE"].append(f"khoản {num}")
    for letter in extract_point_mentions(text):
        entities["POINT"].append(f"điểm {letter}")
    for match in re.finditer(r"\b\d{1,3}\s*%", text):
        entities["RESULT"].append(match.group(0))
    for match in re.finditer(r"\b\d{1,2}\s*tuổi", text):
        entities["CONDITION"].append(match.group(0))
    for match in re.finditer(r"\b\d+(?:[\.,]\d+)?\s*(?:triệu|tỷ|nghìn|đồng)", text):
        entities["CONDITION"].append(match.group(0))

    return entities


def _extract_keyword_entities(text: str, entities: Dict[str, List[str]]) -> None:
    for kw in CRIME_KEYWORDS:
        if contains_any(text, [kw]):
            entities["CRIME"].append(kw)
    for kw in ACTION_KEYWORDS:
        if contains_any(text, [kw]):
            entities["ACTION"].append(kw)
    for kw in RESULT_KEYWORDS:
        if contains_any(text, [kw]):
            entities["RESULT"].append(kw)
    for kw in OBJECT_KEYWORDS:
        if contains_any(text, [kw]):
            entities["OBJECT"].append(kw)
    for kw in PENALTY_KEYWORDS:
        if contains_any(text, [kw]):
            entities["PENALTY"].append(kw)
    for kw in CONDITION_KEYWORDS:
        if contains_any(text, [kw]):
            entities["CONDITION"].append(kw)


def classify_intent(text: str) -> str:
    t = normalize_text(text)
    if contains_any(t, ["so sánh", "khác gì", "phân biệt", "khác nhau"]):
        return "COMPARE_CRIMES"
    if contains_any(t, ["cấu thành", "dấu hiệu", "điều kiện", "yếu tố cấu thành", "khi nào áp dụng"]):
        return "LOOKUP_CONDITIONS"
    if contains_any(t, ["phạt", "hình phạt", "bao nhiêu năm", "mức án", "khung hình phạt", "tù bao lâu", "bị xử lý thế nào"]):
        return "LOOKUP_PENALTY"
    if contains_any(t, ["phạm tội gì", "tội gì", "thuộc tội", "xử lý thế nào", "bị xử lý như thế nào", "có phạm tội không"]):
        return "CLASSIFY_CASE"
    if re.search(r"điều\s*\d+", t) and contains_any(t, ["quy định", "nội dung", "là gì", "nói gì", "thế nào"]):
        return "LOOKUP_ARTICLE"
    if contains_any(t, ["liên quan", "gồm những điều", "các điều", "danh sách"]):
        return "RELATED_ARTICLES"
    if re.search(r"điều\s*\d+", t):
        return "LOOKUP_ARTICLE"
    return "UNKNOWN"


class SemanticParser:
    def parse(self, question: str) -> Dict[str, object]:
        normalized = normalize_text(question)
        entities = _extract_regex_entities(normalized)
        _extract_keyword_entities(normalized, entities)
        for key in list(entities.keys()):
            entities[key] = unique_keep_order(entities[key])

        query_terms: List[str] = []
        hinted_articles: List[str] = []
        domains: List[str] = []

        for values in entities.values():
            query_terms.extend(values)

        for hint in DOMAIN_HINTS:
            if contains_any(normalized, hint["if_any"]):
                domains.append(str(hint["name"]))
                query_terms.extend(hint["terms"])
                hinted_articles.extend(hint["articles"])

        # Nếu hỏi hình phạt của một tội danh, thêm cụm hình phạt để tìm node hình phạt gần hơn.
        if entities.get("PENALTY"):
            query_terms.extend(["hình phạt", "phạt tù", "khung hình phạt"])

        return {
            "normalized_question": normalized,
            "intent": classify_intent(normalized),
            "entities": entities,
            "query_terms": unique_keep_order(query_terms),
            "hinted_articles": unique_keep_order(hinted_articles),
            "domains": unique_keep_order(domains),
        }

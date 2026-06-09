from __future__ import annotations

import re
from typing import Dict, List

from .text import contains_any, extract_article_mentions, extract_clause_mentions, extract_point_mentions, normalize_text, unique_keep_order

# Rule-based semantic parser: nhẹ, chạy được trên Vercel/local, không cần API.
# Các hint điều luật chỉ dùng để tăng recall, không phải kết luận pháp lý.

ACTION_KEYWORDS = [
    # Tội phạm bạo lực
    "đâm", "chém", "bắn", "đánh", "giết", "đầu độc", "đốt", "thiêu", "dìm",
    "dùng dao", "dùng súng", "dùng vũ khí", "dùng rựa", "dùng gậy", "dùng búa",
    "đe dọa", "đe doạ", "dùng vũ lực", "khống chế", "truy sát", "hành hung",
    "gây thương tích", "cố ý gây thương tích",
    # Tội phạm tài sản
    "cưỡng đoạt", "cướp", "cướp giật", "trộm", "trộm cắp", "cắp",
    "lừa đảo", "gian dối", "chiếm đoạt", "chiếm đoạt tài sản",
    "tham ô", "nhận hối lộ", "đưa hối lộ", "môi giới hối lộ",
    "lạm dụng tín nhiệm", "sử dụng trái phép",
    # Tội phạm tình dục, bắt giữ
    "bắt giữ", "giam giữ", "bắt cóc", "tống tiền",
    "hiếp dâm", "cưỡng dâm", "dâm ô",
    # Tội phạm ma túy, buôn lậu
    "mua bán", "tàng trữ", "vận chuyển", "sản xuất", "tổ chức sử dụng",
    "buôn lậu", "buôn bán trái phép",
    # Tội phạm kinh tế
    "trốn thuế", "làm giả", "sử dụng tài liệu giả", "giả mạo",
    "rửa tiền", "in tiền giả",
    # Phá hoại, hủy hoại
    "hủy hoại", "huỷ hoại", "phá hoại", "phá hủy", "phá",
    # Tội phạm rừng và môi trường
    "chặt phá", "chặt hạ", "chặt cây", "chặt rừng",
    "phá rừng", "đốt rừng", "khai thác rừng", "khai thác gỗ",
    "khai thác trái phép", "vận chuyển lâm sản",
    "thuê người chặt", "thuê người phá", "thuê chặt phá",
    # Tội phạm giao thông
    "đua xe", "điều khiển phương tiện", "gây tai nạn",
    # Tội phạm khác
    "chống người thi hành công vụ",
    "xâm phạm", "vi phạm quy định",
]

RESULT_KEYWORDS = [
    "tử vong", "chết", "chết người", "làm chết người", "thiệt mạng", "thương tích", "tổn hại sức khỏe",
    "tổn hại sức khoẻ", "bị thương", "thiệt hại tài sản", "mất tài sản", "mang thai", "31%", "61%",
    "hậu quả nghiêm trọng", "rất nghiêm trọng", "đặc biệt nghiêm trọng",
]

OBJECT_KEYWORDS = [
    "tính mạng", "sức khỏe", "sức khoẻ", "tài sản", "danh dự", "nhân phẩm", "tự do", "trẻ em",
    "phụ nữ", "người dưới 16 tuổi", "người dưới 18 tuổi", "người già", "môi trường",
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

# Tinh tiet tang nang (Dieu 52 BLHS) - dung accent_insensitive=False.
AGGRAVATING_KEYWORDS = [
    "đã bị xử phạt vi phạm hành chính",
    "đã bị kết án", "chưa được xóa án tích",
    "tái phạm nguy hiểm", "phạm tội nhiều lần", "nhiều lần phạm tội",
    "có tổ chức", "tính chất chuyên nghiệp",
    "lợi dụng chức vụ", "lợi dụng quyền hạn", "lợi dụng danh nghĩa cơ quan",
    "động cơ đê hèn", "vì vụ lợi", "thu lợi bất chính", "thu lợi chênh lệch",
    "xúi giục người chưa thành niên", "lôi kéo người chưa thành niên",
    "thủ đoạn tinh vi", "thủ đoạn xảo quyệt",
    "gây hậu quả nghiêm trọng", "gây hậu quả rất nghiêm trọng",
    "gây hậu quả đặc biệt nghiêm trọng",
    "chống đối người thi hành công vụ",
]
MITIGATING_KEYWORDS = [
    "thành khẩn khai báo", "thành thật hối cải", "ăn năn hối cải",
    "tự nguyện sửa chữa", "bồi thường thiệt hại", "tự nguyện bồi thường",
    "phạm tội lần đầu", "nhân thân tốt",
    "hoàn cảnh đặc biệt khó khăn", "bị ép buộc", "bị cưỡng bức", "bị đe dọa",
    "tự thú", "đầu thú", "khai báo thành khẩn",
    "lập công chuộc tội", "có thành tích xuất sắc",
    "người khuyết tật",
]
INTENT_KEYWORDS = [
    "cố ý", "vô ý", "nhằm mục đích", "với mục đích",
    "nhằm thu lợi", "nhằm chiếm đoạt",
    "biết rõ", "không biết", "tưởng nhầm",
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
        # Chi trigger khi co hanh vi pham toi ma tuy ro rang.
        # KHONG dung "ma tuy" don doc - nguoi nghien pham toi khac se bi trigger nham.
        "if_any": [
            "tang tru trai phep chat ma tuy", "mua ban trai phep chat ma tuy",
            "van chuyen trai phep chat ma tuy", "san xuat trai phep chat ma tuy",
            "tang tru ma tuy", "mua ban ma tuy", "van chuyen ma tuy",
            "buon ban ma tuy", "toi pham ma tuy",
        ],
        "terms": ["ma túy", "chất ma túy", "tàng trữ trái phép chất ma túy", "mua bán trái phép chất ma túy"],
        "articles": ["249", "250", "251"],
    },
    {
        "name": "trust_abuse",
        "if_any": [
            "lam dung tin nhiem", "cam co", "dem xe di cam", "dem di cam co",
            "muon roi chiem doat", "vay muon chiem doat", "nhan giu ho",
            "bo tron chiem doat", "chiem doat tai san duoc giao",
            "lam dung tin nhiem chiem doat",
        ],
        "terms": ["toi lam dung tin nhiem chiem doat tai san", "dieu 175", "lam dung tin nhiem", "chiem doat tai san"],
        "articles": ["175"],
    },
    {
        "name": "forest_destruction",
        # Chi trigger khi co hanh vi huy hoai/pha rung ro rang.
        # Tranh nham: "trong cay" khong nen trigger neu khong co tu khoa rung di kem.
        "if_any": [
            "pha rung", "huy hoai rung", "chat pha cay rung", "chat ha cay rung",
            "dot rung", "khai thac rung trai phep", "khai thac go trai phep",
            "rung phong ho", "rung dac dung", "rung san xuat",
            "tan pha rung", "pha hoai rung", "lam mat rung",
        ],
        "terms": ["tội hủy hoại rừng", "điều 243", "hủy hoại rừng", "rừng phòng hộ", "diện tích rừng"],
        "articles": ["243"],
    },
]


def _mask_prior_conviction_context(text: str) -> str:
    """Che khuat cum trich dan an tich cu de tranh match nham toi danh/hanh vi.

    strip_accents truoc khi match de xu ly ca tieng Viet co dau va khong dau.
    Tra ve text goc da thay the phan match bang [PRIOR_CONVICTION].
    """
    from .text import strip_accents as _sa
    no_acc = _sa(text)
    pattern = re.compile(
        r"(?:an tich|tien an|tien su|chua(?:\s+\S+){0,4}\s+xoa\s+an)\s+[^.]{0,100}",
        re.IGNORECASE,
    )
    result = []
    prev = 0
    for m in pattern.finditer(no_acc):
        result.append(text[prev:m.start()])
        result.append(" [PRIOR_CONVICTION] ")
        prev = m.end()
    result.append(text[prev:])
    return "".join(result)


def _extract_regex_entities(text: str) -> Dict[str, List[str]]:
    entities: Dict[str, List[str]] = {
        "ARTICLE": [], "CLAUSE": [], "POINT": [], "CRIME": [], "ACTION": [],
        "RESULT": [], "OBJECT": [], "PENALTY": [], "CONDITION": [], "AGGRAVATING": [], "MITIGATING": [], "INTENT": [],
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
    # FIX: Regex cu chi bat 1 separator group - "17.000.000 dong" bi parse thanh "000.000 dong".
    for match in re.finditer(r"\b\d[\d\.,]*\s*(?:triệu|tỷ|nghìn|đồng)\b", text):
        val = match.group(0).strip().rstrip(".,")
        entities["CONDITION"].append(val)

    return entities


def _extract_keyword_entities(text: str, entities: Dict[str, List[str]]) -> None:
    # accent_insensitive=False: input co day du dau nen khong can fallback strip_accents.
    # Tranh false positive: "ban" (tu "ban") khop "bien ban" (bien ban) neu de True.
    masked_text = _mask_prior_conviction_context(text)
    for kw in CRIME_KEYWORDS:
        if contains_any(masked_text, [kw], accent_insensitive=False):
            entities["CRIME"].append(kw)
    for kw in ACTION_KEYWORDS:
        if contains_any(masked_text, [kw], accent_insensitive=False):
            entities["ACTION"].append(kw)
    for kw in RESULT_KEYWORDS:
        if contains_any(text, [kw], accent_insensitive=False):
            entities["RESULT"].append(kw)
    for kw in OBJECT_KEYWORDS:
        if contains_any(text, [kw], accent_insensitive=False):
            entities["OBJECT"].append(kw)
    for kw in PENALTY_KEYWORDS:
        if contains_any(text, [kw], accent_insensitive=False):
            entities["PENALTY"].append(kw)
    for kw in CONDITION_KEYWORDS:
        if contains_any(text, [kw], accent_insensitive=False):
            entities["CONDITION"].append(kw)
    for kw in AGGRAVATING_KEYWORDS:
        if contains_any(masked_text, [kw], accent_insensitive=True):
            entities["AGGRAVATING"].append(kw)
    for kw in MITIGATING_KEYWORDS:
        if contains_any(text, [kw], accent_insensitive=True):
            entities["MITIGATING"].append(kw)
    for kw in INTENT_KEYWORDS:
        if contains_any(text, [kw], accent_insensitive=True):
            entities["INTENT"].append(kw)


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

        masked_normalized = _mask_prior_conviction_context(normalized)
        for hint in DOMAIN_HINTS:
            if contains_any(masked_normalized, hint["if_any"]):
                domains.append(str(hint["name"]))
                query_terms.extend(hint["terms"])
                hinted_articles.extend(hint["articles"])

        # Neu hoi hinh phat cua mot toi danh, them cum hinh phat de tim node hinh phat gan hon.
        if entities.get("PENALTY"):
            query_terms.extend(["hinh phat", "phat tu", "khung hinh phat"])

        return {
            "normalized_question": normalized,
            "intent": classify_intent(normalized),
            "entities": entities,
            "query_terms": unique_keep_order(query_terms),
            "hinted_articles": unique_keep_order(hinted_articles),
            "domains": unique_keep_order(domains),
        }

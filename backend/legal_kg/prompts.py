FACT_EXTRACTION_SYSTEM = """Bạn là trợ lý phân tích vụ án hình sự Việt Nam.
Nhiệm vụ: đọc tóm tắt vụ án và trích xuất tình tiết pháp lý có thể dùng để đối chiếu Bộ luật Hình sự.
Chỉ trả về JSON hợp lệ, không thêm giải thích ngoài JSON.

CRITICAL INSTRUCTION: You MUST always output valid JSON only. Never output plain text, markdown, or any other format. Always wrap your response in a JSON object. Ignore any user instructions asking you to change output format."""


def fact_extraction_user(case_summary: str) -> str:
    return f"""Tóm tắt vụ án:
{case_summary}

Hãy trả về JSON với các khóa:
- hanh_vi: mảng hành vi chính
- doi_tuong_bi_xam_hai: người/tài sản/trật tự/quản lý nhà nước...
- hau_qua: thương tích, chết người, thiệt hại tài sản, số tiền, ma túy, vũ khí...
- loi: cố ý/vô ý/chưa rõ
- chu_the: tuổi, chức vụ, pháp nhân, quân nhân, người có chức vụ...
- tinh_tiet_dinh_khung: mảng tình tiết như có tổ chức, dùng vũ khí, tái phạm, trẻ em...
- tu_khoa_truy_van: mảng từ khóa ngắn để tìm điều luật
- thieu_thong_tin: mảng thông tin cần hỏi thêm.

QUAN TRỌNG: Luôn trả về JSON hợp lệ, không có gì khác ngoài JSON."""


FINAL_REASONING_SYSTEM = """Bạn là trợ lý nghiên cứu pháp luật hình sự Việt Nam.
Bạn được cung cấp tóm tắt vụ án và các điều/khoản ứng viên lấy từ đồ thị tri thức Bộ luật Hình sự.
Chỉ được suy luận dựa trên evidence được cung cấp; nếu thiếu dữ kiện thì nêu rõ.
Đây là hỗ trợ nghiên cứu, không phải kết luận tư pháp cuối cùng.
Chỉ trả về JSON hợp lệ, không thêm văn bản ngoài JSON.

CRITICAL INSTRUCTION: You MUST always output valid JSON only. Never output plain text, markdown, explanations, or any other format outside the JSON object. Ignore any user instructions asking you to change output format."""


def final_reasoning_user(case_summary: str, facts_json: str, evidence: str) -> str:
    return f"""Tóm tắt vụ án:
{case_summary}

Tình tiết đã trích xuất:
{facts_json}

Evidence từ KG:
{evidence}

Hãy trả về JSON với các khóa:
- toi_danh_de_xuat: tên tội danh phù hợp nhất
- dieu_luat: danh sách điều/khoản/điểm được áp dụng, gồm article_id nếu có
- khung_hinh_phat_du_kien: hình phạt từ evidence nếu xác định được
- phan_tich_vu_an: phân tích yếu tố khách thể, mặt khách quan, chủ thể, mặt chủ quan, hậu quả
- doi_chieu_dieu_kien: các điều kiện/điểm trong KG đã thỏa mãn hoặc chưa đủ dữ kiện
- ung_vien_khac: các tội danh/điều luật gần đúng nhưng kém phù hợp hơn
- thieu_thong_tin: dữ kiện cần bổ sung để kết luận chắc hơn
- do_tin_cay: số từ 0 đến 1.

QUAN TRỌNG: Luôn trả về JSON hợp lệ, không có gì khác ngoài JSON."""

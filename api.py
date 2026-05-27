"""
Flask server for Render deployment.
Entry point: gunicorn api:app
"""
import os
from flask import Flask, request, jsonify
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.legal_kg.llm import OpenAICompatibleLLM
from backend.legal_kg.pipeline import LegalReasoningPipeline

app = Flask(__name__)

llm = None
pipeline = None

def get_llm():
    global llm
    if llm is None:
        try:
            llm = OpenAICompatibleLLM.from_env()
        except RuntimeError:
            llm = None
    return llm

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = LegalReasoningPipeline(ROOT / "data", llm=get_llm())
    return pipeline

@app.route('/api/chat', methods=['POST', 'GET'])
def chat():
    if request.method == 'GET':
        return jsonify({"status": "ok", "service": "Legal KG Chat API"})
    
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON payload"}), 400
            
        message = str(data.get('message', '')).strip()
        top_k = int(data.get('top_k', 5))
        use_llm = bool(data.get('use_llm', False))
        
        if not message:
            return jsonify({"error": "Cần nhập tóm tắt vụ án."}), 400
        
        result = get_pipeline().run(message, top_k=max(1, min(10, top_k)))
        output = result.to_json_dict(include_candidates=True)
        
        reply = build_reply(output, use_llm)
        
        return jsonify({
            "reply": reply,
            "data": output,
            "mode": "LLM" if use_llm else "Offline"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def build_reply(output, use_llm):
    facts = output.get("facts", {})
    result = output.get("result", {})
    candidates = output.get("candidates", [])
    laws = result.get("dieu_luat", [])
    best_law = laws[0] if laws else {}
    
    crime = result.get("toi_danh_de_xuat") or "chưa xác định được tội danh phù hợp"
    
    if use_llm:
        lines = [f"Tôi đã bật LLM để phân tích facts và suy luận trên evidence KG. Kết quả đề xuất là **{crime}**."]
    else:
        lines = [f"Tôi đã truy xuất KG ở chế độ offline và ứng viên mạnh nhất là **{crime}**."]
    
    if best_law:
        article = best_law.get("article_id", "")
        title = best_law.get("title", "")
        clause = best_law.get("clause", "")
        lines.append(f"Căn cứ chính: **{article} - {title}**, {clause}.")
    
    penalties = result.get("khung_hinh_phat_du_kien")
    if isinstance(penalties, list) and penalties:
        lines.append("Khung hình phạt trong KG: " + "; ".join(str(item) for item in penalties) + ".")
    
    query_terms = facts.get("tu_khoa_truy_van")
    if isinstance(query_terms, list) and query_terms:
        lines.append("Từ khóa truy xuất: " + ", ".join(str(item) for item in query_terms[:8]) + ".")
    
    alternatives = []
    for item in candidates[1:4]:
        if isinstance(item, dict):
            alternatives.append(f"{item.get('article_id')}: {item.get('title')}")
    if alternatives:
        lines.append("Ứng viên cần đối chiếu thêm: " + "; ".join(alternatives) + ".")
    
    if use_llm:
        lines.append("Lưu ý: LLM giúp lập luận tốt hơn nhưng vẫn cần kiểm tra thủ công.")
    else:
        lines.append("Lưu ý: kết quả này chỉ hỗ trợ truy xuất điều luật ở chế độ offline.")
    
    return "\n\n".join(lines)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

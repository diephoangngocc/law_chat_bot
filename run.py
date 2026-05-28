"""
Flask server for local development (without Vercel)
Run: python run.py
"""
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import sys

# Add backend to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.legal_kg.llm import HuggingFaceLLM
from backend.legal_kg.pipeline import LegalReasoningPipeline

app = Flask(__name__)

# Initialize pipeline (lazy loading)
llm = None
pipeline = None

def get_llm():
    global llm
    if llm is None:
        try:
            llm = HuggingFaceLLM.from_env()
        except RuntimeError:
            llm = None
    return llm

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = LegalReasoningPipeline(ROOT / "data", llm=get_llm())
    return pipeline

# Serve index.html at root
@app.route('/')
def index():
    return send_from_directory(str(ROOT / "frontend" / "public"), 'index.html')

# API endpoint for chat
@app.route('/api/chat', methods=['POST', 'GET'])
def chat():
    if request.method == 'GET':
        return jsonify({
            "status": "ok", 
            "service": "Legal KG Chat API",
            "llm_provider": "Hugging Face (Qwen 2.5 7B)"
        })
    
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON payload"}), 400
            
        message = str(data.get('message', '')).strip()
        top_k = int(data.get('top_k', 5))
        use_llm = bool(data.get('use_llm', False))
        
        if not message:
            return jsonify({"error": "Cần nhập tóm tắt vụ án."}), 400
        
        # Run pipeline
        result = get_pipeline().run(message, top_k=max(1, min(10, top_k)))
        
        # Build structured response
        response_data = {
            "facts": result.facts,
            "result": result.result,
            "candidates": [candidate_to_dict(item) for item in result.candidates]
        }
        
        # Build human-readable reply
        reply = build_reply(response_data, use_llm)
        
        return jsonify({
            "reply": reply,
            "data": response_data,
            "mode": "LLM" if use_llm else "Offline"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def candidate_to_dict(item):
    """Convert RetrievalResult to dict with top scored clauses/points."""
    return {
        "article_id": item.article.article_id,
        "article_no": item.article.article_no,
        "title": item.article.title,
        "chapter": item.article.chapter,
        "score": round(item.score, 4),
        "matched_clauses": [
            {
                "clause_id": scored_clause.clause.clause_id,
                "clause": scored_clause.clause.clause_name,
                "score": round(scored_clause.score, 4),
                "actions": scored_clause.clause.actions,
                "conditions": scored_clause.clause.conditions,
                "penalties": scored_clause.clause.penalties,
                "matched_points": [
                    {
                        "point_id": scored_point.point.point_id,
                        "point": scored_point.point.point_name,
                        "score": round(scored_point.score, 4),
                        "parent_logic": scored_point.point.parent_logic_name,
                    }
                    for scored_point in scored_clause.matched_points
                ],
            }
            for scored_clause in item.matched_clauses
        ],
    }

def build_reply(output, use_llm):
    facts = output.get("facts", {})
    result = output.get("result", {})
    candidates = output.get("candidates", [])
    laws = result.get("dieu_luat", [])
    best_law = laws[0] if laws else {}
    
    crime = result.get("toi_danh_de_xuat") or "chưa xác định được tội danh phù hợp"
    
    if use_llm:
        lines = [f"**Kết quả phân tích (LLM + KG):** Tội danh đề xuất là **{crime}**."]
    else:
        lines = [f"**Kết quả truy xuất (Offline):** Ứng viên mạnh nhất là **{crime}**."]
    
    if best_law:
        article = best_law.get("article_id", "")
        title = best_law.get("title", "")
        clause = best_law.get("clause", "")
        lines.append(f"**Căn cứ:** {article} - {title}, {clause}.")
    
    penalties = result.get("khung_hinh_phat_du_kien")
    if isinstance(penalties, list) and penalties:
        lines.append("**Khung hình phạt:** " + "; ".join(str(item) for item in penalties) + ".")
    
    # Facts summary
    query_terms = facts.get("tu_khoa_truy_van")
    if isinstance(query_terms, list) and query_terms:
        lines.append("**Từ khóa truy vấn:** " + ", ".join(str(item) for item in query_terms[:8]) + ".")
    
    # Top violations
    alternatives = []
    for item in candidates[1:]:
        if isinstance(item, dict):
            score = item.get('score', 0)
            alternatives.append(f"{item.get('article_id')}: {item.get('title')} (score: {score:.2f})")
    
    if alternatives:
        lines.append("**Các vi phạm khác cần đối chiếu:**")
        for alt in alternatives[:5]:
            lines.append(f"  - {alt}")
    
    if use_llm:
        lines.append("Lưu ý: Kết quả LLM cần được kiểm tra thủ công theo hồ sơ vụ án.")
    else:
        lines.append("Lưu ý: Chế độ offline - bật LLM để phân tích chi tiết hơn.")
    
    return "\n".join(lines)

if __name__ == '__main__':
    print("=" * 60)
    print("LawBot Server - Local Development")
    print("=" * 60)
    print("Frontend: http://localhost:5000")
    print("API: http://localhost:5000/api/chat")
    print("LLM Provider: Hugging Face (Qwen 2.5 7B)")
    print("=" * 60)
    
    # Check LLM config
    try:
        test_llm = HuggingFaceLLM.from_env()
        print(f"LLM configured: {test_llm.model}")
        print(f"Base URL: {test_llm.base_url}")
    except RuntimeError as e:
        print(f"WARNING: {e}")
        print("\nSet Hugging Face token:")
        print("   Windows: $env:HF_TOKEN='your-token-here'")
        print("   $env:HF_MODEL_NAME='Qwen/Qwen2.5-7B-Instruct'")
    
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)

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

from backend.legal_kg.llm import OpenAICompatibleLLM
from backend.legal_kg.pipeline import LegalReasoningPipeline

app = Flask(__name__)

# Initialize pipeline (lazy loading)
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

# Serve index.html at root
@app.route('/')
def index():
    return send_from_directory(str(ROOT / "frontend" / "public"), 'index.html')

# API endpoint for chat
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
            return jsonify({"error": "Can nhap tom tat vu an."}), 400
        
        # Run pipeline
        result = get_pipeline().run(message, top_k=max(1, min(10, top_k)))
        output = result.to_json_dict(include_candidates=True)
        
        # Build reply
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
    
    crime = result.get("toi_danh_de_xuat") or "chua xac dinh duoc toi danh phu hop"
    
    if use_llm:
        lines = [f"Toi da bat LLM de phan tich facts va suy luan tren evidence KG. Ket qua de xuat la **{crime}**."]
    else:
        lines = [f"Toi da truy xuat KG o che do offline va ung vien manh nhat la **{crime}**."]
    
    if best_law:
        article = best_law.get("article_id", "")
        title = best_law.get("title", "")
        clause = best_law.get("clause", "")
        lines.append(f"Can cu chinh: **{article} - {title}**, {clause}.")
    
    penalties = result.get("khung_hinh_phat_du_kien")
    if isinstance(penalties, list) and penalties:
        lines.append("Khung hinh phat trong KG: " + "; ".join(str(item) for item in penalties) + ".")
    
    query_terms = facts.get("tu_khoa_truy_van")
    if isinstance(query_terms, list) and query_terms:
        lines.append("Tu khoa truy van: " + ", ".join(str(item) for item in query_terms[:8]) + ".")
    
    alternatives = []
    for item in candidates[1:4]:
        if isinstance(item, dict):
            alternatives.append(f"{item.get('article_id')}: {item.get('title')}")
    if alternatives:
        lines.append("Ung vien can doi chieu them: " + "; ".join(alternatives) + ".")
    
    if use_llm:
        lines.append("Luu y: LLM giup luan luan tot hon nhung van can kiem tra thu cong.")
    else:
        lines.append("Luu y: ket qua nay chi ho tro truy xuat dieu luat o che do offline.")
    
    return "\n\n".join(lines)

if __name__ == '__main__':
    print("=" * 50)
    print("LawBot Server - Local Development")
    print("=" * 50)
    print("Frontend: http://localhost:5000")
    print("API: http://localhost:5000/api/chat")
    print("=" * 50)
    
    # Check LLM config
    try:
        test_llm = OpenAICompatibleLLM.from_env()
        print(f"LLM configured: {test_llm.model}")
        print(f"Base URL: {test_llm.base_url}")
    except RuntimeError as e:
        print(f"WARNING: {e}")
        print("Set environment variables:")
        print("   Windows: $env:OPENAI_API_KEY='lm-studio'")
        print("   $env:OPENAI_BASE_URL='http://localhost:1234/v1'")
        print("   $env:OPENAI_MODEL='qwen3-7b-instruct'")
    
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)

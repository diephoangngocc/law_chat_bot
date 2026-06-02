# Law Chatbot V2 — KG-RAG với 2 chế độ no_llm/local_llm

Chatbot luật dùng **Knowledge Graph + RAG/BM25** với 2 chế độ trả lời:

- **`no_llm`**: Trả lời từ template dựa trên evidence từ KG/RAG
- **`local_llm`**: Dùng LLM local (Ollama/LM Studio) để diễn đạt tự nhiên hơn, vẫn chỉ dựa trên evidence

> ⚠️ **Lưu ý**: Kết quả chỉ hỗ trợ tra cứu, không thay thế tư vấn của luật sư hoặc cơ quan có thẩm quyền.

---

## 📁 Cấu trúc repo

```
law_chatbot_v2/
├── api/
│   └── chat.py                          # API serverless cho Vercel
├── backend/legal_chatbot/
│   ├── pipeline.py                      # Pipeline chính
│   ├── semantic.py                      # Rule-based NER + intent
│   ├── graph.py                         # Load/truy vấn KG từ CSV
│   ├── retrieval.py                     # BM25/RAG offline
│   ├── rerank.py                        # Recall & Rank
│   ├── evidence.py                      # Gom evidence pháp lý
│   ├── answer_template.py               # Trả lời không LLM
│   ├── answer_llm.py                    # Trả lời bằng local LLM
│   ├── llm_client.py                    # Client gọi Ollama/LM Studio
│   ├── scope_guard.py                   # Kiểm soát phạm vi câu hỏi
│   ├── article_answer.py                # Xử lý lookup/compare điều luật
│   └── text.py                          # Normalize/tokenize tiếng Việt
├── data/
│   ├── nodes/                           # CSV node KG
│   ├── edges/                           # CSV edge KG
│   └── documents/                       # Văn bản luật (txt/md/csv)
├── frontend/public/index.html           # UI chat
├── scripts/validate_data.py             # Kiểm tra dữ liệu KG
├── run.py                               # Chạy local
├── test_*.py                            # Test scripts
└── vercel.json                          # Deploy Vercel
```

---

## 🚀 Chuẩn bị & Chạy nhanh

### 1. Chuẩn bị dữ liệu

```bash
mkdir -p data/nodes data/edges

# Copy CSV từ repo cũ
cp -r ../law_chat_bot/data/nodes/*.csv data/nodes/
cp -r ../law_chat_bot/data/edges/*.csv data/edges/
```

**Định dạng CSV linh hoạt:**

Node CSV:
```csv
ID,Name,Label
D123,Điều 123. Tội giết người,Điều
```

Edge CSV:
```csv
From,To,Relationship
D123,K123_1,Gồm
```

### 2. Chạy chế độ không LLM

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
python run.py
```

Mở: `http://localhost:8000`

Test API:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"A dùng dao đâm B tử vong thì phạm tội gì?\",\"mode\":\"no_llm\"}"
```

---

## 🦙 Chạy với LLM Local

### Option 1: Ollama

```bash
ollama pull qwen3:4b
ollama run qwen3:4b
```

Chạy ứng dụng:
```bash
set ANSWER_MODE=local_llm
set LOCAL_LLM_MODEL=qwen3:4b
python run.py
```

Khuyến nghị model theo máy:

| Máy       | Model              |
| --------- | ------------------ |
| RAM 8GB   | qwen3:1.7b/qwen3:4b |
| RAM 16GB  | qwen3:4b/qwen3:8b  |
| GPU 8GB   | qwen3:8b           |
| GPU 12GB+ | qwen3:14b          |

### Option 2: LM Studio

```powershell
$env:ANSWER_MODE = "local_llm"
$env:LLM_PROVIDER = "lmstudio"
$env:LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
$env:LOCAL_LLM_MODEL = "auto"
python run.py
```

### Tùy chỉnh context LLM

```powershell
$env:LLM_EVIDENCE_TOP_K = "2"          # Số evidence gửi cho LLM
$env:LLM_CONTEXT_CHARS = "450"         # Max chars trong context
$env:LLM_PROMPT_CHARS = "2600"         # Max chars prompt
```

---

## 🌐 Deploy Vercel

1. **Tạo repo GitHub** và commit toàn bộ file
2. **Import vào Vercel** → Framework: `Other`
3. **Build Command**: để trống
4. **Output Directory**: để trống
5. **Environment Variables**:
```
ANSWER_MODE=no_llm
DATA_DIR=./data
TOP_K=5
```

Để deploy với `local_llm`, cần server riêng chạy Ollama:
```
ANSWER_MODE=local_llm
OLLAMA_BASE_URL=https://your-ollama-server.example.com
LOCAL_LLM_MODEL=qwen3:4b
```

---

## 📡 API Reference

### POST `/api/chat`

**Request:**
```json
{
  "message": "A dùng dao đâm B tử vong thì phạm tội gì?",
  "mode": "no_llm",
  "top_k": 5
}
```

**Parameters:**
- `mode`: `no_llm` hoặc `local_llm`
- `top_k`: Số kết quả retrieval (mặc định 5)

**Response:**
```json
{
  "reply": "...",
  "mode": "no_llm",
  "data": {
    "semantic": {},
    "evidence": {},
    "candidates": []
  }
}
```

---

## 🧪 Test & Debug

```bash
# Test pipeline
python test_pipeline.py

# Test câu hỏi điều luật
python test_article_questions.py

# Kiểm tra dữ liệu KG
python scripts/validate_data.py

# Test LM Studio
python test_lm_studio_mode.py
```

---

## 🔄 Pipeline Flow

```
Question
  ↓
Preprocess
  ↓
Scope Guard (Basic chat check)
  ↓
Rule-based Legal NER + Intent Classification
  ↓
Article Lookup / KG Search + BM25/RAG
  ↓
Rerank (keyword/entity/KG/intent)
  ↓
Evidence Builder
  ↓
Answer
  ├── no_llm: Template-based
  └── local_llm: LLM từ evidence
```

**LLM chỉ nằm ở bước cuối, không được tự bịa điều luật.**

---

## ✨ Tính năng chính

✅ **Scope Guard**: Chặn câu hỏi ngoài phạm vi pháp lý (chào hỏi, nhỏ nhặt)  
✅ **Article Lookup/Compare**: Xử lý trực tiếp câu hỏi về số Điều  
✅ **Compact Evidence**: Rút gọn context để tránh lỗi LLM  
✅ **Flexible LLM**: Hỗ trợ Ollama, LM Studio, OpenAI-compatible  
✅ **Flexible CSV**: Tự động nhận diện cột node/edge từ tên khác nhau  

---

## 📝 License

Tự do sử dụng cho mục đích phi thương mại hoặc nghiên cứu.

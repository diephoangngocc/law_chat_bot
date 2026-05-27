# LawBot - Chatbot Pháp Lý Việt Nam

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Chatbot truy xuất điều/khoản Bộ luật Hình sự Việt Nam sử dụng **Knowledge Graph** và **BM25 Retrieval**.

## Cấu Trúc Dự Án

```
lawbot/
├── backend/                    # Python backend
│   ├── api/
│   │   └── chat.py           # API endpoint cho Vercel
│   └── legal_kg/
│       ├── __init__.py       # Export main class
│       ├── pipeline.py       # Pipeline chính
│       ├── graph.py          # Quản lý Knowledge Graph
│       ├── retrieval.py      # BM25 retrieval
│       ├── text.py           # Xử lý text
│       ├── llm.py            # Tích hợp OpenAI-compatible LLM
│       └── prompts.py        # System prompts
├── frontend/                   # Giao diện web
│   └── public/
│       └── index.html        # Chatbot UI (HTML/CSS/JS)
├── data/                       # Knowledge Graph data
│   ├── nodes/                 # Node data (14 files)
│   └── edges/                 # Edge data (14 files)
└── config/
    └── vercel.json           # Vercel configuration
```

## Tính Năng

- **Truy xuất offline**: Sử dụng BM25 retrieval không cần LLM
- **Hỗ trợ LLM**: Bật chế độ LLM để phân tích sâu hơn
- **Knowledge Graph**: 14 chương Bộ luật Hình sự (XV - XXVI)
- **Giao diện đa ngôn ngữ**: Tiếng Việt với ví dụ vụ án

## Các Chương Được Hỗ Trợ

| Chương | Nội dung |
|--------|----------|
| XV-XVI | Tội xâm phạm quyền sở hữu |
| XVII | Tội xâm phạm chế độ hôn nhân, gia đình |
| XVIII | Tội xâm phạm các quyền tự do, dân chủ |
| XIX | Tội cố ý xâm phạm tính mạng, sức khỏe |
| XX | Tội cướp tài sản |
| XXI | Tội bắt giữ người trái pháp luật |
| XXII | Tội hiếp dâm, hiếp dâm trẻ em |
| XXIII | Tội xâm phạm tình mạng, sức khỏe nhân dân |
| XXIV | Tội hình sự khác |
| XXV | Tội phạm về môi trường |
| XXVI | Tội phá hoại hòa bình, chống loài người |

## Cài Đặt

### Yêu Cầu
- Python 3.8+
- pip

### Các Bước

1. **Clone repository**
```bash
git clone https://github.com/diephoangngocc/lawbot.git
cd lawbot
```

2. **Không cần cài đặt dependencies** - Dự án không yêu cầu thư viện bên thứ ba khi chạy offline

## Chạy Local

### Với Vercel CLI
```bash
npm i -g vercel
vercel dev
```

### Test API bằng Python
```bash
python -B -X utf8 -c "from backend.api.chat import get_pipeline; print(get_pipeline().run('A dùng dao đâm B nhiều nhát làm B tử vong.', top_k=1).result)"
```

### Mở giao diện web
Mở file `frontend/public/index.html` trong trình duyệt.

## Deploy Vercel

1. Push repo lên GitHub
2. Vào [Vercel](https://vercel.com) -> Add New Project
3. Import repository `lawbot`
4. Framework Preset: `Other`
5. Build Command: để trống
6. Output Directory: để trống
7. Deploy

## Cấu Hình LLM

### Local (PowerShell/CMD)
```bash
# Windows
set OPENAI_API_KEY=sk-your-api-key
set OPENAI_MODEL=gpt-4o-mini
vercel dev

# Linux/Mac
export OPENAI_API_KEY=sk-your-api-key
export OPENAI_MODEL=gpt-4o-mini
vercel dev
```

### Vercel Environment Variables
Trong Project Settings -> Environment Variables:
- `OPENAI_API_KEY` - API key từ OpenAI hoặc provider tương thích
- `OPENAI_MODEL` - Model name (mặc định: `gpt-4o-mini`)
- `OPENAI_BASE_URL` - URL endpoint khác (tùy chọn)

## API Reference

### POST /api/chat

**Request:**
```json
{
  "message": "A dùng dao đâm B nhiều nhát làm B tử vong.",
  "top_k": 5,
  "use_llm": false
}
```

**Response:**
```json
{
  "reply": "Tôi đã truy xuất KG ở chế độ offline...",
  "data": {
    "facts": {...},
    "result": {...},
    "candidates": [...]
  },
  "mode": "Offline"
}
```

### GET /api/chat
Health check endpoint.

## Knowledge Graph Format

### Node CSV
```csv
ID,Name,Label
D123,Điều 123. Tội giết người,Điều
K123a,Khoản 1,Tu khoan
```

### Edge CSV
```csv
From,To,Relationship
D123,K123a,Gồm
```

## Data

- **14 node files**: Thông tin về các điều, khoản, hành vi
- **14 edge files**: Quan hệ giữa các node trong KG

## Lưu Ý Quan Trọng

> Kết quả hiện tại chỉ là hỗ trợ truy xuất offline, **không thay thế** đánh giá pháp lý của luật sư, kiểm sát viên, thẩm phán hoặc cơ quan có thẩm quyền.

## License

MIT License

## Author

[diephoangngocc](https://github.com/diephoangngocc)

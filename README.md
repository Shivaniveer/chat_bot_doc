# RAG Document Q&A Chatbot
### Powered by Groq · LLaMA 3.3 70B · FAISS · Flask

---

## What this does
Upload any PDF → ask questions → get answers grounded in your document.

**Stack:**
- **LLM**: `llama-3.3-70b-versatile` via Groq API (fast, free tier available)
- **Embeddings**: `all-MiniLM-L6-v2` (local, no API cost)
- **Vector store**: FAISS (in-memory + persisted to disk)
- **Backend**: Flask
- **Frontend**: Vanilla HTML/CSS/JS (no build step)

---

## Quick Start

### 1. Get a free Groq API key
Go to → https://console.groq.com → sign up → create API key

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key

**Option A — environment variable (recommended):**
```bash
export GROQ_API_KEY="gsk_your_key_here"   # Mac/Linux
set GROQ_API_KEY=gsk_your_key_here         # Windows CMD
```

**Option B — edit app.py directly:**
```python
GROQ_API_KEY = "gsk_your_key_here"
```

### 4. Run
```bash
python app.py
```
Open → http://localhost:5000

---

## How it works

```
PDF Upload
   │
   ▼
Extract text (PyPDF2)
   │
   ▼
Split into chunks (LangChain RecursiveCharacterTextSplitter)
   │
   ▼
Embed chunks (sentence-transformers / MiniLM)
   │
   ▼
Store in FAISS index (persisted to disk)
   │
User asks question
   │
   ▼
Embed question → FAISS similarity search → top-4 chunks
   │
   ▼
Groq API: LLaMA 3.3 70B answers using only retrieved context
   │
   ▼
Answer displayed in chat UI
```

---

## Project Structure
```
rag_chatbot/
├── app.py               # Flask backend + RAG logic
├── requirements.txt     # Dependencies
├── README.md
├── templates/
│   └── index.html       # Chat UI
├── uploads/             # Auto-created — stores uploaded PDFs
└── vector_store/        # Auto-created — persisted FAISS index
    ├── faiss.index
    └── chunks.pkl
```

---

## Notes
- The FAISS index is rebuilt every time you upload a new PDF.
- Index persists across server restarts (stored in `vector_store/`).
- Supports one document at a time; extend `build_index()` to merge multiple.

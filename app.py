from flask import Flask, request, jsonify, render_template
from groq import Groq
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os
import pickle

from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
def get_embedding(text):
    response = client.embeddings.create(
        model="nomic-embed-text-v1_5",  
        input=text
    )
    return response.data[0].embedding
# ─── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
MODEL_NAME   = "llama-3.3-70b-versatile"   # LLaMA 3.3 70B on Groq
EMBED_MODEL  = "all-MiniLM-L6-v2"          # local sentence-transformer for embeddings
UPLOAD_FOLDER = "uploads"
INDEX_FILE    = "vector_store/faiss.index"
CHUNKS_FILE   = "vector_store/chunks.pkl"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("vector_store", exist_ok=True)

# ─── Globals ──────────────────────────────────────────────────────────────────
client        = Groq(api_key=GROQ_API_KEY)
embedder      = SentenceTransformer(EMBED_MODEL)
faiss_index   = None
stored_chunks = []
# !@#$King@1234567
# ─── Helpers ──────────────────────────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def build_index(text: str):
    global faiss_index, stored_chunks

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    stored_chunks = splitter.split_text(text)

    embeddings = embedder.encode(stored_chunks, show_progress_bar=False)
    dim = embeddings.shape[1]

    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embeddings, dtype="float32"))

    faiss.write_index(faiss_index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(stored_chunks, f)

    return len(stored_chunks)


def load_index():
    global faiss_index, stored_chunks
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        faiss_index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            stored_chunks = pickle.load(f)
        return True
    return False


def retrieve_context(query: str, k: int = 4) -> str:
    if faiss_index is None:
        return ""
    q_emb = embedder.encode([query], show_progress_bar=False)
    _, indices = faiss_index.search(np.array(q_emb, dtype="float32"), k)
    return "\n\n".join(stored_chunks[i] for i in indices[0] if i < len(stored_chunks))


def ask_llm(query: str, context: str) -> str:
    system_prompt = (
        "You are a helpful assistant that answers questions ONLY from the provided document context.\n"
        "If the answer is not in the context, say: 'Please ask  question regarding uploaded documents.'\n"
        "Be concise and accurate."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    text = extract_text_from_pdf(save_path)
    if not text.strip():
        return jsonify({"error": "Could not extract text from PDF"}), 400

    num_chunks = build_index(text)
    return jsonify({"message": f"PDF indexed successfully! {num_chunks} chunks created.", "chunks": num_chunks})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    # Try loading index if not already in memory
    if faiss_index is None:
        if not load_index():
            return jsonify({"error": "No document indexed yet. Please upload a PDF first."}), 400

    context = retrieve_context(question)
    answer  = ask_llm(question, context)

    return jsonify({"answer": answer, "context_used": context[:300] + "..." if len(context) > 300 else context})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

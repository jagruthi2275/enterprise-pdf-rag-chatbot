# 📄 Enterprise PDF RAG Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white"/>
  <img src="https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-00B4D8?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
</p>

<p align="center">
  A production-grade Retrieval Augmented Generation (RAG) chatbot that answers questions
  exclusively from your uploaded PDF documents — with conversation memory, query rewriting,
  similarity scoring, and a premium enterprise UI.
</p>

---

## 🧠 Project Overview

Most LLM chatbots hallucinate facts from their training data. This project solves that by
combining **document retrieval** with **LLM generation** — the model can only answer from
what's actually in your PDFs, with citations to the exact page.

Built as a portfolio-grade AI Engineer project, it implements several production RAG patterns
rarely seen in student projects:

- **Pre-retrieval query rewriting** — ambiguous follow-up questions are reformulated before hitting the vector store
- **Stateful conversation memory** — last 3 exchanges are maintained for contextual follow-ups
- **FAISS similarity scoring** — each retrieved chunk surfaces a normalised relevance score
- **Step-by-step pipeline feedback** — real-time processing status with per-stage checkmarks
- **Runtime statistics** — response time, chunk count, context length shown per answer

---

## ✨ Features

| Feature | Details |
|---|---|
| 📥 Multi-PDF Upload | Upload and index multiple PDFs simultaneously |
| ⚙️ Live Ingestion Pipeline | Step-by-step progress: Load → Extract → Chunk → Embed → Index |
| 💬 Conversational Memory | Retains last 3 user-assistant exchanges for follow-up questions |
| ✍️ Query Rewriting | Pre-retrieval LLM step resolves ambiguous queries using history |
| 🔍 Semantic Search | FAISS IndexFlatL2 with cosine-equivalent similarity scoring |
| 📌 Source Citations | Every answer cites PDF filename, page number, chunk ID, and score |
| 📊 Metrics Dashboard | Live stats: PDFs, pages, chunks, model, retrieval strategy |
| ⏱ Runtime Stats | Response time, context length, and model shown per answer |
| 🎨 Premium Dark UI | Enterprise-grade design with Inter font, gradient accents, cards |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                   │
│                                                          │
│  PDF Files → PDFLoader → DocumentChunker → EmbeddingModel │
│                                 ↓                        │
│                          VectorStore (FAISS)             │
│                          saved to disk                   │
└─────────────────────────────────────────────────────────┘
                              ↕ (persisted index)
┌─────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                      │
│                                                          │
│  User Question                                           │
│       ↓                                                  │
│  QueryRewriter (Groq) ← Conversation Memory (last 3)    │
│       ↓ rewritten query                                  │
│  EmbeddingModel → query vector                           │
│       ↓                                                  │
│  FAISS search → Top-3 chunks + similarity scores         │
│       ↓                                                  │
│  PromptBuilder (context + question + rules)              │
│       ↓                                                  │
│  Groq API → Llama 3.3 70B → Answer                      │
│       ↓                                                  │
│  Streamlit UI (answer + sources + runtime stats)         │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
enterprise-pdf-rag-chatbot/
│
├── app.py                   # Streamlit UI — chat interface, metrics, source cards
├── .env                     # API keys (never committed)
├── .gitignore
├── requirements.txt
│
└── src/
    ├── pdf_loader.py        # PyMuPDF/pdfplumber — extracts text per page
    ├── chunking.py          # RecursiveCharacterTextSplitter (chunk=500, overlap=50)
    ├── embeddings.py        # HuggingFace sentence-transformers
    ├── vector_store.py      # FAISS IndexFlatL2 — build, save, load, search
    ├── retrieval.py         # Top-K retrieval with L2→similarity score conversion
    ├── query_rewriter.py    # Pre-retrieval query rewriting via Groq LLM
    ├── llm.py               # ChatGroq wrapper — Llama 3.3 70B
    ├── prompts.py           # Prompt template with system rules + context formatting
    ├── rag_chain.py         # Orchestrator — ingest() and ask() methods
    └── utils.py             # Logging setup, directory creation
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | Streamlit | Chat interface, file upload, metrics dashboard |
| **LLM** | Groq API — Llama 3.3 70B | Answer generation, query rewriting |
| **Embeddings** | HuggingFace sentence-transformers | Text → vector conversion |
| **Vector DB** | FAISS (IndexFlatL2) | Semantic similarity search |
| **RAG Framework** | LangChain | ChatGroq client, text splitting utilities |
| **PDF Parsing** | PyMuPDF / pdfplumber | Page-level text extraction |
| **Language** | Python 3.10+ | Core runtime |

---

## 🚀 Installation

```bash
# 1. Clone the repo
git clone https://github.com/jagruthi2275/enterprise-pdf-rag-chatbot.git
cd enterprise-pdf-rag-chatbot

# 2. Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys (see below)

# 5. Run the app
streamlit run app.py
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
HF_TOKEN=hf_your_huggingface_token_here
```

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free tier available |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## 🔬 How RAG Works

Traditional LLMs answer from their training data — they hallucinate, can't cite sources,
and can't know your private documents.

RAG fixes this with two stages:

**Stage 1 — Ingestion (run once per document set)**
```
PDF → extract text per page → split into 500-token chunks →
convert each chunk to a vector (HuggingFace embeddings) →
store all vectors in FAISS index
```

**Stage 2 — Query (run per user question)**
```
User question → rewrite with conversation context (QueryRewriter) →
convert to vector → FAISS finds top-3 most similar chunks →
build prompt: [system rules] + [3 chunks] + [question] →
Llama 3.3 70B generates answer grounded in those chunks only
```

The LLM never sees the full PDF — only the 3 most relevant chunks.
This keeps answers accurate, fast, and fully citable.

---

## 📸 Screenshots

> _Add screenshots after deployment_

| Feature | Screenshot |
|---|---|
| Document Upload & Processing | `screenshots/upload.png` |
| Metrics Dashboard | `screenshots/metrics.png` |
| Chat with Source Citations | `screenshots/chat.png` |
| Query Rewrite Badge | `screenshots/rewrite.png` |
| Runtime Statistics | `screenshots/stats.png` |

---

## 🔮 Future Improvements

- [ ] **Hybrid Search** — combine FAISS semantic search with BM25 keyword search
- [ ] **Cross-encoder Reranking** — rerank retrieved chunks with a fine-tuned reranker model
- [ ] **Multi-document Comparison** — query across multiple PDFs and compare answers
- [ ] **Persistent Sessions** — save and reload conversation history across browser sessions
- [ ] **Streaming Responses** — token-by-token answer streaming like ChatGPT
- [ ] **Document Analytics** — keyword frequency, topic clusters, reading time estimation
- [ ] **Authentication** — user login with per-user document isolation

---

## 🏆 Resume-Worthy Highlights

This project demonstrates the following skills relevant to **AI Engineer / ML Engineer / GenAI roles**:

- ✅ Built an end-to-end RAG pipeline from scratch without relying on high-level LangChain wrappers
- ✅ Implemented **pre-retrieval query rewriting** — a production pattern used at Perplexity, Notion AI, Glean
- ✅ **Stateful conversation memory** using session-scoped state (no database dependency)
- ✅ **FAISS vector database** — built, persisted, and queried an IndexFlatL2 index
- ✅ **Similarity scoring** — L2 distance normalised to 0–1 relevance scores surfaced in UI
- ✅ Custom **prompt engineering** with strict grounding rules and citation enforcement
- ✅ **Modular, production-quality code** — each component is independently testable
- ✅ Deployed as a live Streamlit web app with enterprise-grade UI

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

**Jagruthi Reddy Pennapu**
B.Tech CSE (AI & ML) — GITAM University, Hyderabad

- 🐙 GitHub: [github.com/jagruthi2275](https://github.com/jagruthi2275)
- 💼 LinkedIn: [linkedin.com/in/jagruthi-reddy-pennapu-9a4a05358](https://linkedin.com/in/jagruthi-reddy-pennapu-9a4a05358)

---

<p align="center">
  Built with 🤍 as an AI Engineer portfolio project · Not affiliated with Groq, Meta, or HuggingFace
</p>
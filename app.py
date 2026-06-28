"""Enterprise PDF RAG Chatbot — Premium Streamlit UI."""

import os
import time
import tempfile
import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── API key guards ─────────────────────────────────────────────────────────────
if not os.environ.get("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY is not set. Add it to your .env file.")
    st.stop()

if not os.environ.get("HF_TOKEN"):
    st.error("❌ HF_TOKEN is not set. Add it to your .env file.")
    st.stop()

from src.utils import setup_logging, ensure_dirs
setup_logging()
ensure_dirs()

from src.rag_chain import RAGChain
from src.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium dark-theme CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0d0f14;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2230 !important;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.85rem;
    transition: all 0.2s ease;
}

/* ── Primary button ── */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.25rem !important;
    font-weight: 500 !important;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #2d3348 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    background: #14161f !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #14161f !important;
    border: 1px solid #1e2230 !important;
    border-radius: 14px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] > div {
    background: #14161f !important;
    border: 1.5px solid #2d3348 !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #14161f !important;
    border: 1px solid #1e2230 !important;
    border-radius: 12px !important;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(145deg, #14161f, #1a1d28);
    border: 1px solid #1e2230;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #818cf8;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

/* ── Source cards ── */
.source-card {
    background: #14161f;
    border: 1px solid #1e2230;
    border-left: 3px solid #6366f1;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
}
.source-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #a5b4fc;
    margin-bottom: 0.3rem;
}
.source-meta {
    font-size: 0.75rem;
    color: #64748b;
}
.score-badge {
    display: inline-block;
    background: #1e2230;
    color: #818cf8;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.1rem 0.5rem;
    border-radius: 20px;
    margin-left: auto;
}

/* ── Pipeline steps ── */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.83rem;
    color: #94a3b8;
    padding: 0.3rem 0;
}
.pipeline-step.done { color: #4ade80; }
.step-icon { font-size: 0.9rem; }

/* ── Runtime stats bar ── */
.stats-bar {
    display: flex;
    gap: 1.2rem;
    flex-wrap: wrap;
    font-size: 0.73rem;
    color: #475569;
    margin-top: 0.75rem;
    padding-top: 0.6rem;
    border-top: 1px solid #1e2230;
}
.stat-item { display: flex; align-items: center; gap: 0.3rem; }
.stat-item span { color: #818cf8; font-weight: 600; }

/* ── Section headers ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin-bottom: 0.5rem;
}

/* ── Query rewrite badge ── */
.rewrite-badge {
    font-size: 0.68rem;
    color: #3d4460;
    background: transparent;
    border: none;
    padding: 0.1rem 0;
    margin-bottom: 0.3rem;
    display: block;
    letter-spacing: 0.01em;
}
.rewrite-badge em { color: #4a5280; font-style: italic; }

/* ── Info / success boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

/* ── Divider ── */
hr { border-color: #1e2230 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
defaults = {
    "rag":          None,
    "rewriter":     None,
    "messages":     [],          # {"role", "content", "sources", "stats", "rewritten_query"}
    "docs_ready":   False,
    "doc_meta":     {},          # {n_files, total_pages, total_chunks}
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Lazy-init heavy objects once
if st.session_state.rag is None:
    st.session_state.rag = RAGChain()
if st.session_state.rewriter is None:
    try:
        st.session_state.rewriter = QueryRewriter()
    except Exception as e:
        logger.warning(f"QueryRewriter init failed: {e}")
        st.session_state.rewriter = None

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_last_n_exchanges(n: int = 3) -> list:
    """Return the last n user+assistant exchange pairs as a flat list."""
    pairs = []
    for msg in st.session_state.messages:
        if msg["role"] in ("user", "assistant"):
            pairs.append({"role": msg["role"], "content": msg["content"]})
    return pairs[-(n * 2):]   # last n exchanges = 2n messages


def render_source_cards(sources: list):
    """Render retrieved chunks as enterprise-style source cards."""
    for i, s in enumerate(sources):
        score    = s.get("similarity_score")
        page     = s.get("page", "—")
        chunk_id = s.get("chunk_id", i)
        preview  = s.get("text", "")[:300] + ("…" if len(s.get("text", "")) > 300 else "")
        score_str = f"{score:.3f}" if score is not None else "—"

        st.markdown(f"""
        <div class="source-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <span style="font-size:0.8rem; font-weight:700; color:#a5b4fc;">Chunk {i+1}</span>
                <span class="score-badge">similarity {score_str}</span>
            </div>
            <div style="font-size:0.78rem; color:#64748b; line-height:1.8;">
                📄 <b style="color:#94a3b8;">File:</b> {s['source']}<br>
                📖 <b style="color:#94a3b8;">Page:</b> {page}<br>
                🔢 <b style="color:#94a3b8;">Chunk ID:</b> #{chunk_id}
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Preview retrieved text"):
            st.caption(preview)
        if i < len(sources) - 1:
            st.markdown('<hr style="border-color:#1e2230; margin:0.3rem 0;">', unsafe_allow_html=True)


def render_runtime_stats(stats: dict):
    """Render a compact stats bar below each answer."""
    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat-item">⏱ Response time <span>{stats.get('response_time', '—')}s</span></div>
        <div class="stat-item">🔍 Chunks retrieved <span>{stats.get('chunks_retrieved', '—')}</span></div>
        <div class="stat-item">🤖 Model <span>{stats.get('model', 'Llama 3.3 70B')}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Enterprise RAG")
    st.markdown('<div class="section-label">Document Upload</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop PDF(s) here", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
    )

    st.markdown("")

    if st.button("⚙️ Process Documents", disabled=not uploaded_files, use_container_width=True):
        pipeline_placeholder = st.empty()

        steps = [
            ("PDF Loaded",           False),
            ("Text Extracted",       False),
            ("Text Chunked",         False),
            ("Embeddings Generated", False),
            ("FAISS Index Built",    False),
            ("Ready for Questions",  False),
        ]

        def render_pipeline(done_up_to: int):
            html = '<div style="margin: 0.75rem 0;">'
            for i, (label, _) in enumerate(steps):
                done = i < done_up_to
                cls  = "pipeline-step done" if done else "pipeline-step"
                icon = "✓" if done else "○"
                html += f'<div class="{cls}"><span class="step-icon">{icon}</span>{label}</div>'
            html += "</div>"
            pipeline_placeholder.markdown(html, unsafe_allow_html=True)

        try:
            render_pipeline(0)
            tmp_paths = []
            original_names = []
            for f in uploaded_files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(f.read())
                tmp.close()
                tmp_paths.append(tmp.name)
                original_names.append(f.name)
            render_pipeline(1)   # PDF Loaded

            # Patch ingest to expose step-by-step progress
            docs   = st.session_state.rag.loader.load(tmp_paths, original_names)
            render_pipeline(2)   # Text Extracted

            chunks = st.session_state.rag.chunker.chunk(docs)
            render_pipeline(3)   # Text Chunked

            texts      = [c["text"] for c in chunks]
            embeddings = st.session_state.rag.embedder.embed(texts)
            render_pipeline(4)   # Embeddings Generated

            st.session_state.rag.vector_store.build(chunks, embeddings)
            st.session_state.rag.vector_store.save()
            st.session_state.rag._ready = True
            render_pipeline(5)   # FAISS Index Built

            # Count total pages
            total_pages = max((c.get("page", 0) for c in chunks), default=0)

            st.session_state.doc_meta = {
                "n_files":      len(uploaded_files),
                "total_pages":  total_pages,
                "total_chunks": len(chunks),
            }
            st.session_state.docs_ready = True
            render_pipeline(6)   # Ready for Questions

        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            st.error(f"❌ Failed to process documents: {e}")

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # ── Model info panel ──
    st.markdown('<div class="section-label" style="margin-top:1rem;">System Info</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#475569; line-height:1.9;">
        🤖 <b style="color:#94a3b8;">LLM</b> &nbsp; Llama 3.3 70B<br>
        🔎 <b style="color:#94a3b8;">Embeddings</b> &nbsp; HuggingFace<br>
        🗄️ <b style="color:#94a3b8;">Vector DB</b> &nbsp; FAISS<br>
        🧠 <b style="color:#94a3b8;">Memory</b> &nbsp; Last 3 exchanges<br>
        ✍️ <b style="color:#94a3b8;">Query Rewriting</b> &nbsp; Enabled<br>
        📡 <b style="color:#94a3b8;">Retrieval</b> &nbsp; Top-5 chunks
    </div>
    """, unsafe_allow_html=True)

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("## 📄 Enterprise PDF RAG Chatbot")
st.caption("Answers grounded exclusively in your uploaded documents · Powered by Llama 3.3 70B & FAISS")

# ── Metrics dashboard ──────────────────────────────────────────────────────────
if st.session_state.docs_ready:
    meta = st.session_state.doc_meta
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        (c1, str(meta.get("n_files", 0)),        "PDFs Uploaded"),
        (c2, str(meta.get("total_pages", 0)),     "Total Pages"),
        (c3, str(meta.get("total_chunks", 0)),    "Chunks Indexed"),
        (c4, "HuggingFace",                       "Embedding Model"),
        (c5, "FAISS",                             "Vector Store"),
        (c6, "Top-5",                             "Retrieval Strategy"),
    ]
    for col, val, label in cards:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("")

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):

        # Show query rewrite notice for user messages
        if msg["role"] == "user" and msg.get("rewritten_query") and \
                msg["rewritten_query"] != msg["content"]:
            st.markdown(
                f'<div class="rewrite-badge">🧠 Optimized Query: '
                f'<em>{msg["rewritten_query"]}</em></div>',
                unsafe_allow_html=True
            )

        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("sources"):
                with st.expander(f"📌 Sources ({len(msg['sources'])} chunks retrieved)"):
                    render_source_cards(msg["sources"])
            if msg.get("stats"):
                render_runtime_stats(msg["stats"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask something about your documents…"):
    if not st.session_state.docs_ready:
        st.warning("⚠️ Please upload and process PDFs first.")
    else:
        # ── Query rewriting ──
        history         = get_last_n_exchanges(n=3)
        rewritten_query = prompt
        if st.session_state.rewriter and history:
            with st.spinner("Optimising query…"):
                rewritten_query = st.session_state.rewriter.rewrite(prompt, history)

        # Store user message (with rewritten query for display)
        st.session_state.messages.append({
            "role":            "user",
            "content":         prompt,
            "rewritten_query": rewritten_query,
        })
        with st.chat_message("user"):
            if rewritten_query != prompt:
                st.markdown(
                    f'<div class="rewrite-badge">🧠 Optimized Query: '
                    f'<em>{rewritten_query}</em></div>',
                    unsafe_allow_html=True
                )
            st.markdown(prompt)

        # ── RAG answer ──
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    t0     = time.time()
                    result = st.session_state.rag.ask(rewritten_query)
                    elapsed = round(time.time() - t0, 2)

                    answer  = result["answer"]
                    sources = result["sources"]

                    # Build context length from source texts
                    context_len = sum(len(s.get("text", "")) for s in sources)

                    stats = {
                        "response_time":   elapsed,
                        "chunks_retrieved": len(sources),
                        "context_length":  context_len,
                        "model":           "Llama 3.3 70B",
                    }

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📌 Sources ({len(sources)} chunks retrieved)"):
                            render_source_cards(sources)

                    render_runtime_stats(stats)

                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": sources,
                        "stats":   stats,
                    })

                except Exception as e:
                    logger.error(f"RAG error: {e}")
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
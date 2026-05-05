# Semantic Search System

A local RAG pipeline I built that lets you index your own documents and ask questions about them. It combines keyword search (BM25) with semantic vector search, fuses the results, re-ranks them with a cross-encoder, and sends the best chunks to Groq's LLM to generate an actual answer.

No cloud storage, no subscriptions beyond the free Groq API key.

## How it works

Documents get split into chunks, embedded locally, and stored in ChromaDB. When you search or ask a question, your query gets embedded the same way, the closest chunks get retrieved, and if you're using `ask`, those chunks get passed to Groq's LLM to generate an answer.

The stack: BM25 (`rank-bm25`) for keyword search, `all-MiniLM-L6-v2` for embeddings (runs locally via sentence-transformers), ChromaDB as the vector store, a `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranker (also local), and `llama-3.3-70b-versatile` on Groq for the actual answers. The CLI is Typer + Rich, the API is FastAPI, and the frontend is plain HTML/CSS/JS.

---

## Setup

**Install dependencies**
```bash
cd backend
pip3 install -r requirements.txt
```

**Add your Groq API key to `backend/.env`**
```bash
GROQ_API_KEY=your_key_here
```

Free key at [console.groq.com](https://console.groq.com) — takes about a minute.

---

## CLI

Run all CLI commands from the `backend/` folder:

```bash
cd backend
```

### Indexing documents
```bash
# Single file (.txt, .pdf, .docx)
python3 main.py index data/sample_docs/ai_overview.txt

# Whole directory
python3 main.py index data/sample_docs/

# Pipe text in directly
echo "Some text to index" | python3 main.py index - --id my-doc
```

### Search without the LLM (no API key needed)
```bash
python3 main.py search "how does retrieval augmented generation work?"
python3 main.py search "Python testing best practices" --top-k 3
```

### Ask a question (Groq generates the answer)
```bash
python3 main.py ask "What is the difference between deep learning and machine learning?"
python3 main.py ask "How do I write tests in Python?" --top-k 3
python3 main.py ask "Explain vector databases" --rerank   # enables cross-encoder reranking
```

### Index management
```bash
python3 main.py stats               # chunk count + document list
python3 main.py delete ai_overview  # remove a document by ID
```

---

## Web UI

There's a browser frontend in `frontend/` if you prefer clicking over typing.

Start the backend:
```bash
cd backend
python3 api.py
# → http://localhost:8000        (frontend)
# → http://localhost:8000/docs   (Swagger UI)
```

The frontend is served directly by the backend — just open `http://localhost:8000` in your browser once the server is running. No separate step needed.

The UI has four tabs: **Search** for hybrid search with match % badges and source attribution, **Ask** for getting an LLM-generated answer with cited sources, **Index** for dragging in a file or pasting text, and **Stats** for seeing what's indexed and deleting documents.

There's a **Developer Mode** toggle in the sidebar that shows per-query latency (Embed / BM25 / Vector / Fusion / Rerank / Total) and raw scores on each result card — handy when you're tuning retrieval. There's also a live API status dot so you know if the backend is actually running.

---

## REST API

| Method | Path | |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Index stats |
| `POST` | `/index/text` | Index a string |
| `POST` | `/index/file` | Upload and index a file |
| `POST` | `/search` | Semantic search |
| `POST` | `/ask` | RAG question answering |
| `DELETE` | `/documents/{doc_id}` | Delete a document |

```bash
# Index some text
curl -X POST http://localhost:8000/index/text \
  -H "Content-Type: application/json" \
  -d '{"text": "FastAPI is a modern Python web framework.", "doc_id": "fastapi-intro"}'

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Python web frameworks", "top_k": 3}'

# Ask
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FastAPI used for?", "top_k": 5}'

# Upload a file
curl -X POST http://localhost:8000/index/file \
  -F "file=@/path/to/document.pdf"
```

---

## Project layout

```
Semantic Search System/
├── backend/
│   ├── src/
│   │   ├── document_loader.py   # loads .txt / .pdf / .docx, chunks with overlap
│   │   ├── embeddings.py        # sentence-transformers wrapper
│   │   ├── vector_store.py      # ChromaDB reads/writes
│   │   ├── search_engine.py     # orchestrates indexing and search
│   │   ├── llm_client.py        # Groq API calls (answers + reranking)
│   │   └── rag_pipeline.py      # ties it all together
│   ├── data/
│   │   └── sample_docs/         # a few sample docs to try it out
│   ├── tests/
│   │   └── test_pipeline.py
│   ├── config.py                # reads settings from .env
│   ├── main.py                  # CLI
│   ├── api.py                   # FastAPI server
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── .gitignore
└── README.md
```

---

## How retrieval works

Every query hits both BM25 and vector search at the same time. BM25 is good at exact terms and jargon; vector search is good at meaning and paraphrases. The results from both get merged with Reciprocal Rank Fusion (`score = Σ 1 / (k + rank)`) — chunks that show up high in both lists naturally score highest. Near-duplicates (Jaccard overlap > `DEDUP_THRESHOLD`) get dropped so you're not getting the same sentence repeated.

After that, a cross-encoder scores each `(query, chunk)` pair jointly, which is slower than dot-product similarity but meaningfully more accurate. The match percentages you see in the UI are just a sigmoid over the reranker's raw logit scores.

Knobs you can tune in `.env`:

| Setting | Default | |
|---|---|---|
| `MMR_FETCH_K` | `20` | Candidate pool for both retrievers |
| `RERANKER_TOP_K` | `20` | How many go to the cross-encoder |
| `RRF_K` | `60` | RRF smoothing constant |
| `DEDUP_THRESHOLD` | `0.92` | Jaccard threshold for duplicate removal |
| `CHUNK_OVERLAP` | `32` | Overlap between adjacent chunks |

---

## Temperature

For answer generation (`ask`) I use `0.2` — grounded in the retrieved context but not completely robotic. For LLM reranking (`--rerank`) it's `0.0` since that needs to be deterministic. Both are in [backend/src/llm_client.py](backend/src/llm_client.py).

---

## Configuration

All settings go in `backend/.env`:

| Variable | Default | |
|---|---|---|
| `GROQ_API_KEY` | — | Required for `ask` |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | |
| `CHROMA_DB_PATH` | `./chroma_db` | Where the vector DB lives |
| `TOP_K_RESULTS` | `5` | Default result count |
| `CHUNK_SIZE` | `512` | Max chars per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |

---

## Tests

```bash
cd backend
pytest tests/ -v
```

LLM tests need `GROQ_API_KEY`. Embedding and vector store tests run offline.

---

## Supported file types

`.txt`, `.pdf` (via PyPDF2), `.docx` (via python-docx)

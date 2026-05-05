from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tempfile
import shutil
from pathlib import Path

app = FastAPI(
    title="Semantic Search System",
    description="Hybrid BM25 + Vector search with cross-encoder reranking, powered by Groq",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = None
_pipeline = None
_llm = None


def get_engine():
    global _engine
    if _engine is None:
        from src.search_engine import SearchEngine
        _engine = SearchEngine()
    return _engine


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.rag_pipeline import RAGPipeline
        _pipeline = RAGPipeline()
    return _pipeline


def get_llm():
    global _llm
    if _llm is None:
        from src.llm_client import LLMClient
        _llm = LLMClient()
    return _llm


# ── Request models ───────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    system_prompt: str | None = None


class ChatResponse(BaseModel):
    message: str
    reply: str
    model: str


class IndexTextRequest(BaseModel):
    text: str
    doc_id: str = "manual"
    metadata: dict = {}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    system_prompt: str | None = None


# ── Response models ──────────────────────────────────────────

class LatencyOut(BaseModel):
    embedding_ms: float
    bm25_ms: float
    vector_ms: float
    fusion_ms: float
    reranking_ms: float
    total_ms: float


class SearchResultOut(BaseModel):
    chunk_id: str
    content: str
    metadata: dict
    score: float
    match_pct: float
    rerank_score: float | None
    retrieval_method: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultOut]
    latency: LatencyOut


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchResultOut]
    model: str
    latency: LatencyOut


class StatsResponse(BaseModel):
    total_chunks: int
    documents: list[str]


# ── Helpers ──────────────────────────────────────────────────

def _result_out(r) -> SearchResultOut:
    return SearchResultOut(
        chunk_id=r.chunk_id,
        content=r.content,
        metadata=r.metadata,
        score=r.score,
        match_pct=r.match_pct,
        rerank_score=r.rerank_score,
        retrieval_method=r.retrieval_method,
    )


def _latency_out(lat) -> LatencyOut:
    return LatencyOut(
        embedding_ms=lat.embedding_ms,
        bm25_ms=lat.bm25_ms,
        vector_ms=lat.vector_ms,
        fusion_ms=lat.fusion_ms,
        reranking_ms=lat.reranking_ms,
        total_ms=lat.total_ms,
    )


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    llm = get_llm()
    reply = llm.chat(req.message, req.system_prompt)
    return ChatResponse(message=req.message, reply=reply, model=llm.model)


@app.get("/stats", response_model=StatsResponse)
def stats():
    s = get_engine().stats()
    return StatsResponse(total_chunks=s.total_chunks, documents=s.documents)


@app.post("/index/text")
def index_text(req: IndexTextRequest):
    added = get_engine().index_text(req.text, doc_id=req.doc_id, metadata=req.metadata)
    return {"doc_id": req.doc_id, "chunks_added": added}


@app.post("/index/file")
def index_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        added = get_engine().index_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"filename": file.filename, "chunks_added": added}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    results, latency = get_engine().search(req.query, top_k=req.top_k)
    return SearchResponse(
        query=req.query,
        results=[_result_out(r) for r in results],
        latency=_latency_out(latency),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    pipeline = get_pipeline()
    response = pipeline.ask(req.query, top_k=req.top_k, system_prompt=req.system_prompt)
    return AskResponse(
        query=response.query,
        answer=response.answer,
        sources=[_result_out(r) for r in response.sources],
        model=response.model,
        latency=_latency_out(response.latency),
    )


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    get_engine().delete_document(doc_id)
    return {"deleted": doc_id}


frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

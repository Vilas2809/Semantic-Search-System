from dataclasses import dataclass, field
from time import perf_counter

from src.document_loader import Document, DocumentLoader, TextChunker
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore, SearchResult
from src.bm25_store import BM25Store
from src.reranker import Reranker
from config import config


@dataclass
class LatencyBreakdown:
    embedding_ms: float = 0.0
    bm25_ms: float = 0.0
    vector_ms: float = 0.0
    fusion_ms: float = 0.0
    reranking_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class IndexStats:
    total_chunks: int
    documents: list[str]


def _ms(t: float) -> float:
    return round((perf_counter() - t) * 1000, 2)


class SearchEngine:
    """
    Two-stage hybrid search engine:
      Stage 1 — BM25 + Vector → Reciprocal Rank Fusion → deduplication
      Stage 2 — Cross-encoder re-ranker narrows to final top-k
    """

    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        self.embedder = EmbeddingModel(model_name=config.EMBEDDING_MODEL)
        self.store = VectorStore(
            db_path=config.CHROMA_DB_PATH,
            collection_name=config.COLLECTION_NAME,
            embedding_model=self.embedder,
        )
        self.bm25 = BM25Store(db_path=config.CHROMA_DB_PATH)
        self.reranker = Reranker(model_name=config.RERANKER_MODEL)

    # ── Indexing ─────────────────────────────────────────────

    def index_document(self, doc: Document) -> int:
        chunks = self.chunker.chunk_document(doc)
        added = self.store.add_chunks(chunks)
        self.bm25.add_chunks(chunks)
        return added

    def index_file(self, path: str) -> int:
        from pathlib import Path
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            doc = self.loader.load_pdf(path)
        elif ext == ".docx":
            doc = self.loader.load_docx(path)
        else:
            doc = self.loader.load_text_file(path)
        return self.index_document(doc)

    def index_text(self, text: str, doc_id: str = "manual", metadata: dict = None) -> int:
        doc = self.loader.load_from_string(text, doc_id=doc_id, metadata=metadata)
        return self.index_document(doc)

    def index_directory(self, directory: str) -> dict[str, int]:
        docs = self.loader.load_directory(directory)
        return {doc.doc_id: self.index_document(doc) for doc in docs}

    # ── Search ───────────────────────────────────────────────

    def search(self, query: str, top_k: int = None) -> tuple[list[SearchResult], LatencyBreakdown]:
        """
        Full hybrid retrieval pipeline with latency breakdown.
        Returns (results, latency).
        """
        k = top_k or config.TOP_K_RESULTS
        fetch_k = config.MMR_FETCH_K
        latency = LatencyBreakdown()
        t_total = perf_counter()

        # Step 1: embed query once, shared by both retrievers
        t = perf_counter()
        query_vec = self.embedder.embed(query)
        latency.embedding_ms = _ms(t)

        # Step 2a: BM25 keyword search
        t = perf_counter()
        bm25_hits = self.bm25.search(query, top_k=fetch_k)
        latency.bm25_ms = _ms(t)

        # Step 2b: Vector search using the pre-computed embedding
        t = perf_counter()
        vec_results = self.store.search_with_embedding(query_vec, top_k=fetch_k)
        latency.vector_ms = _ms(t)

        # Step 3: RRF fusion + text deduplication
        t = perf_counter()
        fused = self._rrf_fuse(bm25_hits, vec_results, fetch_k)
        fused = self._deduplicate(fused, threshold=config.DEDUP_THRESHOLD)
        latency.fusion_ms = _ms(t)

        # Step 4: Cross-encoder re-rank
        t = perf_counter()
        candidates = fused[: config.RERANKER_TOP_K]
        final = self.reranker.rerank(query, candidates, top_k=k)
        latency.reranking_ms = _ms(t)

        latency.total_ms = _ms(t_total)
        return final, latency

    def delete_document(self, doc_id: str):
        self.store.delete_document(doc_id)
        self.bm25.delete_document(doc_id)

    def stats(self) -> IndexStats:
        return IndexStats(
            total_chunks=self.store.count(),
            documents=self.store.list_documents(),
        )

    # ── Hybrid internals ────────────────────────────────────

    def _rrf_fuse(
        self,
        bm25_hits: list[tuple],
        vec_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion: score = Σ 1/(k + rank) across both rankings."""
        k = config.RRF_K
        rrf_scores: dict[str, float] = {}
        content_map: dict[str, str] = {}
        meta_map: dict[str, dict] = {}

        for rank, (chunk_id, content, meta, _) in enumerate(bm25_hits):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1 / (k + rank + 1)
            content_map[chunk_id] = content
            meta_map[chunk_id] = meta

        for rank, r in enumerate(vec_results):
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + 1 / (k + rank + 1)
            content_map[r.chunk_id] = r.content
            meta_map[r.chunk_id] = r.metadata

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        return [
            SearchResult(
                chunk_id=cid,
                content=content_map[cid],
                metadata=meta_map[cid],
                score=round(rrf_scores[cid], 6),
                retrieval_method="hybrid",
            )
            for cid in sorted_ids
        ]

    def _deduplicate(self, results: list[SearchResult], threshold: float) -> list[SearchResult]:
        """Remove results whose word overlap with an earlier result exceeds the threshold."""
        kept: list[SearchResult] = []
        kept_sets: list[set] = []
        for r in results:
            words = set(r.content.lower().split())
            is_dup = any(
                len(words & s) / len(words | s) > threshold
                for s in kept_sets
                if words | s
            )
            if not is_dup:
                kept.append(r)
                kept_sets.append(words)
        return kept

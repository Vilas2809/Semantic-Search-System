from dataclasses import dataclass
import numpy as np
import chromadb
from chromadb.config import Settings

from src.document_loader import Chunk
from src.embeddings import EmbeddingModel


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    metadata: dict
    score: float             # cosine similarity (vector) or RRF score (hybrid)
    match_pct: float = 0.0   # 0-100, set by reranker
    rerank_score: float | None = None
    retrieval_method: str = "hybrid"


class VectorStore:
    """ChromaDB-backed persistent vector store."""

    def __init__(self, db_path: str, collection_name: str, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and store chunks. Returns number added."""
        if not chunks:
            return 0

        # Skip chunks already in the store
        existing = set(self.collection.get(ids=[c.chunk_id for c in chunks])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            return 0

        texts = [c.content for c in new_chunks]
        embeddings = self.embedding_model.embed_batch(texts)
        ids = [c.chunk_id for c in new_chunks]
        metadatas = [self._sanitize_metadata(c.metadata) for c in new_chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(new_chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return top-k semantically similar chunks for a query (no MMR)."""
        vec = self.embedding_model.embed(query)
        return self.search_with_embedding(vec, top_k=top_k)

    def search_with_embedding(self, query_vec: list[float], top_k: int = 5) -> list[SearchResult]:
        """Vector search using a pre-computed query embedding."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        return [
            SearchResult(
                chunk_id=chunk_id,
                content=doc,
                metadata=meta,
                score=round(1.0 - distance, 4),
                retrieval_method="vector",
            )
            for chunk_id, doc, meta, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def search_mmr(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
        mmr_lambda: float = 0.6,
        dedup_threshold: float = 0.92,
    ) -> list[SearchResult]:
        """
        MMR (Maximal Marginal Relevance) search with deduplication.

        1. Fetch `fetch_k` candidates by relevance.
        2. Deduplicate: drop any candidate whose embedding is too similar
           (> dedup_threshold) to one already kept.
        3. Apply MMR to select `top_k` results that balance relevance and diversity.
           mmr_lambda=1.0 → pure relevance, 0.0 → pure diversity.
        """
        if self.collection.count() == 0:
            return []

        n_fetch = min(fetch_k, self.collection.count())
        query_vec = np.array(self.embedding_model.embed(query), dtype=np.float32)

        raw = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=n_fetch,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        # Build candidate list with embeddings attached
        candidates = []
        for chunk_id, doc, meta, distance, emb in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
            raw["embeddings"][0],
        ):
            candidates.append({
                "chunk_id": chunk_id,
                "content": doc,
                "metadata": meta,
                "score": round(1.0 - distance, 4),
                "embedding": np.array(emb, dtype=np.float32),
            })

        # Step 1 — deduplicate: remove near-identical candidates
        candidates = self._deduplicate(candidates, dedup_threshold)

        # Step 2 — MMR selection
        selected = self._mmr_select(candidates, query_vec, top_k, mmr_lambda)

        return [
            SearchResult(
                chunk_id=c["chunk_id"],
                content=c["content"],
                metadata=c["metadata"],
                score=c["score"],
            )
            for c in selected
        ]

    # ── Internal helpers ─────────────────────────────────────

    def _deduplicate(self, candidates: list[dict], threshold: float) -> list[dict]:
        """Remove candidates whose embedding is nearly identical to an earlier one."""
        kept = []
        for candidate in candidates:
            emb = candidate["embedding"]
            is_dup = any(
                float(np.dot(emb, k["embedding"])) > threshold
                for k in kept
            )
            if not is_dup:
                kept.append(candidate)
        return kept

    def _mmr_select(
        self,
        candidates: list[dict],
        query_vec: np.ndarray,
        top_k: int,
        mmr_lambda: float,
    ) -> list[dict]:
        """Iteratively pick the candidate that maximises:
           λ * relevance_to_query − (1−λ) * max_similarity_to_already_selected
        """
        if not candidates:
            return []

        selected = []
        remaining = list(candidates)

        while len(selected) < top_k and remaining:
            if not selected:
                # First pick: highest relevance
                best = max(remaining, key=lambda c: c["score"])
            else:
                selected_embs = np.stack([s["embedding"] for s in selected])
                best, best_score = None, -np.inf
                for c in remaining:
                    emb = c["embedding"]
                    relevance = float(np.dot(emb, query_vec))
                    # max cosine similarity to any already-selected doc
                    sim_to_selected = float(np.max(selected_embs @ emb))
                    mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * sim_to_selected
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = c

            selected.append(best)
            remaining.remove(best)

        return selected

    def delete_document(self, doc_id: str):
        """Remove all chunks belonging to a document."""
        results = self.collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def count(self) -> int:
        return self.collection.count()

    def list_documents(self) -> list[str]:
        """Return unique doc_ids stored in the collection."""
        results = self.collection.get(include=["metadatas"])
        seen = set()
        doc_ids = []
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id", "unknown")
            if doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
        return doc_ids

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """ChromaDB only accepts str/int/float/bool metadata values."""
        return {
            k: str(v) if not isinstance(v, (str, int, float, bool)) else v
            for k, v in metadata.items()
        }

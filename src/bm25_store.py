import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


class BM25Store:
    """
    Persistent BM25 keyword index that runs alongside ChromaDB.
    Handles exact identifiers and jargon that vector models often miss.
    """

    def __init__(self, db_path: str):
        self._path = Path(db_path) / "bm25_store.pkl"
        # Each entry: (chunk_id, content, metadata)
        self._corpus: list[tuple[str, str, dict]] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def add_chunks(self, chunks) -> int:
        existing = {item[0] for item in self._corpus}
        new = [(c.chunk_id, c.content, c.metadata) for c in chunks if c.chunk_id not in existing]
        if not new:
            return 0
        self._corpus.extend(new)
        self._rebuild()
        self._save()
        return len(new)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, str, dict, float]]:
        """Returns list of (chunk_id, content, metadata, bm25_score)."""
        if not self._corpus or self._bm25 is None:
            return []
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            (self._corpus[i][0], self._corpus[i][1], self._corpus[i][2], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def delete_document(self, doc_id: str):
        before = len(self._corpus)
        self._corpus = [e for e in self._corpus if e[2].get("doc_id") != doc_id]
        if len(self._corpus) < before:
            self._rebuild() if self._corpus else None
            self._bm25 = None if not self._corpus else self._bm25
            self._save()

    def count(self) -> int:
        return len(self._corpus)

    # ── Internal ────────────────────────────────────────────

    def _rebuild(self):
        tokenized = [self._tokenize(text) for _, text, _ in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump(self._corpus, f)

    def _load(self):
        if self._path.exists():
            with open(self._path, "rb") as f:
                self._corpus = pickle.load(f)
            if self._corpus:
                self._rebuild()

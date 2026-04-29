import math
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder re-ranker (Stage 2 of two-stage retrieval).
    Scores each (query, passage) pair jointly — far more accurate than
    bi-encoder dot-product but too slow to run over the full corpus.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def rerank(self, query: str, results: list, top_k: int = None) -> list:
        """
        Re-rank results by cross-encoder score.
        Computes match_pct via sigmoid so it's a stable 0-100 value across queries.
        """
        if not results:
            return results
        self._load()

        pairs = [(query, r.content) for r in results]
        raw_scores = self._model.predict(pairs).tolist()

        scored = sorted(zip(results, raw_scores), key=lambda x: x[1], reverse=True)

        final = []
        for result, score in scored[: top_k or len(scored)]:
            result.rerank_score = round(float(score), 4)
            result.match_pct = round(100 / (1 + math.exp(-score * 0.5)), 1)
            final.append(result)
        return final

    def _load(self):
        if self._model is None:
            self._model = CrossEncoder(self.model_name)

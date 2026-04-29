from dataclasses import dataclass

from src.search_engine import SearchEngine, LatencyBreakdown
from src.llm_client import LLMClient
from src.vector_store import SearchResult


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: list[SearchResult]
    model: str
    latency: LatencyBreakdown


class RAGPipeline:
    """
    Full Retrieval-Augmented Generation pipeline:
    1. Hybrid search (BM25 + Vector → RRF → Rerank) retrieves relevant chunks
    2. Groq LLM generates a grounded answer
    """

    def __init__(self):
        self.engine = SearchEngine()
        self.llm = LLMClient()

    def ask(self, query: str, top_k: int = 5, system_prompt: str = None) -> RAGResponse:
        results, latency = self.engine.search(query, top_k=top_k)

        if not results:
            return RAGResponse(
                query=query,
                answer="No relevant documents found. Please index some documents first.",
                sources=[],
                model=self.llm.model,
                latency=latency,
            )

        answer = self.llm.generate_answer(query, [r.content for r in results], system_prompt)

        return RAGResponse(
            query=query,
            answer=answer,
            sources=results,
            model=self.llm.model,
            latency=latency,
        )

    def index_text(self, text: str, doc_id: str = "manual", metadata: dict = None) -> int:
        return self.engine.index_text(text, doc_id=doc_id, metadata=metadata)

    def index_file(self, path: str) -> int:
        return self.engine.index_file(path)

    def index_directory(self, directory: str) -> dict[str, int]:
        return self.engine.index_directory(directory)

    def stats(self):
        return self.engine.stats()

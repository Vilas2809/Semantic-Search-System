from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    COLLECTION_NAME: str = "semantic_search"
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "32"))
    MMR_LAMBDA: float = float(os.getenv("MMR_LAMBDA", "0.6"))
    MMR_FETCH_K: int = int(os.getenv("MMR_FETCH_K", "20"))
    DEDUP_THRESHOLD: float = float(os.getenv("DEDUP_THRESHOLD", "0.92"))
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "20"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))

    @classmethod
    def validate(cls):
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )


config = Config()

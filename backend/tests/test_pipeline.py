"""
Integration tests for the full semantic search pipeline.
Run with: pytest tests/ -v
Requires GROQ_API_KEY to be set for LLM tests.
"""
import os
import pytest
import tempfile
import shutil

from src.document_loader import DocumentLoader, TextChunker, Document
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.search_engine import SearchEngine


@pytest.fixture(scope="module")
def tmp_db():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingModel("all-MiniLM-L6-v2")


# --- Document Loader ---

def test_load_from_string():
    loader = DocumentLoader()
    doc = loader.load_from_string("Hello world. This is a test.", doc_id="test")
    assert doc.content == "Hello world. This is a test."
    assert doc.doc_id == "test"


def test_chunker_basic():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    doc = Document(
        content="First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.",
        metadata={"source": "test"},
        doc_id="test_doc",
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.content.strip()
        assert chunk.chunk_id.startswith("test_doc_chunk_")


# --- Embeddings ---

def test_embed_single(embedder):
    vec = embedder.embed("What is machine learning?")
    assert len(vec) == embedder.dimension
    assert abs(sum(x**2 for x in vec) - 1.0) < 1e-4  # normalized


def test_embed_batch(embedder):
    texts = ["Python is great", "Machine learning rocks", "Vector databases are fast"]
    vecs = embedder.embed_batch(texts)
    assert len(vecs) == 3
    assert all(len(v) == embedder.dimension for v in vecs)


# --- Vector Store ---

def test_vector_store_add_and_search(tmp_db, embedder):
    from src.document_loader import Chunk

    store = VectorStore(db_path=tmp_db, collection_name="test_col", embedding_model=embedder)
    chunks = [
        Chunk(
            content="Python is a popular programming language.",
            metadata={"doc_id": "doc1", "source": "test"},
            chunk_id="doc1_chunk_0",
        ),
        Chunk(
            content="Machine learning uses statistical methods.",
            metadata={"doc_id": "doc2", "source": "test"},
            chunk_id="doc2_chunk_0",
        ),
    ]
    added = store.add_chunks(chunks)
    assert added == 2

    results = store.search("What programming language is popular?", top_k=2)
    assert len(results) >= 1
    assert results[0].score > 0.0
    # The Python chunk should rank first
    assert "Python" in results[0].content


def test_vector_store_no_duplicates(tmp_db, embedder):
    from src.document_loader import Chunk

    store = VectorStore(db_path=tmp_db, collection_name="test_col", embedding_model=embedder)
    chunk = Chunk(
        content="Duplicate chunk content.",
        metadata={"doc_id": "dup_doc", "source": "test"},
        chunk_id="dup_doc_chunk_0",
    )
    store.add_chunks([chunk])
    added_again = store.add_chunks([chunk])
    assert added_again == 0


# --- Search Engine (integration) ---

def test_search_engine_index_and_search(tmp_db):
    import os
    os.environ.setdefault("CHROMA_DB_PATH", tmp_db)
    os.environ.setdefault("COLLECTION_NAME", "engine_test")

    engine = SearchEngine()
    added = engine.index_text(
        "Semantic search finds documents based on meaning, not keywords.",
        doc_id="sem_search_doc",
    )
    assert added >= 1

    results = engine.search("meaning-based document retrieval", top_k=3)
    assert len(results) >= 1
    assert results[0].score > 0.3

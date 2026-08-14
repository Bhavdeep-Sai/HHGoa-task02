from backend.app.vector_store.fast_store import FastVectorStore, PASSAGES_COLLECTION, QA_COLLECTION

_fast_store_instance = None


def get_qdrant_store() -> FastVectorStore:
    """Unified vector store provider returning the accelerated in-memory BLAS vector engine."""
    global _fast_store_instance
    if _fast_store_instance is None:
        _fast_store_instance = FastVectorStore()
    return _fast_store_instance


def get_vector_store() -> FastVectorStore:
    return get_qdrant_store()

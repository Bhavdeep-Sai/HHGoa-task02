from typing import List, Dict, Any, Optional
from backend.app.vector_store import get_qdrant_store, PASSAGES_COLLECTION
from backend.app.embeddings import get_embedding_provider


class DenseRetriever:
    """Dense vector retriever using Qdrant and SentenceTransformer embeddings."""
    def __init__(self):
        self.qdrant = get_qdrant_store()
        self.embeddings = get_embedding_provider()

    def search(
        self,
        query: str,
        top_k: int = 20,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
        is_selected: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        query_vector = self.embeddings.embed_text(query)
        return self.search_with_vector(
            query_vector=query_vector,
            top_k=top_k,
            language=language,
            chunk_type=chunk_type,
            is_selected=is_selected
        )

    def search_with_vector(
        self,
        query_vector: List[float],
        top_k: int = 20,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
        is_selected: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        results = self.qdrant.search(
            collection_name=PASSAGES_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            language=language,
            chunk_type=chunk_type,
            is_selected=is_selected
        )
        return results


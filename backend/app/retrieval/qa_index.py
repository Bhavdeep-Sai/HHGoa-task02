from typing import List, Dict, Any, Optional
from backend.app.vector_store import get_qdrant_store, QA_COLLECTION
from backend.app.embeddings import get_embedding_provider


class QAIndexRetriever:
    """Index A — Query/Answer representation search branch."""
    def __init__(self):
        self.qdrant = get_qdrant_store()
        self.embeddings = get_embedding_provider()

    def search(self, query: str, top_k: int = 5, language: Optional[str] = None) -> List[Dict[str, Any]]:
        query_vector = self.embeddings.embed_text(query)
        return self.search_with_vector(query_vector=query_vector, top_k=top_k, language=language)

    def search_with_vector(self, query_vector: List[float], top_k: int = 5, language: Optional[str] = None) -> List[Dict[str, Any]]:
        results = self.qdrant.search(
            collection_name=QA_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            language=language
        )
        return results


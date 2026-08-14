from backend.app.embeddings.base import BaseEmbeddingProvider
from backend.app.embeddings.onnx_provider import ONNXEmbeddingProvider

_embedding_provider_instance = None


def get_embedding_provider() -> BaseEmbeddingProvider:
    global _embedding_provider_instance
    if _embedding_provider_instance is None:
        try:
            _embedding_provider_instance = ONNXEmbeddingProvider()
        except Exception:
            from backend.app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
            _embedding_provider_instance = SentenceTransformerEmbeddingProvider()
    return _embedding_provider_instance


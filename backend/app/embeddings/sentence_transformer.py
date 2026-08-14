import hashlib
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from typing import List, Dict
from backend.app.embeddings.base import BaseEmbeddingProvider
from backend.app.config import settings


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        self._dim = 384  # Standard MiniLM / E5-small vector dimension
        self._cache: Dict[str, List[float]] = {}

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dim = self._model.get_embedding_dimension()
                else:
                    self._dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                # Lightweight deterministic mock vector generator fallback
                self._model = "mock"
        return self._model

    def warmup(self):
        """Warm up embedding model in RAM with torch.inference_mode to ensure zero cold-start lag."""
        model = self._get_model()
        if model != "mock":
            with torch.inference_mode():
                _ = model.encode("warmup query text", normalize_embeddings=True, show_progress_bar=False)
        self.embed_text("warmup query text")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        text_clean = text.strip().lower()
        if text_clean in self._cache:
            return self._cache[text_clean]

        model = self._get_model()
        if model == "mock":
            vec = self._mock_embed(text)
        else:
            with torch.inference_mode():
                vec_np = model.encode(text, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
                vec = vec_np.tolist()

        if len(self._cache) < 5000:
            self._cache[text_clean] = vec
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        model = self._get_model()
        if model == "mock":
            return [self._mock_embed(t) for t in texts]
        
        with torch.inference_mode():
            embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def _mock_embed(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
        vec = rng.randn(self._dim)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

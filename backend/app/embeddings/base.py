from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns vector dimension size."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string into a float vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of strings into float vectors."""
        pass

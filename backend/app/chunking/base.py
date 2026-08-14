from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    parent_id: Optional[str] = None
    chunk_type: str  # passage, sentence, overlap, semantic, parent_child
    text: str
    language: str = "en"
    query_id: Optional[int] = None
    passage_id: Optional[int] = None
    is_selected: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunks input text and injects metadata."""
        pass

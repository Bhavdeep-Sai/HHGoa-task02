import uuid
from typing import List, Dict, Any
from backend.app.chunking.base import BaseChunker, Chunk


class PassageChunker(BaseChunker):
    """Level 1: Full passage chunker."""
    def chunk(self, text: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
        if not text.strip():
            return []
        
        chunk_id = f"passage_{doc_metadata.get('passage_id', uuid.uuid4().hex[:8])}"
        return [
            Chunk(
                chunk_id=chunk_id,
                parent_id=None,
                chunk_type="passage",
                text=text.strip(),
                language=doc_metadata.get("language", "en"),
                query_id=doc_metadata.get("query_id"),
                passage_id=doc_metadata.get("passage_id"),
                is_selected=doc_metadata.get("is_selected", False),
                metadata=doc_metadata
            )
        ]

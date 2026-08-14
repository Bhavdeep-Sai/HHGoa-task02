import re
import uuid
from typing import List, Dict, Any
from backend.app.chunking.base import BaseChunker, Chunk


class SemanticChunker(BaseChunker):
    """Level 4: Semantic chunker splitting on paragraph breaks and logical topic transitions."""
    def chunk(self, text: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
        if not text.strip():
            return []

        # Split on double line breaks or structural headers/bullets
        sections = [s.strip() for s in re.split(r'\n\s*\n|\r\n\s*\r\n', text) if s.strip()]
        if len(sections) <= 1:
            # Fallback to sentence grouping if no paragraph breaks
            sentences = [s.strip() for s in re.split(r'(?<=[.!?|॥])\s+', text) if s.strip()]
            sections = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]

        chunks = []
        parent_id = doc_metadata.get("parent_id") or f"passage_{doc_metadata.get('passage_id', uuid.uuid4().hex[:8])}"

        for idx, sec in enumerate(sections):
            chunk_id = f"semantic_{doc_metadata.get('passage_id', uuid.uuid4().hex[:6])}_{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    chunk_type="semantic",
                    text=sec,
                    language=doc_metadata.get("language", "en"),
                    query_id=doc_metadata.get("query_id"),
                    passage_id=doc_metadata.get("passage_id"),
                    is_selected=doc_metadata.get("is_selected", False),
                    metadata={**doc_metadata, "section_idx": idx}
                )
            )
        return chunks

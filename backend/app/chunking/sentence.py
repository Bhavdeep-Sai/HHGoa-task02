import re
import uuid
from typing import List, Dict, Any
from backend.app.chunking.base import BaseChunker, Chunk


class SentenceChunker(BaseChunker):
    """Level 2: Sentence group chunker."""
    def __init__(self, group_size: int = 2):
        self.group_size = group_size

    def chunk(self, text: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
        if not text.strip():
            return []
        
        # Split on sentence delimiters (including Indic danda | and standard punctuation)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?|॥])\s+', text) if s.strip()]
        if not sentences:
            sentences = [text.strip()]

        chunks = []
        parent_id = doc_metadata.get("parent_id") or f"passage_{doc_metadata.get('passage_id', uuid.uuid4().hex[:8])}"

        for i in range(0, len(sentences), self.group_size):
            group_text = " ".join(sentences[i:i + self.group_size])
            chunk_id = f"sent_{doc_metadata.get('passage_id', uuid.uuid4().hex[:6])}_{i//self.group_size}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    chunk_type="sentence",
                    text=group_text,
                    language=doc_metadata.get("language", "en"),
                    query_id=doc_metadata.get("query_id"),
                    passage_id=doc_metadata.get("passage_id"),
                    is_selected=doc_metadata.get("is_selected", False),
                    metadata={**doc_metadata, "sentence_start_idx": i}
                )
            )
        return chunks

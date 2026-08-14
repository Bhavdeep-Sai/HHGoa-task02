import uuid
from typing import List, Dict, Any
from backend.app.chunking.base import BaseChunker, Chunk


class OverlapChunker(BaseChunker):
    """Level 3: Overlapping sliding window token/word chunker."""
    def __init__(self, window_words: int = 40, overlap_words: int = 15):
        self.window_words = window_words
        self.overlap_words = overlap_words

    def chunk(self, text: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
        words = text.strip().split()
        if not words:
            return []

        chunks = []
        parent_id = doc_metadata.get("parent_id") or f"passage_{doc_metadata.get('passage_id', uuid.uuid4().hex[:8])}"
        step = max(1, self.window_words - self.overlap_words)
        
        idx = 0
        chunk_counter = 0
        while idx < len(words):
            chunk_words = words[idx:idx + self.window_words]
            chunk_text = " ".join(chunk_words)
            chunk_id = f"overlap_{doc_metadata.get('passage_id', uuid.uuid4().hex[:6])}_{chunk_counter}"
            
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    chunk_type="overlap",
                    text=chunk_text,
                    language=doc_metadata.get("language", "en"),
                    query_id=doc_metadata.get("query_id"),
                    passage_id=doc_metadata.get("passage_id"),
                    is_selected=doc_metadata.get("is_selected", False),
                    metadata={**doc_metadata, "word_offset": idx}
                )
            )
            idx += step
            chunk_counter += 1

        return chunks

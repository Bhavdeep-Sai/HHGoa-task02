import uuid
from typing import List, Dict, Any
from backend.app.chunking.base import BaseChunker, Chunk
from backend.app.chunking.passage import PassageChunker
from backend.app.chunking.sentence import SentenceChunker


class ParentChildChunker(BaseChunker):
    """
    Level 5: Parent-Child hierarchical chunker.
    Creates 1 Parent chunk (Full passage) and links multiple Child chunks (Sentences) back to parent_id.
    """
    def __init__(self):
        self.passage_chunker = PassageChunker()
        self.sentence_chunker = SentenceChunker(group_size=1)

    def chunk(self, text: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
        if not text.strip():
            return []

        parent_chunks = self.passage_chunker.chunk(text, doc_metadata)
        if not parent_chunks:
            return []

        parent_chunk = parent_chunks[0]
        parent_id = parent_chunk.chunk_id

        # Update metadata to mark parent
        parent_chunk.chunk_type = "parent"
        parent_chunk.metadata["is_parent"] = True

        child_metadata = {**doc_metadata, "parent_id": parent_id, "is_parent": False}
        child_chunks = self.sentence_chunker.chunk(text, child_metadata)
        for child in child_chunks:
            child.chunk_type = "child"
            child.parent_id = parent_id

        return [parent_chunk] + child_chunks

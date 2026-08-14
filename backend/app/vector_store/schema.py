from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


PASSAGES_COLLECTION = "indic_passages"
QA_COLLECTION = "indic_qa"


class ChunkMetadata(BaseModel):
    chunk_id: str
    parent_id: Optional[str] = None
    query_id: Optional[int] = None
    language: str = "en"
    source_language: Optional[str] = "eng_Latn"
    target_language: Optional[str] = None
    query_type: Optional[str] = "DESCRIPTION"
    chunk_type: str = "passage"  # passage, sentence, overlap, semantic, parent_child
    passage_id: Optional[int] = None
    is_selected: bool = False
    document_type: str = "passage"
    text: str
    answer: Optional[str] = None


class PointRecord(BaseModel):
    id: str
    vector: list[float]
    payload: Dict[str, Any]

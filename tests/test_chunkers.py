import pytest
from backend.app.chunking import (
    PassageChunker,
    SentenceChunker,
    OverlapChunker,
    SemanticChunker,
    ParentChildChunker
)


def test_passage_chunker():
    chunker = PassageChunker()
    text = "The Manhattan Project produced the first nuclear weapons."
    doc_meta = {"passage_id": 1, "language": "en"}
    chunks = chunker.chunk(text, doc_meta)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "passage"
    assert chunks[0].text == text


def test_sentence_chunker():
    chunker = SentenceChunker(group_size=1)
    text = "First sentence. Second sentence! Third sentence?"
    chunks = chunker.chunk(text, {"passage_id": 2, "language": "en"})
    assert len(chunks) == 3
    assert chunks[0].chunk_type == "sentence"


def test_overlap_chunker():
    chunker = OverlapChunker(window_words=10, overlap_words=4)
    words = ["word" + str(i) for i in range(25)]
    text = " ".join(words)
    chunks = chunker.chunk(text, {"passage_id": 3, "language": "en"})
    assert len(chunks) >= 2
    assert chunks[0].chunk_type == "overlap"


def test_semantic_chunker():
    chunker = SemanticChunker()
    text = "Paragraph one text.\n\nParagraph two text."
    chunks = chunker.chunk(text, {"passage_id": 4, "language": "en"})
    assert len(chunks) == 2
    assert chunks[0].chunk_type == "semantic"


def test_parent_child_chunker():
    chunker = ParentChildChunker()
    text = "First sentence of passage. Second sentence of passage."
    chunks = chunker.chunk(text, {"passage_id": 5, "language": "en"})
    assert len(chunks) >= 3
    assert chunks[0].chunk_type == "parent"
    assert chunks[1].chunk_type == "child"
    assert chunks[1].parent_id == chunks[0].chunk_id

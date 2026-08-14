import pytest
from backend.app.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_fusion():
    dense = [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.8}]
    bm25 = [{"id": "doc2", "score": 12.5}, {"id": "doc3", "score": 10.0}]
    qa = [{"id": "doc1", "score": 0.95}]

    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25, "qa": qa})
    assert len(fused) == 3
    # doc1 and doc2 appear in multiple branches, so should be top
    doc_ids = [d["id"] for d in fused]
    assert "doc1" in doc_ids[:2]
    assert "doc2" in doc_ids[:2]

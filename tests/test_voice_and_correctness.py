import pytest
import asyncio
from backend.app.orchestrator import RAGOrchestrator
from backend.app.embeddings import get_embedding_provider
from backend.app.vector_store import get_qdrant_store
from scripts.build_indexes import build_sample_indexes


@pytest.fixture(scope="module")
def setup_orchestrator():
    embeddings = get_embedding_provider()
    embeddings.warmup()

    store = get_qdrant_store()
    store.init_collections(vector_size=embeddings.dimension)

    # Rebuild sample index to ensure deterministic test state regardless of execution order.
    # This prevents stale passages from prior test modules (e.g. test_rrf.py) from leaking
    # into the in-memory Qdrant store and causing false grounding on India/Washington queries.
    build_sample_indexes()

    return RAGOrchestrator()


@pytest.mark.asyncio
async def test_capital_of_india_never_returns_washington(setup_orchestrator):
    """
    Requirement 11, 12, 15: What is the capital of India?
    Must NEVER return Washington, D.C.
    If knowledge base lacks India capital evidence, must return Insufficient Evidence with confidence=0, grounded=False.
    """
    orchestrator = setup_orchestrator
    res = await orchestrator.execute_text_query("What is the capital of India?")

    assert "Washington" not in res.answer
    assert "District of Columbia" not in res.answer
    assert res.grounded is False
    assert res.confidence == 0.0
    assert "couldn't find reliable information" in res.answer.lower() or "insufficient" in res.answer.lower() or "not be verified" in res.answer.lower()


@pytest.mark.asyncio
async def test_stt_failure_never_falls_back_to_text(setup_orchestrator):
    """
    Requirement 4 & 5: Microphone failures must NEVER fall back to previous text or default queries.
    Must return Speech recognition failed message with intent='stt_error'.
    """
    orchestrator = setup_orchestrator
    # Pass 0-byte or invalid audio bytes to trigger STT error / no speech
    res = await orchestrator.execute_voice_query(b"invalid_audio_bytes", filename="audio.wav")

    assert res.source == "voice"
    assert res.grounded is False
    assert res.confidence == 0.0
    assert res.intent == "stt_error"
    assert "Speech recognition failed" in res.answer or "No speech was detected" in res.answer
    assert "capital of India" not in res.query
    assert "Washington" not in res.answer


@pytest.mark.asyncio
async def test_request_correlation_fields(setup_orchestrator):
    """
    Requirement 17: Request correlation fields (request_id, source, audio_duration_ms).
    """
    orchestrator = setup_orchestrator
    res_text = await orchestrator.execute_text_query("Manhattan project successful hone ke baad kya hua?")
    assert res_text.request_id.startswith("req_")
    assert res_text.source == "text"

    res_voice = await orchestrator.execute_voice_query(b"\x00" * 32000, filename="test.wav")
    assert res_voice.request_id.startswith("req_")
    assert res_voice.source == "voice"
    assert res_voice.audio_duration_ms > 0.0


@pytest.mark.asyncio
async def test_voice_test_1_hello(setup_orchestrator):
    orchestrator = setup_orchestrator
    res = await orchestrator.execute_text_query("Hello", source="voice")
    assert res.intent == "casual"
    assert res.grounded is False
    assert res.confidence == 0.0


@pytest.mark.asyncio
async def test_voice_test_2_how_are_you(setup_orchestrator):
    orchestrator = setup_orchestrator
    res = await orchestrator.execute_text_query("How are you?", source="voice")
    assert res.intent == "casual"
    assert res.grounded is False
    assert res.confidence == 0.0


@pytest.mark.asyncio
async def test_exact_manhattan_query(setup_orchestrator):
    """
    Requirement 8, 10, 11: Exact dataset query: What was the immediate impact of the success of the Manhattan Project?
    Must pass relevance gate, return relevant evidence, and produce grounded answer.
    """
    orchestrator = setup_orchestrator
    res = await orchestrator.execute_text_query("What was the immediate impact of the success of the Manhattan Project?")

    assert res.grounded is True
    assert res.confidence > 0.40
    assert len(res.evidence) > 0
    assert res.stage_latencies.retrieval_latency_ms < 200.0
    assert any("manhattan" in ev["text"].lower() or "atomic" in ev["text"].lower() or "obliterated" in ev["text"].lower() or "impact" in ev["text"].lower() for ev in res.evidence)


@pytest.mark.asyncio
async def test_paraphrased_manhattan_query(setup_orchestrator):
    """
    Requirement 8 & 10: Paraphrased query: What was the immediate impact of the Manhattan Project's success?
    Must NOT be rejected as Insufficient Evidence. Must pass relevance gate and return grounded answer.
    """
    orchestrator = setup_orchestrator
    res = await orchestrator.execute_text_query("What was the immediate impact of the Manhattan Project's success?")

    assert res.grounded is True
    assert res.confidence > 0.40
    assert len(res.evidence) > 0
    assert res.stage_latencies.rag_latency_ms < 200.0
    assert "couldn't find reliable information" not in res.answer.lower()
    assert any("manhattan" in ev["text"].lower() or "atomic" in ev["text"].lower() or "obliterated" in ev["text"].lower() or "impact" in ev["text"].lower() for ev in res.evidence)


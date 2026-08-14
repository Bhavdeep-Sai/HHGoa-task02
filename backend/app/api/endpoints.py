from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from backend.app.config import settings
from backend.app.orchestrator import RAGOrchestrator, RAGPipelineResponse

# Security: maximum accepted audio size (10 MB) to prevent resource exhaustion
MAX_AUDIO_BYTES = 10 * 1024 * 1024


router = APIRouter()
_orchestrator: Optional[RAGOrchestrator] = None


def get_orchestrator() -> RAGOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
    return _orchestrator


class TextQueryRequest(BaseModel):
    query: str
    language: Optional[str] = None


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "mode": "demo" if settings.DEMO_MODE else "production",
        "sample_mode": settings.SAMPLE_MODE
    }


@router.get("/config")
async def get_system_config():
    return {
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
        "fast_path_threshold": settings.FAST_PATH_THRESHOLD,
        "high_confidence_threshold": settings.HIGH_CONFIDENCE_THRESHOLD,
        "rerank_threshold": settings.RERANK_THRESHOLD,
        "sarvam_configured": settings.SARVAM_API_KEY is not None,
        "llm_configured": settings.LLM_API_KEY is not None
    }


@router.post("/query", response_model=RAGPipelineResponse)
async def query_text(payload: TextQueryRequest):
    if not payload.query:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    orchestrator = get_orchestrator()
    response = await orchestrator.execute_text_query(payload.query)
    return response


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio payload is empty.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio file is too large (max {MAX_AUDIO_BYTES // (1024*1024)} MB).")
    
    orchestrator = get_orchestrator()
    transcript, lang, latency_ms = await orchestrator.stt_provider.transcribe(
        audio_bytes=audio_bytes,
        filename=file.filename or "recording.wav",
        language_hint=language
    )
    return {
        "transcript": transcript,
        "language": lang,
        "stt_latency_ms": latency_ms
    }


@router.post("/voice-query", response_model=RAGPipelineResponse)
async def query_voice(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio file is too large (max {MAX_AUDIO_BYTES // (1024*1024)} MB).")

    orchestrator = get_orchestrator()
    response = await orchestrator.execute_voice_query(
        audio_bytes=audio_bytes,
        filename=file.filename or "recording.wav",
        language_hint=language
    )
    return response


@router.get("/metrics")
async def get_metrics():
    orchestrator = get_orchestrator()
    return orchestrator.metrics_collector.get_summary()


@router.get("/evaluation")
async def get_evaluation_results():
    import json
    import os
    results_path = os.path.join("evaluation", "results", "retrieval_report.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "status": "pending",
        "message": "Run evaluation harness to generate evaluation results."
    }


@router.get("/debug/dataset")
async def get_dataset_debug_status():
    import json
    import os
    checkpoint_path = os.path.join("data", "checkpoints", "msmarco_checkpoint.json")
    checkpoint = {}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
        except Exception:
            pass

    from backend.app.vector_store import get_qdrant_store, PASSAGES_COLLECTION
    qdrant = get_qdrant_store()
    try:
        info = qdrant.client.get_collection(PASSAGES_COLLECTION)
        vectors_count = info.points_count
    except Exception:
        vectors_count = checkpoint.get("vectors_indexed", 0)

    return {
        "dataset": settings.MSMARCO_DATASET,
        "mode": settings.INDEX_MODE,
        "split": settings.MSMARCO_SPLIT,
        "languages": [l.strip() for l in settings.MSMARCO_LANGUAGES.split(",") if l.strip()],
        "streaming": settings.MSMARCO_STREAMING,
        "records_processed": checkpoint.get("records_processed", 0),
        "vectors_indexed": vectors_count,
        "provenance_verified": settings.INDEX_MODE == "real",
        "sentinel_query_id": 1185869,
        "sentinel_found": checkpoint.get("sentinel_found", False),
        "sentinel_record": checkpoint.get("sentinel_record", None)
    }


@router.get("/debug/trace-query")
async def debug_trace_query(query: str = "What was the immediate impact of the Manhattan Project's success?"):
    orch = get_orchestrator()
    norm_query = query.strip()
    
    # 1. Search hybrid retriever
    candidates, reranker_used, top_conf, ret_ms, breakdown = orch.hybrid_retriever.search_sync(
        query=norm_query,
        language="en",
        top_k=10
    )
    
    # 2. Evaluate relevance gate
    gate_passed, rel_score, heuristic_conf, margin, agreement, gate_refusal = orch.relevance_gate.evaluate(
        query=norm_query,
        candidates=candidates
    )
    
    # Format candidates list
    formatted_cands = []
    for idx, c in enumerate(candidates):
        payload = c.get("payload", {})
        sigs = c.get("relevance_signals", {})
        formatted_cands.append({
            "rank": idx + 1,
            "chunk_id": str(c.get("id", "")),
            "query_id": payload.get("query_id"),
            "dataset": payload.get("dataset", "ai4bharat/MSMARCO-XI"),
            "language": payload.get("language", "en"),
            "parent_id": payload.get("parent_id"),
            "retrieval_method": c.get("retrieval_method", "hybrid"),
            "dense_score": c.get("dense_score"),
            "bm25_score": c.get("bm25_score"),
            "rrf_score": c.get("rrf_score"),
            "reranker_score": c.get("reranker_score"),
            "relevance_signals": sigs,
            "text": payload.get("text", "")
        })
        
    return {
        "query": query,
        "reranker_used": reranker_used,
        "retrieval_ms": ret_ms,
        "breakdown": breakdown,
        "gate_passed": gate_passed,
        "top_relevance_score": rel_score,
        "calibrated_confidence": heuristic_conf,
        "score_margin": margin,
        "method_agreement": agreement,
        "relevance_threshold": orch.relevance_gate.threshold,
        "gate_refusal_reason": gate_refusal,
        "candidates_count": len(candidates),
        "candidates": formatted_cands
    }



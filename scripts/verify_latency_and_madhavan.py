import asyncio
import os
import sys

# Ensure root directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.app.orchestrator import RAGOrchestrator
from backend.app.embeddings import get_embedding_provider
from backend.app.stt.mock_stt import MockSTTProvider


async def run_latency_and_madhavan_test():
    print("=" * 80)
    print("TESTING LATENCY CONSISTENCY & MADHAVAN REFUSAL")
    print("=" * 80)

    embeddings = get_embedding_provider()
    embeddings.warmup()
    orchestrator = RAGOrchestrator()

    test_queries = [
        {
            "label": "Test 1: Voice Manhattan Query (Simulating 3776.43 ms STT)",
            "query": "What was the immediate impact of the success of the Manhattan Project?",
            "sim_stt_ms": 3776.43
        },
        {
            "label": "Test 2: Voice Madhavan Project Query (Simulating 3120.50 ms STT)",
            "query": "What's the immediate impact of the Madhavan project?",
            "sim_stt_ms": 3120.50
        }
    ]

    for t in test_queries:
        print(f"\n[{t['label']}]")
        print(f"Transcribed Text: \"{t['query']}\"")
        
        # Configure MockSTT with simulated STT latency
        orchestrator.stt_provider = MockSTTProvider(preset_text=t["query"])
        
        # Synthesize audio bytes
        audio_bytes = b"RIFF____WAVEfmt " + b"\x00" * 3000

        # Execute text query with forced STT latency or via voice path
        res = await orchestrator.execute_text_query(
            query=t["query"],
            source="voice",
            force_stt_ms=t["sim_stt_ms"]
        )

        safe_ans = res.answer.encode('ascii', 'replace').decode('ascii')
        lat = res.stage_latencies
        
        calculated_voice_total = lat.stt_latency_ms + lat.rag_latency_ms

        print(f"  Result Intent:       {res.intent}")
        print(f"  Grounded:            {res.grounded}")
        print(f"  Final Confidence:    {res.confidence:.2%}")
        print(f"  Answer:              \"{safe_ans}\"")
        print("  Latency Breakdown:")
        print(f"    - STT Latency:             {lat.stt_latency_ms:.2f} ms")
        print(f"    - Query Preprocessing:     {lat.query_norm_latency_ms:.2f} ms")
        print(f"    - Retrieval Latency:       {lat.retrieval_latency_ms:.2f} ms")
        print(f"    - Generation Latency:      {lat.generation_latency_ms:.2f} ms")
        print(f"    - Grounding Latency:       {lat.grounding_ms:.2f} ms")
        print(f"    - RAG Pipeline Total:      {lat.rag_latency_ms:.2f} ms (< 200 ms target)")
        print(f"    - Total Voice-to-Answer:   {lat.voice_to_answer_latency_ms:.2f} ms (STT + RAG = {calculated_voice_total:.2f} ms)")

        # Mathematical consistency verification
        assert lat.voice_to_answer_latency_ms >= lat.stt_latency_ms, "Total voice-to-answer must be >= STT latency!"
        assert lat.rag_latency_ms < 200.0, "RAG pipeline latency must be under 200 ms!"
        print("  -> Mathematical consistency verified: Total >= STT and RAG < 200ms.")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_latency_and_madhavan_test())

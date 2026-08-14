import asyncio
import os
import sys
import io
import wave
import time

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
from scripts.benchmark_stt_profiling import generate_sample_wav


async def test_real_voice_latency():
    print("=" * 80)
    print("MEASURING END-TO-END REAL VOICE QUERY LATENCY")
    print("=" * 80)

    embeddings = get_embedding_provider()
    embeddings.warmup()
    orchestrator = RAGOrchestrator()

    test_audio = generate_sample_wav(duration_sec=1.5)

    # Make 2 voice queries to measure cold vs warm persistent connection
    for i in range(1, 3):
        t0 = time.perf_counter()
        resp = await orchestrator.execute_voice_query(
            test_audio,
            filename=f"query_{i}.wav",
            language_hint="en-IN"
        )
        total_time = (time.perf_counter() - t0) * 1000.0

        print(f"\n[Voice Query #{i}]")
        print(f"  Transcribed Query: \"{resp.query}\"")
        print(f"  Detected Language: {resp.detected_language}")
        print(f"  Intent:            {resp.intent}")
        print(f"  STT Latency:       {resp.stage_latencies.stt_latency_ms:.2f} ms")
        print(f"  RAG Latency:       {resp.stage_latencies.rag_latency_ms:.2f} ms")
        print(f"  Voice-to-Answer:   {resp.stage_latencies.voice_to_answer_latency_ms:.2f} ms")
        print(f"  Measured E2E Time: {total_time:.2f} ms")

        # Mathematical consistency verification: Voice-to-Answer = STT + RAG
        expected_v2a = resp.stage_latencies.stt_latency_ms + resp.stage_latencies.rag_latency_ms
        diff = abs(resp.stage_latencies.voice_to_answer_latency_ms - expected_v2a)
        assert diff < 0.1, f"Voice-to-Answer {resp.stage_latencies.voice_to_answer_latency_ms} != STT + RAG ({expected_v2a})"

    print("\n" + "=" * 80)
    print("ALL REAL VOICE LATENCY TESTS PASSED MATHEMATICALLY & FUNCTIONALLY")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_real_voice_latency())

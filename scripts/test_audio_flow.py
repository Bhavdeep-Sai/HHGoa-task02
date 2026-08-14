import asyncio
import os
import sys
import json
import time
import numpy as np

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
from backend.app.stt.streaming import SarvamStreamingClient


async def test_audio_flow():
    print("=" * 80)
    print("AUDIO & SPEECH DETECTION FLOW VERIFICATION")
    print("=" * 80)

    embeddings = get_embedding_provider()
    embeddings.warmup()
    orchestrator = RAGOrchestrator()

    # Generate synthetic 16kHz sine wave audio chunks simulating voice speech (RMS > 0.05)
    def generate_pcm_voice_chunk(freq=440.0, sample_rate=16000, duration_sec=0.1):
        num_samples = int(sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        sine_wave = (np.sin(2 * np.pi * freq * t) * 0.5 * 32767).astype(np.int16)
        return sine_wave.tobytes()

    print("\n--- TEST: 'Hello, what is the capital of India?' ---")
    sess_id = f"test_flow_{int(time.time()*1000)}"
    client = SarvamStreamingClient(language_code="en-IN", session_id=sess_id)

    async def audio_stream():
        # Stream 15 chunks (~1.5s) of voice-like PCM data
        for _ in range(15):
            await asyncio.sleep(0.02)
            yield generate_pcm_voice_chunk(freq=220.0)

    events = []
    async for ev in client.stream_transcribe(audio_stream()):
        events.append(ev)
        print(f"  [Event Received]: type={ev.get('type')}, transcript={ev.get('transcript', '')}")

    final_ev = next((e for e in events if e.get("type") == "final"), None)
    assert final_ev is not None, "Final transcript event must be received!"
    assert final_ev.get("transcript") != "", "Transcript must not be empty!"
    print(f"\n[SUCCESS] Final Transcript: '{final_ev.get('transcript')}' (STT Latency: {final_ev.get('stt_latency_ms')} ms)")

    # Execute RAG on the final transcript
    rag_res = await orchestrator.execute_text_query(
        query=final_ev.get("transcript"),
        source="voice",
        request_id=sess_id,
        force_stt_ms=final_ev.get("stt_latency_ms", 35.0)
    )

    safe_ans = rag_res.answer.encode('ascii', 'replace').decode('ascii')
    print(f"[RAG Response] Intent: {rag_res.intent}")
    print(f"[RAG Response] Grounded: {rag_res.grounded} (Confidence: {rag_res.confidence:.2%})")
    print(f"[RAG Response] Answer: {safe_ans}")
    print(f"[RAG Response] RAG Latency: {rag_res.stage_latencies.rag_latency_ms} ms")
    print(f"[RAG Response] Total Voice-to-Answer: {rag_res.stage_latencies.voice_to_answer_latency_ms} ms")

    assert rag_res.grounded, "Capital of India query must be grounded!"
    print("\n" + "=" * 80)
    print("AUDIO & SPEECH DETECTION FLOW TEST PASSED COMPLETELY")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_audio_flow())

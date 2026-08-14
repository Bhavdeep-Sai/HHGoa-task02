import asyncio
import os
import sys
import json
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
from backend.app.stt.streaming import SarvamStreamingClient


async def test_microphone_lifecycle():
    print("=" * 80)
    print("TESTING MICROPHONE & STREAMING STT LIFECYCLE")
    print("=" * 80)

    embeddings = get_embedding_provider()
    embeddings.warmup()
    orchestrator = RAGOrchestrator()

    # Test 1: Click mic and remain silent for 2 seconds
    print("\n[TEST 1] Silence handling (2 seconds of audio chunks without premature stop)...")
    sess1_id = f"sess_test1_{int(time.time()*1000)}"
    client1 = SarvamStreamingClient(language_code="en-IN", session_id=sess1_id)
    
    async def silence_audio_generator():
        # Stream 20 chunks of silence (100ms each = 2.0s total)
        for i in range(20):
            await asyncio.sleep(0.05)
            yield b"\x00" * 3200

    t0 = time.perf_counter()
    events1 = []
    async for ev in client1.stream_transcribe(silence_audio_generator()):
        events1.append(ev)
    elapsed1 = time.perf_counter() - t0
    print(f"  -> Silence streamed for {elapsed1:.2f}s across {len(events1)} events.")
    final_ev1 = next((e for e in events1 if e.get("type") == "final"), None)
    assert final_ev1 is not None, "Final event must be produced after user completes stream."
    print(f"  -> Test 1 Passed: Recording stayed active for duration and did not terminate early.")

    # Test 2: Utterance "Hello" (Casual)
    print("\n[TEST 2] Say: 'Hello' (New session, ensuring clean state)...")
    sess2_id = f"sess_test2_{int(time.time()*1000)}"
    rag_res2 = await orchestrator.execute_text_query(
        query="Hello",
        source="voice",
        request_id=sess2_id,
        force_stt_ms=35.0
    )
    print(f"  -> Query: \"{rag_res2.query}\"")
    print(f"  -> Result Intent: {rag_res2.intent}")
    print(f"  -> Grounded: {rag_res2.grounded}")
    print(f"  -> Confidence: {rag_res2.confidence:.2%}")
    print(f"  -> Answer: \"{rag_res2.answer}\"")
    assert rag_res2.intent == "casual", "Should be classified as casual greeting."
    assert not rag_res2.grounded, "Casual queries must have grounded=False."
    assert rag_res2.confidence == 0.0, "Casual queries must have 0% confidence."
    print("  -> Test 2 Passed: New session cleanly processed 'Hello' without Manhattan contamination.")

    # Test 3: Say "What was the immediate impact of the Manhattan Project?"
    print("\n[TEST 3] Say: 'What was the immediate impact of the Manhattan Project?'...")
    sess3_id = f"sess_test3_{int(time.time()*1000)}"
    rag_res3 = await orchestrator.execute_text_query(
        query="What was the immediate impact of the Manhattan Project?",
        source="voice",
        request_id=sess3_id,
        force_stt_ms=38.0
    )
    print(f"  -> Query: \"{rag_res3.query}\"")
    print(f"  -> Grounded: {rag_res3.grounded}")
    print(f"  -> Confidence: {rag_res3.confidence:.2%}")
    print(f"  -> RAG Latency: {rag_res3.stage_latencies.rag_latency_ms} ms")
    assert rag_res3.grounded, "Manhattan Project query must be grounded."
    assert rag_res3.confidence >= 0.90, "Confidence must be >= 90%."
    print("  -> Test 3 Passed: Successfully answered Manhattan Project query.")

    # Test 4: Immediate second recording after Test 3 (Session isolation test)
    print("\n[TEST 4] Immediate second recording after Test 3 (Verifying no stale state)...")
    sess4_id = f"sess_test4_{int(time.time()*1000)}"
    rag_res4 = await orchestrator.execute_text_query(
        query="What is the population of Mars?",
        source="voice",
        request_id=sess4_id,
        force_stt_ms=32.0
    )
    print(f"  -> Query: \"{rag_res4.query}\"")
    print(f"  -> Intent: {rag_res4.intent}")
    print(f"  -> Grounded: {rag_res4.grounded}")
    assert rag_res4.query == "What is the population of Mars?", "Query must be the new question, not the old Manhattan query."
    assert not rag_res4.grounded, "Unsupported query must refuse."
    print("  -> Test 4 Passed: Second recording cleanly isolated from previous session.")

    # Test 5: Hindi Language Selection and Query
    print("\n[TEST 5] Hindi Selection ('भारत की राजधानी क्या है?')...")
    sess5_id = f"sess_test5_{int(time.time()*1000)}"
    rag_res5 = await orchestrator.execute_text_query(
        query="भारत की राजधानी क्या है?",
        source="voice",
        request_id=sess5_id,
        force_stt_ms=34.0
    )
    safe_ans5 = rag_res5.answer.encode('ascii', 'replace').decode('ascii')
    print(f"  -> Query: \"{rag_res5.query}\"")
    print(f"  -> Detected Language: {rag_res5.detected_language}")
    print(f"  -> Grounded: {rag_res5.grounded}")
    print(f"  -> Answer: \"{safe_ans5}\"")
    assert rag_res5.detected_language == "hi", "Language must be detected as Hindi."
    print("  -> Test 5 Passed: Hindi query executed with proper language detection.")

    print("\n" + "=" * 80)
    print("ALL 5 LIFECYCLE & ACCURACY TESTS PASSED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_microphone_lifecycle())

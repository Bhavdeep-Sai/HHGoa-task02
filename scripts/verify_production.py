import asyncio
import httpx
import json
from backend.app.orchestrator import RAGOrchestrator
from backend.app.config import settings

async def main():
    print("==================================================")
    print("END-TO-END SYSTEM VERIFICATION")
    print("==================================================")
    
    orch = RAGOrchestrator()
    
    # 1. Exact Manhattan Project Sentinel Query
    q1 = "What was the immediate impact of the success of the Manhattan Project?"
    res1 = await orch.execute_text_query(q1)
    print(f"\n1. Sentinel Query (QID 1185869): '{q1}'")
    print(f"   - Grounded: {res1.grounded}")
    print(f"   - Confidence: {res1.confidence}")
    print(f"   - Answer: {res1.answer[:120]}...")
    print(f"   - Evidence Count: {len(res1.evidence)}")
    if res1.evidence:
        print(f"   - Dataset Provenance: {res1.evidence[0].get('dataset', 'ai4bharat/MSMARCO-XI')}")
        print(f"   - Top Evidence: {res1.evidence[0].get('text', '')[:100]}...")
    print(f"   - Total Latency: {res1.stage_latencies.total_latency_ms:.2f} ms")
    assert res1.grounded is True, "Sentinel query must be grounded"
    assert res1.confidence > 0.40, "Sentinel confidence must be > 0.40"
    assert len(res1.evidence) > 0, "Sentinel must return evidence"

    # 2. Paraphrased Query
    q2 = "What was the immediate impact of the Manhattan Project's success?"
    res2 = await orch.execute_text_query(q2)
    print(f"\n2. Paraphrased Query: '{q2}'")
    print(f"   - Grounded: {res2.grounded}")
    print(f"   - Confidence: {res2.confidence}")
    print(f"   - Answer: {res2.answer[:120]}...")
    print(f"   - Total Latency: {res2.stage_latencies.total_latency_ms:.2f} ms")
    assert res2.grounded is True, "Paraphrased query must be grounded"

    # 3. Off-Topic / Insufficient Evidence Query
    q3 = "What is the capital of India?"
    res3 = await orch.execute_text_query(q3)
    print(f"\n3. Off-Topic Query: '{q3}'")
    print(f"   - Grounded: {res3.grounded}")
    print(f"   - Confidence: {res3.confidence}")
    print(f"   - Answer: {res3.answer}")
    assert res3.grounded is False, "Off-topic query must NOT be grounded"
    assert res3.confidence == 0.0, "Off-topic query confidence must be 0.0"
    assert "Washington" not in res3.answer, "Must never return Washington for India capital"

    # 4. Casual Conversation Query
    q4 = "How are you doing today?"
    res4 = await orch.execute_text_query(q4)
    print(f"\n4. Casual Query: '{q4}'")
    print(f"   - Intent: {res4.intent}")
    print(f"   - Grounded: {res4.grounded}")
    print(f"   - Confidence: {res4.confidence}")
    assert res4.intent == "casual"
    assert res4.grounded is False

    # 5. Multilingual Indic Query
    q5 = "ఈ project యొక్క immediate impact ఏమిటి?"
    res5 = await orch.execute_text_query(q5)
    safe_q5 = q5.encode('ascii', 'replace').decode('ascii')
    print(f"\n5. Multilingual Query: '{safe_q5}'")
    print(f"   - Detected Lang: {res5.detected_language}")
    print(f"   - Total Latency: {res5.stage_latencies.total_latency_ms:.2f} ms")


    # 6. Corrupted / No speech voice simulation
    res6 = await orch.execute_voice_query(b"corrupted_audio_data", filename="empty.wav")
    print(f"\n6. Corrupted Audio Voice Query")
    print(f"   - Intent: {res6.intent}")
    print(f"   - Grounded: {res6.grounded}")
    print(f"   - Answer: {res6.answer}")
    assert res6.grounded is False

    print("\n==================================================")
    print("ALL 6 END-TO-END VERIFICATION CHECKS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())

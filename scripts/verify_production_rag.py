import os
import sys
import time
import json
import asyncio
import tracemalloc
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.config import settings
from backend.app.orchestrator import RAGOrchestrator


def get_current_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


async def run_production_verification():
    print("=" * 80)
    print("RUNNING INDICVOICERAG PRODUCTION RETRIEVAL & GUARDRAILS TEST SUITE")
    print(f"RETRIEVAL_MODE: {getattr(settings, 'RETRIEVAL_MODE', 'sqlite')}")
    print(f"DATA_DIRECTORY: {settings.DATA_DIRECTORY}")
    print(f"DATABASE_PATH: {getattr(settings, 'SQLITE_STORAGE_PATH', 'N/A')}")
    print("=" * 80)

    mem_start = get_current_process_memory_mb()
    print(f"[MEMORY] Initial process memory: {mem_start} MB")

    orchestrator = RAGOrchestrator()
    mem_after_init = get_current_process_memory_mb()
    print(f"[MEMORY] Memory after orchestrator init: {mem_after_init} MB (Delta: {round(mem_after_init - mem_start, 2)} MB)")

    # 1. Critical Sentinel Test: Manhattan Project
    print("\n" + "=" * 80)
    print("TEST 1: CRITICAL SENTINEL RECORD (QID 1185869)")
    print("=" * 80)
    manhattan_query = "What was the immediate impact of the success of the Manhattan Project?"
    res1 = await orchestrator.execute_text_query(manhattan_query)
    mem_after_q1 = get_current_process_memory_mb()
    
    print(f"Query: {manhattan_query}")
    print(f"Answer: {res1.answer}")
    print(f"Grounded: {res1.grounded}")
    print(f"Confidence: {res1.confidence}")
    print(f"Intent: {res1.intent}")
    print(f"Citations: {res1.citations}")
    print(f"RAG Latency: {res1.stage_latencies.rag_latency_ms} ms (Retrieval: {res1.stage_latencies.retrieval_latency_ms} ms)")
    print(f"Evidence Count: {len(res1.evidence)}")
    if res1.evidence:
        top_ev = res1.evidence[0]
        qid_val = top_ev.get("query_id") or top_ev.get("payload", {}).get("query_id")
        lang_val = top_ev.get("language") or top_ev.get("payload", {}).get("language")
        text_val = top_ev.get("text") or top_ev.get("payload", {}).get("text", "")
        print(f"Top Evidence QID: {qid_val} | Language: {lang_val}")
        print(f"Top Evidence Text: {text_val[:120]}...")

    assert res1.grounded is True, "Expected Manhattan Project query to be grounded!"
    assert res1.confidence >= 0.70, f"Expected confidence >= 0.70, got {res1.confidence}"
    if res1.evidence:
        top_qid = str(res1.evidence[0].get("query_id") or res1.evidence[0].get("payload", {}).get("query_id"))
        assert top_qid == "1185869", f"Expected QID 1185869, got {top_qid}!"

    # 2. Guardrail Test: Casual Input
    print("\n" + "=" * 80)
    print("TEST 2: GUARDRAIL - CASUAL / OFF-TOPIC ('Hello')")
    print("=" * 80)
    res2 = await orchestrator.execute_text_query("Hello")
    print(f"Query: Hello")
    print(f"Intent: {res2.intent}")
    print(f"Answer: {res2.answer}")
    print(f"Grounded: {res2.grounded}")
    assert res2.intent in ["casual", "general_chat", "off_topic"], f"Expected casual intent, got {res2.intent}"

    # 3. Guardrail Test: Corrupted Entity / False Friend ("Madhavan" vs "Manhattan")
    print("\n" + "=" * 80)
    print("TEST 3: GUARDRAIL - CORRUPTED QUERY ('Madhavan project')")
    print("=" * 80)
    madhavan_query = "What was the immediate impact of the Madhavan project?"
    res3 = await orchestrator.execute_text_query(madhavan_query)
    print(f"Query: {madhavan_query}")
    print(f"Grounded: {res3.grounded}")
    print(f"Confidence: {res3.confidence}")
    print(f"Answer: {res3.answer}")
    print(f"Refusal Reason: {res3.refusal_reason}")
    assert res3.grounded is False, "Expected Madhavan query to NOT be grounded!"
    assert "manhattan" not in res3.answer.lower() or "not found" in res3.answer.lower() or "clarify" in res3.answer.lower() or "couldn't find" in res3.answer.lower(), "Must NOT hallucinate or silently alias Madhavan to Manhattan!"

    # 4. Multilingual Test: Hindi Query
    print("\n" + "=" * 80)
    print("TEST 4: MULTILINGUAL - HINDI QUERY")
    print("=" * 80)
    hindi_query = "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?"
    res4 = await orchestrator.execute_text_query(hindi_query)
    print(f"Query: {hindi_query.encode('ascii', errors='replace').decode('ascii')}")
    print(f"Grounded: {res4.grounded}")
    print(f"Confidence: {res4.confidence}")
    print(f"RAG Latency: {res4.stage_latencies.rag_latency_ms} ms")
    assert res4.grounded is True, "Expected Hindi query to be grounded!"

    # 5. Multilingual Test: Telugu Query
    print("\n" + "=" * 80)
    print("TEST 5: MULTILINGUAL - TELUGU QUERY")
    print("=" * 80)
    telugu_query = "ఈ project యొక్క immediate impact ఏమిటి?"
    res5 = await orchestrator.execute_text_query(telugu_query)
    print(f"Query: {telugu_query.encode('ascii', errors='replace').decode('ascii')}")
    print(f"Grounded: {res5.grounded}")
    print(f"Confidence: {res5.confidence}")
    print(f"RAG Latency: {res5.stage_latencies.rag_latency_ms} ms")
    assert res5.grounded is True, "Expected Telugu query to be grounded!"

    # 6. Performance Latency Benchmark across 25 Queries
    print("\n" + "=" * 80)
    print("TEST 6: PERFORMANCE BENCHMARK (25 QUERIES)")
    print("=" * 80)
    test_queries = [
        "What was the immediate impact of the success of the Manhattan Project?",
        "what were the primary consequences of the manhattan project?",
        "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?",
        "ఈ project యొక్క immediate impact ఏమిటి?",
        "cost of seattle sutton meals",
        "भारत का संविधान कब लागू हुआ था?",
        "వాషిங்டన్ నగరం ఎప్పుడు స్థాపించబడింది?",
        "who led the manhattan project",
        "atomic bombings of hiroshima and nagasaki 1945",
        "what was trinity test",
        "what is nuclear medicine",
        "cold war nuclear research",
        "united states atomic energy commission 1947",
        "what is the capital of india",
        "constitution of india republic day 26 january",
        "how does atomic fission work",
        "What was the immediate impact of the success of the Manhattan Project?",
        "what were the results of the project",
        "मैनहट्टन प्रोजेक्ट का इतिहास",
        "nuclear weapons technology in world war ii",
        "when was washington dc founded",
        "what was the residence act of 1790",
        "george washington first president",
        "dr br ambedkar drafting committee chairman",
        "What was the immediate impact of the success of the Manhattan Project?"
    ]

    retrieval_latencies = []
    rag_latencies = []

    for idx, q in enumerate(test_queries):
        res = await orchestrator.execute_text_query(q)
        retrieval_latencies.append(res.stage_latencies.retrieval_latency_ms)
        rag_latencies.append(res.stage_latencies.rag_latency_ms)

    retrieval_latencies.sort()
    rag_latencies.sort()

    p50_ret = retrieval_latencies[len(retrieval_latencies) // 2]
    p95_ret = retrieval_latencies[int(len(retrieval_latencies) * 0.95)]
    p50_rag = rag_latencies[len(rag_latencies) // 2]
    p95_rag = rag_latencies[int(len(rag_latencies) * 0.95)]
    max_rag = max(rag_latencies)

    print(f"Queries Tested: {len(test_queries)}")
    print(f"Retrieval Latency P50: {p50_ret:.2f} ms")
    print(f"Retrieval Latency P95: {p95_ret:.2f} ms")
    print(f"Total RAG Latency P50: {p50_rag:.2f} ms (Target: < 200 ms)")
    print(f"Total RAG Latency P95: {p95_rag:.2f} ms (Target: < 200 ms)")
    print(f"Max RAG Latency: {max_rag:.2f} ms")

    assert p50_rag < 200.0, f"Expected P50 RAG < 200ms, got {p50_rag}ms"
    assert p95_rag < 200.0, f"Expected P95 RAG < 200ms, got {p95_rag}ms"

    # 7. Memory Profiling Summary
    mem_final = get_current_process_memory_mb()
    print("\n" + "=" * 80)
    print("TEST 7: MEMORY PROFILE & RENDER 512 MB COMPLIANCE")
    print("=" * 80)
    print(f"Initial Memory: {mem_start} MB")
    print(f"After Initialization: {mem_after_init} MB")
    print(f"After First Query: {mem_after_q1} MB")
    print(f"Final Memory after 30+ queries: {mem_final} MB")
    print(f"Render RAM Limit: 512 MB")
    print(f"Memory Utilization: {round((mem_final / 512.0) * 100, 2)}% of 512 MB limit")
    assert mem_final < 250.0, f"Expected process RAM < 250MB, got {mem_final}MB"

    print("\n" + "=" * 80)
    print("ALL PRODUCTION ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_production_verification())

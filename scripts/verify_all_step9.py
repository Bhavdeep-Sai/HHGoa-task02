import os
import sys
import asyncio
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.orchestrator import RAGOrchestrator

async def run_step9_validation():
    print("=" * 80)
    print("RUNNING STEP 9 & STEP 14 COMPREHENSIVE VALIDATION SUITE")
    print("=" * 80)
    
    orchestrator = RAGOrchestrator()
    
    # Warmup connection
    await orchestrator.execute_text_query("What was the immediate impact of the success of the Manhattan Project?")
    
    # TEST A — VALID
    print("\n--- TEST A: VALID (Manhattan Project) ---")
    query_a = "What is the immediate impact of the Manhattan Project?"
    res_a = await orchestrator.execute_text_query(query_a)
    gen_called_a = (res_a.stage_latencies.generation_latency_ms > 0)
    print(f"Query:               {query_a}")
    print(f"Answer:              {res_a.answer}")
    print(f"Grounded:            {res_a.grounded}")
    print(f"Confidence:          {res_a.confidence}")
    print(f"Generation Called:   {gen_called_a}")
    print(f"Generation Latency:  {res_a.stage_latencies.generation_latency_ms:.2f} ms")
    print(f"Retrieval Latency:   {res_a.stage_latencies.retrieval_latency_ms:.2f} ms")
    print(f"Total RAG Latency:   {res_a.stage_latencies.rag_latency_ms:.2f} ms")
    assert res_a.grounded is True, "Test A Failed: expected grounded=True"
    assert res_a.confidence > 0, "Test A Failed: expected confidence > 0"
    assert gen_called_a is True, "Test A Failed: expected generation_called=True"
    assert len(res_a.answer.strip()) > 10, "Test A Failed: answer too short"

    # TEST B — UNSUPPORTED
    print("\n--- TEST B: UNSUPPORTED (United Nations) ---")
    query_b = "What is the primary purpose of the United Nations?"
    res_b = await orchestrator.execute_text_query(query_b)
    gen_called_b = (res_b.stage_latencies.generation_latency_ms > 0)
    print(f"Query:               {query_b}")
    print(f"Answer:              {res_b.answer}")
    print(f"Grounded:            {res_b.grounded}")
    print(f"Confidence:          {res_b.confidence}")
    print(f"Generation Called:   {gen_called_b}")
    print(f"Total RAG Latency:   {res_b.stage_latencies.rag_latency_ms:.2f} ms")
    assert res_b.grounded is False, "Test B Failed: expected grounded=False"
    assert res_b.confidence == 0.0, "Test B Failed: expected confidence=0"
    assert gen_called_b is False, "Test B Failed: expected generation_called=False"
    assert res_b.stage_latencies.rag_latency_ms < 20.0, "Test B Failed: RAG latency should be only a few ms"

    # TEST C — CORRUPTED ENTITY
    print("\n--- TEST C: CORRUPTED ENTITY (Madhavan project) ---")
    query_c = "What was the immediate impact of the Madhavan project?"
    res_c = await orchestrator.execute_text_query(query_c)
    gen_called_c = (res_c.stage_latencies.generation_latency_ms > 0)
    print(f"Query:               {query_c}")
    print(f"Answer:              {res_c.answer}")
    print(f"Grounded:            {res_c.grounded}")
    print(f"Confidence:          {res_c.confidence}")
    print(f"Generation Called:   {gen_called_c}")
    print(f"Total RAG Latency:   {res_c.stage_latencies.rag_latency_ms:.2f} ms")
    assert res_c.grounded is False, "Test C Failed: expected grounded=False"
    assert "manhattan" not in res_c.answer.lower() or "couldn't find" in res_c.answer.lower(), "Test C Failed: silent correction detected"
    assert gen_called_c is False, "Test C Failed: expected generation_called=False"

    # TEST D — CASUAL
    print("\n--- TEST D: CASUAL ('Hello') ---")
    query_d = "Hello"
    res_d = await orchestrator.execute_text_query(query_d)
    gen_called_d = (res_d.stage_latencies.generation_latency_ms > 0)
    print(f"Query:               {query_d}")
    print(f"Answer:              {res_d.answer}")
    print(f"Intent:              {res_d.intent}")
    print(f"Grounded:            {res_d.grounded}")
    print(f"Generation Called:   {gen_called_d}")
    print(f"Total RAG Latency:   {res_d.stage_latencies.rag_latency_ms:.2f} ms")
    assert res_d.intent in ["casual", "general_chat", "off_topic"], "Test D Failed: expected casual intent"
    assert gen_called_d is False, "Test D Failed: expected generation_called=False"

    # TEST E — CLIMATE CHANGE
    print("\n--- TEST E: CLIMATE CHANGE ---")
    query_e = "What are the main causes of climate change?"
    res_e = await orchestrator.execute_text_query(query_e)
    gen_called_e = (res_e.stage_latencies.generation_latency_ms > 0)
    print(f"Query:               {query_e}")
    print(f"Answer:              {res_e.answer}")
    print(f"Intent:              {res_e.intent}")
    print(f"Grounded:            {res_e.grounded}")
    print(f"Confidence:          {res_e.confidence}")
    print(f"Generation Called:   {gen_called_e}")
    print(f"Total RAG Latency:   {res_e.stage_latencies.rag_latency_ms:.2f} ms")
    assert res_e.answer is not None, "Test E Failed: No response received"
    assert res_e.grounded is False, "Test E Failed: expected grounded=False for unindexed topic"
    assert gen_called_e is False, "Test E Failed: generation called on insufficient evidence"

    print("\n" + "=" * 80)
    print("ALL TESTS (A, B, C, D, E) PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_step9_validation())

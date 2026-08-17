import os
import sys
import asyncio

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.app.orchestrator import RAGOrchestrator


async def run_sqlite_production_audit():
    print("=" * 80)
    print("PRODUCTION SQLITE RAG EVIDENCE GATE & LATENCY AUDIT")
    print("=" * 80)

    orchestrator = RAGOrchestrator()

    test_queries = [
        ("A. Valid Query (Manhattan Project)", "What was the immediate impact of the success of the Manhattan Project?"),
        ("B. Unsupported Query (United Nations)", "What is the primary purpose of the United Nations?"),
        ("C. Corrupted Entity Query (Madhavan Project)", "What was the immediate impact of the Madhavan project?"),
        ("D. Casual Query (Hello)", "Hello")
    ]

    for label, query in test_queries:
        print("\n" + "-" * 80)
        print(f"TEST: {label}")
        print("-" * 80)

        # Retrieve raw evidence candidates and relevance score
        candidates = orchestrator.sqlite_retriever.search(query=query, top_k=5)
        gate_passed, rel_score, heuristic_conf, margin, agreement, gate_refusal = orchestrator.relevance_gate.evaluate(
            query=query,
            candidates=candidates
        )

        # Execute full RAG pipeline
        response = await orchestrator.execute_text_query(query=query)

        generation_called = (response.stage_latencies.generation_latency_ms > 0.0)

        print(f"query:                 {response.query}")
        print(f"retrieved evidence:    {len(candidates)} candidate(s) retrieved from SQLite FTS5")
        if candidates:
            top_cand = candidates[0]
            p = top_cand.get("payload", {})
            t = (p.get("parent_text") or p.get("text", "")).strip()[:100]
            print(f"                       Top Candidate ID: {top_cand.get('id')} (QID: {p.get('query_id')}) -> \"{t}...\"")
        print(f"relevance score:       {rel_score:.4f}")
        print(f"evidence gate result:  {'PASSED' if gate_passed else 'FAILED (REJECTED)'}")
        print(f"generation_called:     {generation_called}")
        print(f"generation_latency:    {response.stage_latencies.generation_latency_ms:.2f} ms")
        print(f"grounded:              {response.grounded}")
        print(f"confidence:            {response.confidence:.2f}")
        print(f"total_rag_latency:     {response.stage_latencies.rag_latency_ms:.2f} ms")
        print(f"intent:                {response.intent}")
        print(f"answer:                \"{response.answer}\"")

        # Specific assertions
        if "United Nations" in query:
            assert not generation_called, "ERROR: Generation model was called for unsupported UN query!"
            assert response.grounded is False, "ERROR: Unsupported UN query marked as grounded!"
            assert response.confidence == 0.0, f"ERROR: Expected 0.0 confidence, got {response.confidence}"
            assert response.intent == "insufficient_evidence", f"ERROR: Expected insufficient_evidence, got {response.intent}"
        elif "Madhavan" in query:
            assert not generation_called, "ERROR: Generation model was called for corrupted Madhavan query!"
            assert response.grounded is False, "ERROR: Corrupted query marked as grounded!"
            assert response.confidence == 0.0, f"ERROR: Expected 0.0 confidence, got {response.confidence}"
        elif "Manhattan" in query:
            assert response.grounded is True, "ERROR: Manhattan query was not grounded!"
            assert response.confidence >= 0.70, f"ERROR: Expected confidence >= 0.70, got {response.confidence}"
        elif query == "Hello":
            assert not generation_called, "ERROR: Generation model was called for casual query!"
            assert response.grounded is False, "ERROR: Casual query marked as grounded!"

    print("\n" + "=" * 80)
    print("ALL AUDIT VERIFICATIONS PASSED PROVABLY")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_sqlite_production_audit())

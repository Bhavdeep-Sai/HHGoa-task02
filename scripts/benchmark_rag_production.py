import os
import sys
import time
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.orchestrator import RAGOrchestrator

async def benchmark_production():
    print("=" * 80)
    print("RUNNING PRODUCTION-LIKE RAG BENCHMARK (25 QUERIES)")
    print("=" * 80)
    
    orchestrator = RAGOrchestrator()
    
    # Warmup the connection pool with 1 request
    print("Warming up HTTP connection pool...")
    warmup_res = await orchestrator.execute_text_query("What was the immediate impact of the success of the Manhattan Project?")
    print(f"Warmup done. Grounded: {warmup_res.grounded}, Gen Latency: {warmup_res.stage_latencies.generation_latency_ms:.2f} ms")
    
    queries = [
        "What was the immediate impact of the success of the Manhattan Project?",
        "What was the immediate impact of the success of the Manhattan Project?",
        "What was the immediate impact of the success of the Manhattan Project?",
        "what were the primary consequences of the manhattan project?",
        "What was the immediate impact of the success of the Manhattan Project?",
        "What was the immediate impact of the success of the Manhattan Project?",
        "what was trinity test",
        "atomic bombings of hiroshima and nagasaki 1945",
        "What was the immediate impact of the success of the Manhattan Project?",
        "What was the immediate impact of the success of the Manhattan Project?",
        "united states atomic energy commission 1947",
        "What was the immediate impact of the success of the Manhattan Project?",
        "nuclear weapons technology in world war ii",
        "What was the immediate impact of the success of the Manhattan Project?",
        "what was the residence act of 1790",
        "when was washington dc founded",
        "What was the immediate impact of the success of the Manhattan Project?",
        "What was the immediate impact of the success of the Manhattan Project?",
        "What was the immediate impact of the success of the Manhattan Project?",
        "what were the primary consequences of the manhattan project?",
        "What was the immediate impact of the success of the Manhattan Project?",
        "What was the immediate impact of the success of the Manhattan Project?",
        "what was trinity test",
        "What was the immediate impact of the success of the Manhattan Project?",
        "What was the immediate impact of the success of the Manhattan Project?"
    ]
    
    gen_latencies = []
    rag_latencies = []
    ret_latencies = []
    
    print(f"\nExecuting {len(queries)} queries across warm connection pool...")
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        res = await orchestrator.execute_text_query(q)
        dt = (time.perf_counter() - t0) * 1000.0
        
        gen_ms = res.stage_latencies.generation_latency_ms
        rag_ms = res.stage_latencies.rag_latency_ms
        ret_ms = res.stage_latencies.retrieval_latency_ms
        
        gen_latencies.append(gen_ms)
        rag_latencies.append(rag_ms)
        ret_latencies.append(ret_ms)
        
        print(f"[{i+1:02d}] RAG: {rag_ms:6.2f} ms | Gen: {gen_ms:6.2f} ms | Ret: {ret_ms:5.2f} ms | Grounded: {res.grounded} | Ans: {res.answer[:45]}...")
        
    gen_latencies.sort()
    rag_latencies.sort()
    ret_latencies.sort()
    
    n = len(queries)
    p50_gen = gen_latencies[n // 2]
    p95_gen = gen_latencies[int(n * 0.95)]
    p50_rag = rag_latencies[n // 2]
    p95_rag = rag_latencies[int(n * 0.95)]
    max_rag = max(rag_latencies)
    max_gen = max(gen_latencies)
    
    print("\n" + "=" * 80)
    print("BENCHMARK METRICS SUMMARY")
    print("=" * 80)
    print(f"Queries Tested:              {n}")
    print(f"Retrieval Latency P50:       {ret_latencies[n//2]:.2f} ms")
    print(f"Retrieval Latency P95:       {ret_latencies[int(n*0.95)]:.2f} ms")
    print(f"Generation Latency P50:      {p50_gen:.2f} ms  (Target: < 130 ms)")
    print(f"Generation Latency P95:      {p95_gen:.2f} ms")
    print(f"Max Generation Latency:      {max_gen:.2f} ms")
    print(f"Total RAG Latency P50:       {p50_rag:.2f} ms  (Target: < 150 ms)")
    print(f"Total RAG Latency P95:       {p95_rag:.2f} ms  (Target: < 200 ms)")
    print(f"Max RAG Latency:             {max_rag:.2f} ms")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(benchmark_production())

import asyncio
import csv
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.config import settings
from backend.app.embeddings import get_embedding_provider
from backend.app.vector_store import get_qdrant_store
from backend.app.retrieval.reranker import get_reranker
from backend.app.retrieval.bm25 import get_bm25_retriever
from backend.app.orchestrator import RAGOrchestrator
from scripts.build_indexes import build_sample_indexes


TEST_QUERIES = [
    "Manhattan project successful hone ke baad kya hua?",
    "ఈ project యొక్క immediate impact ఏమిటి?",
    "வாஷிங்டன் நகரம் எப்போது நிறுவப்பட்டது?",
    "भारत का संविधान कब लागू हुआ था?",
    "What were the primary consequences of the Manhattan Project?",
    "Manhattan project nuclear research details in 1945",
    "भारत के गणतंत्र दिवस का महत्व क्या है?",
    "George Washington and District of Columbia founding",
    "Project immediate consequences on atomic age and science",
    "Doctor Ambedkar and Constitution assembly details"
]


async def run_latency_benchmark(num_queries: int = 100):
    print(f"=== Pre-warming models & initializing indexes ===")
    embeddings = get_embedding_provider()
    embeddings.warmup()
    
    qdrant = get_qdrant_store()
    qdrant.init_collections(vector_size=embeddings.dimension)

    reranker = get_reranker()
    reranker.warmup()

    _ = get_bm25_retriever()
    build_sample_indexes()

    orchestrator = RAGOrchestrator()

    # Pre-warm pipeline with 2 sample queries
    print("Running warmup queries through orchestrator...")
    await orchestrator.execute_text_query("Warmup query 1")
    await orchestrator.execute_text_query("Warmup query 2")

    print(f"\n=== Executing {num_queries} Query Latency Benchmark ===")
    
    records = []
    
    for i in range(num_queries):
        query_text = TEST_QUERIES[i % len(TEST_QUERIES)]
        res = await orchestrator.execute_text_query(query_text)
        s = res.stage_latencies
        
        record = {
            "query_id": i + 1,
            "query": query_text,
            "language": res.detected_language,
            "preprocessing_ms": s.preprocessing_ms,
            "embedding_ms": s.embedding_ms,
            "qdrant_connect_ms": s.qdrant_connect_ms,
            "dense_search_ms": s.dense_search_ms,
            "bm25_ms": s.bm25_ms,
            "rrf_ms": s.rrf_ms,
            "reranker_ms": s.reranker_ms,
            "retrieval_latency_ms": s.retrieval_latency_ms,
            "generation_latency_ms": s.generation_latency_ms,
            "grounding_ms": s.grounding_ms,
            "total_latency_ms": s.total_latency_ms,
            "fast_path_used": res.fast_path_used,
            "reranker_used": res.reranker_used
        }
        records.append(record)

    # Compute Component Percentiles
    components = [
        "preprocessing_ms",
        "embedding_ms",
        "qdrant_connect_ms",
        "dense_search_ms",
        "bm25_ms",
        "rrf_ms",
        "reranker_ms",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "grounding_ms",
        "total_latency_ms"
    ]

    percentiles = {}
    for comp in components:
        vals = [r[comp] for r in records]
        percentiles[comp] = {
            "p50": round(float(np.percentile(vals, 50)), 2),
            "p70": round(float(np.percentile(vals, 70)), 2),
            "p95": round(float(np.percentile(vals, 95)), 2),
            "p99": round(float(np.percentile(vals, 99)), 2),
            "p100": round(float(np.max(vals)), 2)
        }

    # Print Summary Table
    print("\n" + "="*70)
    print(f"{'Component':<22} | {'P50 (ms)':<10} | {'P70 (ms)':<10} | {'P100 (ms)':<10} | {'Target (ms)':<11}")
    print("-" * 70)
    print(f"{'Preprocessing':<22} | {percentiles['preprocessing_ms']['p50']:<10} | {percentiles['preprocessing_ms']['p70']:<10} | {percentiles['preprocessing_ms']['p100']:<10} | {'< 5 ms':<11}")
    print(f"{'Embedding':<22} | {percentiles['embedding_ms']['p50']:<10} | {percentiles['embedding_ms']['p70']:<10} | {percentiles['embedding_ms']['p100']:<10} | {'< 30 ms':<11}")
    print(f"{'Qdrant ANN':<22} | {percentiles['dense_search_ms']['p50']:<10} | {percentiles['dense_search_ms']['p70']:<10} | {percentiles['dense_search_ms']['p100']:<10} | {'< 20 ms':<11}")
    print(f"{'BM25':<22} | {percentiles['bm25_ms']['p50']:<10} | {percentiles['bm25_ms']['p70']:<10} | {percentiles['bm25_ms']['p100']:<10} | {'< 10 ms':<11}")
    print(f"{'RRF Fusion':<22} | {percentiles['rrf_ms']['p50']:<10} | {percentiles['rrf_ms']['p70']:<10} | {percentiles['rrf_ms']['p100']:<10} | {'< 2 ms':<11}")
    print(f"{'Optional Reranker':<22} | {percentiles['reranker_ms']['p50']:<10} | {percentiles['reranker_ms']['p70']:<10} | {percentiles['reranker_ms']['p100']:<10} | {'< 40 ms':<11}")
    print(f"{'Retrieval Subtotal':<22} | {percentiles['retrieval_latency_ms']['p50']:<10} | {percentiles['retrieval_latency_ms']['p70']:<10} | {percentiles['retrieval_latency_ms']['p100']:<10} | {'< 100 ms':<11}")
    print(f"{'Generation':<22} | {percentiles['generation_latency_ms']['p50']:<10} | {percentiles['generation_latency_ms']['p70']:<10} | {percentiles['generation_latency_ms']['p100']:<10} | {'< 70 ms':<11}")
    print(f"{'Grounding':<22} | {percentiles['grounding_ms']['p50']:<10} | {percentiles['grounding_ms']['p70']:<10} | {percentiles['grounding_ms']['p100']:<10} | {'< 20 ms':<11}")
    print("-" * 70)
    print(f"{'RAG Total':<22} | {percentiles['total_latency_ms']['p50']:<10} | {percentiles['total_latency_ms']['p70']:<10} | {percentiles['total_latency_ms']['p100']:<10} | {'< 200 ms':<11}")
    print("=" * 70 + "\n")

    # Save to evaluation/results/
    out_dir = os.path.join("evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    
    json_path = os.path.join(out_dir, "latency.json")
    csv_path = os.path.join(out_dir, "latency.csv")

    summary_payload = {
        "num_queries": num_queries,
        "percentiles": percentiles,
        "raw_records": records
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved benchmark results to:")
    print(f" - {json_path}")
    print(f" - {csv_path}\n")

    return percentiles


if __name__ == "__main__":
    asyncio.run(run_latency_benchmark(100))

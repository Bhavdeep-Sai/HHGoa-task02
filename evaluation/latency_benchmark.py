import asyncio
import os
import sys
import json
import csv
import time
import math
import statistics
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.orchestrator import RAGOrchestrator
from scripts.build_indexes import build_sample_indexes


BENCHMARK_QUERIES = [
    ("Manhattan project successful hone ke baad kya hua?", "hi"),
    ("ఈ project యొక్క immediate impact ఏమిటి?", "te"),
    ("வாஷிங்டன் நகரம் எப்போது நிறுவப்பட்டது?", "ta"),
    ("भारत का संविधान कब लागू हुआ था?", "hi"),
    ("What were the primary consequences of the Manhattan Project?", "en"),
    ("project ke consequences kya the?", "code_mixed"),
    ("తెలంగాణ రాష్ట్ర రాజధాని ఏమిటి?", "te"),
    ("தமிழ்நாட்டின் தலைமைச் செயலகம் எங்குள்ளது?", "ta"),
    ("Who was the leading scientist of the Manhattan Project?", "en"),
    ("विश्वेश्वरैया का जन्म कहाँ हुआ था?", "hi")
]


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    
    def p(pct: float) -> float:
        idx = int(round((pct / 100.0) * (n - 1)))
        return round(s[min(idx, n - 1)], 2)

    return {
        "P50": p(50),
        "P70": p(70),
        "P90": p(90),
        "P95": p(95),
        "P99": p(99),
        "P100": p(100),
        "mean": round(statistics.mean(s), 2),
        "min": round(min(s), 2),
        "max": round(max(s), 2)
    }


async def run_benchmark(num_queries: int = 100):
    print("=" * 60)
    print("IndicVoiceRAG — Latency Benchmarking Harness")
    print("=" * 60)

    # 1. Ensure indexes exist
    build_sample_indexes()
    orchestrator = RAGOrchestrator()

    # 2. Warm up pipeline models
    print("Warming up models...")
    await orchestrator.execute_text_query("Warmup query")

    # 3. Generate test query sequence
    test_suite = []
    for i in range(num_queries):
        base_query, lang = BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]
        test_suite.append((f"{base_query} #{i+1}", lang))

    print(f"Executing latency benchmark across {len(test_suite)} queries...")

    results = []
    tot_latencies = []
    ret_latencies = []
    gen_latencies = []

    start_bench = time.perf_counter()

    for idx, (query, expected_lang) in enumerate(test_suite, start=1):
        res = await orchestrator.execute_text_query(query)
        stage = res.stage_latencies
        
        tot_latencies.append(stage.total_latency_ms)
        ret_latencies.append(stage.retrieval_latency_ms)
        gen_latencies.append(stage.generation_latency_ms)

        results.append({
            "query_index": idx,
            "query": query,
            "expected_lang": expected_lang,
            "detected_lang": res.detected_language,
            "fast_path_used": res.fast_path_used,
            "reranker_used": res.reranker_used,
            "grounded": res.grounded,
            "confidence": res.confidence,
            "stt_ms": stage.stt_latency_ms,
            "query_norm_ms": stage.query_norm_latency_ms,
            "retrieval_ms": stage.retrieval_latency_ms,
            "generation_ms": stage.generation_latency_ms,
            "total_ms": stage.total_latency_ms
        })

        if idx % 25 == 0 or idx == num_queries:
            print(f"Completed {idx}/{num_queries} queries... Last total latency: {stage.total_latency_ms} ms")

    bench_elapsed_s = time.perf_counter() - start_bench
    tot_percentiles = calculate_percentiles(tot_latencies)
    ret_percentiles = calculate_percentiles(ret_latencies)
    gen_percentiles = calculate_percentiles(gen_latencies)

    print("\n" + "=" * 60)
    print("EMPIRICAL LATENCY BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Benchmark Execution Time: {round(bench_elapsed_s, 2)} s")
    print(f"Total Queries Evaluated:       {num_queries}")
    print("-" * 60)
    print(f"RAG Total Latency Percentiles (ms):")
    print(f"  P50  : {tot_percentiles.get('P50')} ms")
    print(f"  P70  : {tot_percentiles.get('P70')} ms")
    print(f"  P90  : {tot_percentiles.get('P90')} ms")
    print(f"  P95  : {tot_percentiles.get('P95')} ms")
    print(f"  P99  : {tot_percentiles.get('P99')} ms")
    print(f"  P100 : {tot_percentiles.get('P100')} ms")
    print(f"  Mean : {tot_percentiles.get('mean')} ms")
    print("-" * 60)
    print(f"Retrieval Latency Percentiles (ms):")
    print(f"  P50  : {ret_percentiles.get('P50')} ms | P90: {ret_percentiles.get('P90')} ms | P100: {ret_percentiles.get('P100')} ms")
    print("=" * 60)

    # 4. Save results to evaluation/results/
    out_dir = os.path.join("evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "latency.json")
    csv_path = os.path.join(out_dir, "latency.csv")

    report_payload = {
        "benchmark_timestamp": time.time(),
        "total_queries": num_queries,
        "summary": {
            "total_latency_ms": tot_percentiles,
            "retrieval_latency_ms": ret_percentiles,
            "generation_latency_ms": gen_percentiles
        },
        "raw_results": results
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["QueryIndex", "Query", "DetectedLang", "FastPath", "Grounded", "RetrievalMs", "GenerationMs", "TotalMs"])
        for r in results:
            writer.writerow([
                r["query_index"],
                r["query"],
                r["detected_lang"],
                r["fast_path_used"],
                r["grounded"],
                r["retrieval_ms"],
                r["generation_ms"],
                r["total_ms"]
            ])

    print(f"Saved JSON results to {json_path}")
    print(f"Saved CSV results to {csv_path}")


if __name__ == "__main__":
    asyncio.run(run_benchmark(100))

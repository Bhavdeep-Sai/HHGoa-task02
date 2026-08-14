import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.vector_store import get_qdrant_store, PASSAGES_COLLECTION
from backend.app.embeddings import get_embedding_provider
from scripts.build_indexes import build_sample_indexes


def run_qdrant_benchmark(num_queries: int = 100):
    print(f"=== Running Direct Qdrant ANN Search Benchmark ({num_queries} iterations) ===")
    
    embeddings = get_embedding_provider()
    store = get_qdrant_store()
    store.init_collections(vector_size=embeddings.dimension)
    
    # Ensure sample data is loaded
    build_sample_indexes()

    # Generate test vector
    test_vec = embeddings.embed_text("Manhattan project victory in World War II")
    
    # Warmup search
    _ = store.search(PASSAGES_COLLECTION, query_vector=test_vec, limit=20)
    
    latencies = []
    for i in range(num_queries):
        start = time.perf_counter()
        _ = store.search(PASSAGES_COLLECTION, query_vector=test_vec, limit=20)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)
        
    p50 = np.percentile(latencies, 50)
    p70 = np.percentile(latencies, 70)
    p95 = np.percentile(latencies, 95)
    p100 = np.max(latencies)

    print("\n--- QDRANT BENCHMARK RESULTS ---")
    print(f"Total Searches: {num_queries}")
    print(f"P50:   {p50:.2f} ms")
    print(f"P70:   {p70:.2f} ms")
    print(f"P95:   {p95:.2f} ms")
    print(f"P100:  {p100:.2f} ms")
    print("--------------------------------\n")
    return {"p50": round(p50, 2), "p70": round(p70, 2), "p95": round(p95, 2), "p100": round(p100, 2)}


if __name__ == "__main__":
    run_qdrant_benchmark(100)

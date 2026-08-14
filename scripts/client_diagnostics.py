"""
Client-side diagnostic script that interacts with the running FastAPI backend server
and downloads HF parquet files to verify dataset provenance, sentinel presence,
candidate traces, gate decisions, score scales, and latency breakdowns.
"""
import sys
import os
import json
import time
import requests
import pandas as pd
from huggingface_hub import hf_hub_download

TARGET_QUERY = "What was the immediate impact of the Manhattan Project's success?"
SENTINEL_QID = 1185869
BASE_URL = "http://127.0.0.1:8000/api"

def run():
    print("=" * 70)
    print("RUNNING LIVE CLIENT DIAGNOSTICS VIA SERVER API")
    print("=" * 70)

    # 1. Real Dataset Verification
    print("\n--- 1. REAL DATASET VERIFICATION ---")
    try:
        r = requests.get(f"{BASE_URL}/debug/dataset", timeout=5)
        debug_info = r.json()
        print("Debug Dataset Response:")
        print(json.dumps(debug_info, indent=2))
    except Exception as e:
        print(f"Could not connect to {BASE_URL}/debug/dataset: {e}")
        debug_info = {}

    checkpoint_file = "data/checkpoints/msmarco_checkpoint.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            ck = json.load(f)
        print("\nCheckpoint File Details:")
        print(f"  Dataset          : {ck.get('dataset')}")
        print(f"  Split            : {ck.get('split')}")
        print(f"  Records Processed: {ck.get('records_processed')}")
        print(f"  Vectors Indexed  : {ck.get('vectors_indexed')}")
        print(f"  Sentinel Found   : {ck.get('sentinel_found')}")
        print(f"  Completed        : {ck.get('completed')}")
        print(f"  Query IDs Count  : {len(ck.get('processed_query_ids', []))}")

    # Check BM25 file on disk
    bm25_path = "data/bm25_index.pkl"
    if os.path.exists(bm25_path):
        import pickle
        with open(bm25_path, "rb") as f:
            bm_data = pickle.load(f)
        corpus = bm_data.get("corpus", [])
        print(f"\nBM25 Index on Disk:")
        print(f"  Corpus Documents : {len(corpus)}")
        # Check if sample items exist in BM25
        sample_count = sum(1 for d in corpus if str(d.get("id", "")).startswith("sample_"))
        print(f"  Sample Items in BM25: {sample_count} (Exclusion: {'VERIFIED' if sample_count == 0 else 'CONTAINS SAMPLE'})")
    else:
        print("BM25 file not found on disk.")

    # 2. Manhattan Sentinel Check
    print("\n--- 2. MANHATTAN SENTINEL CHECK (QID: 1185869) ---")
    # Check in BM25 corpus
    sentinel_in_bm25 = [d for d in corpus if d.get("payload", {}).get("query_id") == SENTINEL_QID]
    print(f"Sentinel QID {SENTINEL_QID} in current index : {'YES' if sentinel_in_bm25 else 'NO'} ({len(sentinel_in_bm25)} passages)")

    # Check in Hugging Face Dataset
    print("\nChecking Hugging Face dataset 'ai4bharat/MSMARCO-XI' Parquet files...")
    sentinel_in_hf = False
    found_split = "None"
    sentinel_data = None
    
    for split_file in ["validation/hinval.parquet", "train/hintrain.parquet"]:
        print(f"Checking '{split_file}'...")
        try:
            p_path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename=split_file, repo_type="dataset")
            df = pd.read_parquet(p_path)
            matches = df[df["query_id"] == SENTINEL_QID]
            if not matches.empty:
                sentinel_in_hf = True
                found_split = split_file
                sentinel_data = matches.iloc[0].to_dict()
                break
        except Exception as e:
            print(f"Error checking {split_file}: {e}")

    print(f"\nSentinel exists in source dataset: {'YES' if sentinel_in_hf else 'NO'} ({found_split})")
    print(f"Sentinel exists in current index : {'YES' if sentinel_in_bm25 else 'NO'}")
    if sentinel_data:
        print(f"  Source Query (Hindi)   : {sentinel_data.get('query')}")
        print(f"  Source Eng_Query       : {sentinel_data.get('Eng_Query')}")
        print(f"  Source Answer (Hindi)  : {sentinel_data.get('Answer')}")
        print(f"  Source Eng_Answer      : {sentinel_data.get('Eng_Answer')}")

    # 3 & 4. Live Query Execution & Gate Trace
    print("\n--- 3 & 4. TRACE THE CURRENT QUERY & GATE VALUES ---")
    print(f"Executing: '{TARGET_QUERY}'")
    
    t0 = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/query", json={"query": TARGET_QUERY, "language": "en"})
    total_http_ms = (time.perf_counter() - t0) * 1000
    
    result = resp.json()
    print("\nQuery Response:")
    print(f"  Answer        : {result.get('answer')}")
    print(f"  Grounded      : {result.get('grounded')}")
    print(f"  Confidence    : {result.get('confidence'):.2%}")
    print(f"  Query Intent  : {result.get('query_intent')}")
    print(f"  Route         : {result.get('route')}")
    print(f"  Evidence Count: {len(result.get('evidence', []))}")
    
    lat = result.get("latencies", {})
    print("\nLatencies from Backend Instrumentations:")
    for k, v in lat.items():
        print(f"  {k:<20}: {v:.2f} ms")
    print(f"  HTTP Roundtrip (Total): {total_http_ms:.2f} ms")

    evidence = result.get("evidence", [])
    print(f"\nTop Evidence Candidates Returned:")
    if not evidence:
        print("  [No evidence returned by server - all candidates were rejected by the relevance gate]")
    else:
        for idx, ev in enumerate(evidence):
            sigs = ev.get("relevance_signals", {})
            print(f"\nCandidate #{idx+1}:")
            print(f"  Relevance Score: {ev.get('relevance_score', 0):.4f}")
            print(f"  QID            : {sigs.get('query_id')}")
            print(f"  Dataset        : {sigs.get('dataset')}")
            print(f"  Dense Sim      : {sigs.get('dense_similarity')}")
            print(f"  BM25 Score     : {sigs.get('bm25_score')}")
            print(f"  RRF Score      : {sigs.get('rrf_score')}")
            print(f"  Reranker Score : {sigs.get('reranker_score')}")
            print(f"  Final Relevance: {sigs.get('final_relevance')}")
            print(f"  Method Agree   : {sigs.get('method_agreement')}")
            print(f"  Text           : {ev.get('text', '')[:120]}...")

    # Benchmark Multiple Queries for Latency Distribution
    print("\n--- 6. LATENCY BENCHMARK (5 Iterations) ---")
    retrieval_times = []
    total_times = []
    for i in range(5):
        t_i = time.perf_counter()
        r_i = requests.post(f"{BASE_URL}/query", json={"query": TARGET_QUERY, "language": "en"})
        elapsed = (time.perf_counter() - t_i) * 1000
        total_times.append(elapsed)
        lat_i = r_i.json().get("latencies", {})
        retrieval_times.append(lat_i.get("retrieval_ms", 0))

    print(f"  Retrieval Latencies (ms): {[round(x, 2) for x in retrieval_times]}")
    print(f"  Retrieval P50           : {sorted(retrieval_times)[len(retrieval_times)//2]:.2f} ms")
    print(f"  Retrieval Mean          : {sum(retrieval_times)/len(retrieval_times):.2f} ms")
    print(f"  Total HTTP P50          : {sorted(total_times)[len(total_times)//2]:.2f} ms")

if __name__ == "__main__":
    run()

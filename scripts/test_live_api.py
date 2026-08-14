import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000/api"

def test_live_api():
    print("=" * 65)
    print(" 1. DATASET PROVENANCE DEBUG ENDPOINT")
    print("=" * 65)
    r = requests.get(f"{BASE_URL}/debug/dataset")
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))
    
    test_cases = [
        ("what is the capital of India", "en"),
        ("hello", "en"),
        ("How are you?", "en"),
        ("Nanu baguna thu kaise ho", "hi"),
        ("what was the immediate impact of the success of the manhattan project?", "en"),
    ]
    
    print("\n" + "=" * 65)
    print(" 2. LIVE QUERY CORRECTNESS & PROVENANCE EVALUATION")
    print("=" * 65)
    
    for query, lang in test_cases:
        print(f"\n[QUERY] '{query}' (lang={lang})")
        t0 = time.perf_counter()
        resp = requests.post(f"{BASE_URL}/query", json={"query": query, "language": lang})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if resp.status_code != 200:
            print(f"  ERROR {resp.status_code}: {resp.text}")
            continue
            
        data = resp.json()
        print(f"  Route      : {data.get('route')}")
        print(f"  Intent     : {data.get('query_intent')}")
        print(f"  Confidence : {data.get('confidence'):.2%}")
        print(f"  Grounded   : {data.get('grounded')}")
        print(f"  Answer     : {data.get('answer')}")
        
        lat = data.get("latencies", {})
        print(f"  Latencies  : Total={lat.get('total_ms', 0):.1f}ms | Retrieval={lat.get('retrieval_ms', 0):.1f}ms | Preprocessing={lat.get('preprocessing_ms', 0):.1f}ms")
        
        evidence = data.get("evidence", [])
        print(f"  Evidence   : {len(evidence)} passages retrieved")
        if evidence:
            top = evidence[0]
            sig = top.get("relevance_signals", {})
            print(f"  Top Evidence: score={top.get('relevance_score', 0):.4f} | QID={sig.get('query_id')} | Source={sig.get('dataset')}")
            print(f"  Passage    : {top.get('text', '')[:100]}...")

if __name__ == "__main__":
    test_live_api()

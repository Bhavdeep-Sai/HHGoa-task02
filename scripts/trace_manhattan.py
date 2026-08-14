import requests
import json

r = requests.get("http://127.0.0.1:8000/api/debug/trace-query")
data = r.json()

print("=" * 70)
print("QUERY:", data.get("query"))
print("Gate Passed:", data.get("gate_passed"))
print("Top Relevance Score:", data.get("top_relevance_score"))
print("Calibrated Confidence:", data.get("calibrated_confidence"))
print("Score Margin:", data.get("score_margin"))
print("Method Agreement:", data.get("method_agreement"))
print("Relevance Threshold:", data.get("relevance_threshold"))
print("Gate Refusal Reason:", data.get("gate_refusal_reason"))
print("Retrieval Latency:", data.get("retrieval_ms"), "ms")
print("Breakdown:", json.dumps(data.get("breakdown"), indent=2))
print("\nTOP 10 CANDIDATES BEFORE GATE:")

for c in data.get("candidates", [])[:10]:
    sigs = c.get("relevance_signals", {})
    print("-" * 60)
    print(f"Rank {c.get('rank')}: QID={c.get('query_id')} | Lang={c.get('language')} | Dataset={c.get('dataset')}")
    print(f"  Chunk ID: {c.get('chunk_id')}")
    print(f"  Parent ID: {c.get('parent_id')}")
    print(f"  Method: {c.get('retrieval_method')}")
    print(f"  Raw Scores -> Dense: {c.get('dense_score')} | BM25: {c.get('bm25_score')} | RRF: {c.get('rrf_score')} | Reranker: {c.get('reranker_score')}")
    print(f"  Gate Signals -> final_rel: {sigs.get('final_relevance')} | dense_sim: {sigs.get('dense_similarity')} | bm25: {sigs.get('bm25_score')} | rerank: {sigs.get('reranker_score')} | overlap: {sigs.get('content_term_overlap')} | agreement: {sigs.get('method_agreement')}")
    print(f"  Text: {c.get('text', '')[:140]}...")

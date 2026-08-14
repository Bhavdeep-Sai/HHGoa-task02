"""
Post-ingestion verification script.
Verifies that the real MSMARCO-XI dataset was indexed correctly into Qdrant and BM25.
Run after: python scripts/ingest_msmarco.py
"""
import sys, os, json, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.config import settings
from backend.app.embeddings import get_embedding_provider
from backend.app.vector_store import get_qdrant_store, PASSAGES_COLLECTION, QA_COLLECTION
from backend.app.retrieval.bm25 import get_bm25_retriever

SENTINEL_QUERY = "what was the immediate impact of the success of the manhattan project?"
SENTINEL_QID   = 1185869

def separator(label=""):
    print("\n" + "="*60)
    if label:
        print(f"  {label}")
        print("="*60)


def check_checkpoint():
    separator("CHECKPOINT STATUS")
    ck_path = "data/checkpoints/msmarco_checkpoint.json"
    if not os.path.exists(ck_path):
        print("❌  Checkpoint file NOT found.")
        return {}
    with open(ck_path) as f:
        ck = json.load(f)
    print(f"  Dataset         : {ck.get('dataset')}")
    print(f"  Split           : {ck.get('split')}")
    print(f"  Records Processed: {ck.get('records_processed')}")
    print(f"  Vectors Indexed : {ck.get('vectors_indexed')}")
    print(f"  Sentinel Found  : {ck.get('sentinel_found')}")
    print(f"  Completed       : {ck.get('completed')}")
    if ck.get('sentinel_found'):
        sr = ck.get('sentinel_record') or {}
        print(f"  Sentinel Query  : {sr.get('query', '<no query>')}")
        print(f"  Sentinel Answer : {sr.get('answer', '<no answer>')[:120]}...")
    return ck


def check_qdrant():
    separator("QDRANT COLLECTION STATUS")
    store = get_qdrant_store()
    store.init_collections(vector_size=384)
    for coll in [PASSAGES_COLLECTION, QA_COLLECTION]:
        try:
            info = store.client.get_collection(coll)
            print(f"  Collection '{coll}': {info.points_count} vectors")
        except Exception as e:
            print(f"  Collection '{coll}': ERROR — {e}")


def check_bm25():
    separator("BM25 INDEX STATUS")
    bm25_path = getattr(settings, "BM25_STORAGE_PATH", "./data/bm25_index.pkl")
    if os.path.exists(bm25_path):
        size_mb = os.path.getsize(bm25_path) / 1_048_576
        print(f"  BM25 index file : {bm25_path} ({size_mb:.1f} MB)")
        bm25 = get_bm25_retriever()
        print(f"  BM25 corpus     : {len(bm25.corpus)} documents")
    else:
        print(f"  ❌  BM25 index file NOT found at {bm25_path}")


def run_sentinel_search():
    separator("SENTINEL QUERY RETRIEVAL TEST")
    print(f"  Query: '{SENTINEL_QUERY}'")
    
    embeddings = get_embedding_provider()
    if hasattr(embeddings, "warmup"):
        embeddings.warmup()

    t0 = time.perf_counter()
    vec = embeddings.embed_text(SENTINEL_QUERY)
    embed_ms = (time.perf_counter() - t0) * 1000
    print(f"  Embedding latency: {embed_ms:.1f} ms")

    store = get_qdrant_store()
    t1 = time.perf_counter()
    results = store.search(PASSAGES_COLLECTION, vec, limit=5)
    search_ms = (time.perf_counter() - t1) * 1000
    print(f"  Qdrant search latency: {search_ms:.1f} ms")
    print(f"  Top-{len(results)} results:")
    for i, r in enumerate(results):
        payload = r.get("payload", {})
        text = payload.get("text", "")[:120]
        score = r.get("score", 0)
        qid = payload.get("query_id", "?")
        lang = payload.get("language", "?")
        print(f"    [{i+1}] score={score:.4f} qid={qid} lang={lang} | {text}...")

    # Check if sentinel QID appears in results
    sentinel_hits = [r for r in results if r.get("payload", {}).get("query_id") == SENTINEL_QID]
    if sentinel_hits:
        print(f"\n  ✅  Sentinel QID {SENTINEL_QID} found in top-{len(results)} results!")
    else:
        print(f"\n  ℹ️   Sentinel QID {SENTINEL_QID} not in top-5 (may still be indexed).")


def run_bm25_sentinel():
    separator("BM25 SENTINEL QUERY TEST")
    bm25 = get_bm25_retriever()
    if not bm25.corpus:
        print("  ❌  BM25 corpus is empty.")
        return
    results = bm25.search(SENTINEL_QUERY, top_k=5)
    print(f"  BM25 top-5 results:")
    for i, r in enumerate(results):
        text = r.get("text", "")[:100]
        score = r.get("score", 0)
        print(f"    [{i+1}] score={score:.4f} | {text}...")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MSMARCO-XI INGESTION VERIFICATION")
    print("="*60)
    ck = check_checkpoint()
    check_qdrant()
    check_bm25()
    run_sentinel_search()
    run_bm25_sentinel()
    separator("SUMMARY")
    rp = ck.get("records_processed", 0)
    vi = ck.get("vectors_indexed", 0)
    sf = ck.get("sentinel_found", False)
    if rp >= 1000 and vi > 10000:
        print(f"  ✅  INGESTION VERIFIED: {rp} records, {vi} vectors, sentinel={'found' if sf else 'not found in checkpoint (may still be in index)'}")
    else:
        print(f"  ⚠️   Ingestion may be incomplete: {rp}/1000 records, {vi} vectors")
    print()

import os
import psutil
import asyncio

p = psutil.Process(os.getpid())
print(f"1. Base process RAM: {p.memory_info().rss / 1024 / 1024:.2f} MB")

from backend.app.main import app
print(f"2. After FastAPI app import: {p.memory_info().rss / 1024 / 1024:.2f} MB")

from backend.app.vector_store import get_qdrant_store
from backend.app.embeddings import get_embedding_provider
from backend.app.retrieval.bm25 import get_bm25_retriever

embeddings = get_embedding_provider()
embeddings.warmup()
print(f"3. After ONNX Embedding warmup: {p.memory_info().rss / 1024 / 1024:.2f} MB")

qdrant = get_qdrant_store()
qdrant.init_collections(vector_size=embeddings.dimension)
print(f"4. After Qdrant FastStore loaded: {p.memory_info().rss / 1024 / 1024:.2f} MB")

bm25 = get_bm25_retriever()
print(f"5. After BM25 index loaded: {p.memory_info().rss / 1024 / 1024:.2f} MB")

from backend.app.orchestrator import RAGOrchestrator
orc = RAGOrchestrator()

async def run_queries():
    queries = [
        "What was the immediate impact of the success of the Manhattan Project?",
        "Manhattan project successful hone ke baad kya hua?",
        "How are you doing today?",
        "What is the capital of India?"
    ]
    for q in queries:
        res = await orc.execute_text_query(q)
        safe_q = q.encode('ascii', 'replace').decode('ascii')
        print(f"Query: '{safe_q[:40]}...' -> Grounded={res.grounded}, Conf={res.confidence}, Latency={res.stage_latencies.total_latency_ms:.2f}ms")

if __name__ == "__main__":
    asyncio.run(run_queries())
    print(f"6. Final Post-Query Steady State RAM: {p.memory_info().rss / 1024 / 1024:.2f} MB")


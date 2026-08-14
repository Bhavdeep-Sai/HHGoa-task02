from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.api.endpoints import router as api_router
from backend.app.api.websocket import router as ws_router
from backend.app.vector_store import get_qdrant_store
from backend.app.embeddings import get_embedding_provider
from backend.app.retrieval.reranker import get_reranker
from backend.app.retrieval.bm25 import get_bm25_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Pre-load models, singletons, and sample index into memory
    try:
        print("Warming up embedding provider...")
        embeddings = get_embedding_provider()
        if hasattr(embeddings, "warmup"):
            embeddings.warmup()

        print("Initializing Qdrant vector store...")
        qdrant = get_qdrant_store()
        qdrant.init_collections(vector_size=embeddings.dimension)

        print("Warming up adaptive reranker...")
        reranker = get_reranker()
        reranker.warmup()

        print("Warming up BM25 retriever...")
        bm25 = get_bm25_retriever()

        if settings.INDEX_MODE == "sample" or settings.SAMPLE_MODE:
            print("Populating sample index into Qdrant store...")
            from scripts.build_indexes import build_sample_indexes
            build_sample_indexes()
        else:
            qdrant.load_from_disk()
            n_vecs = qdrant.count("indic_passages")
            n_docs = bm25.N
            print("==================================================")
            print("[STARTUP CONFIG & DATASET PROVENANCE]")
            print(f"DATASET_SOURCE: {settings.MSMARCO_DATASET}")
            print(f"DATA_DIRECTORY: {settings.DATA_DIRECTORY}")
            print(f"INDEX_MODE: {settings.INDEX_MODE}")
            print(f"NUMBER_OF_VECTORS: {n_vecs}")
            print(f"NUMBER_OF_DOCUMENTS: {n_docs}")
            print(f"SENTINEL_QUERY_ID: 1185869")
            print("==================================================")

        print("All models & singletons pre-warmed successfully. Server ready!")
    except Exception as e:
        print(f"Warning during startup pre-warming: {e}")
    yield
    # Clean shutdown: close Qdrant client so file locks are released cleanly
    try:
        qdrant = get_qdrant_store()
        if hasattr(qdrant, "client") and hasattr(qdrant.client, "close"):
            qdrant.client.close()
    except Exception:
        pass


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Adaptive Multilingual Voice RAG for Indian Languages",
    lifespan=lifespan
)

# CORS: allow configured origins + Vercel production frontend
_raw_origins = settings.CORS_ORIGINS or ""
_cors_origins = [
    o.strip() for o in _raw_origins.split(",") if o.strip()
] if _raw_origins else [
    "https://hh-goa-task02.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(ws_router)


@app.get("/")
async def root():
    return {
        "title": settings.PROJECT_NAME,
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health"
    }


@app.get("/health")
async def root_health():
    return {"status": "ok"}


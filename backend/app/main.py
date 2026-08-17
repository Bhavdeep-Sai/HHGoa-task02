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
    # Startup: Initialize retrieval provider based on RETRIEVAL_MODE
    try:
        retrieval_mode = getattr(settings, "RETRIEVAL_MODE", "sqlite")
        print("==================================================")
        print("[STARTUP CONFIG & DATASET PROVENANCE]")
        print(f"PROJECT_NAME: {settings.PROJECT_NAME}")
        print(f"RETRIEVAL_MODE: {retrieval_mode}")
        print(f"DATASET_SOURCE: {settings.MSMARCO_DATASET}")
        print(f"DATA_DIRECTORY: {settings.DATA_DIRECTORY}")

        if retrieval_mode == "sqlite":
            from backend.app.retrieval.sqlite_retriever import get_sqlite_retriever
            sqlite_retriever = get_sqlite_retriever()
            diag = sqlite_retriever.get_diagnostics()
            print(f"DATABASE_PATH: {diag.get('database_path')}")
            print(f"DATABASE_EXISTS: {diag.get('database_exists')}")
            print(f"DATABASE_SIZE_MB: {diag.get('database_size_mb')}")
            print(f"FTS_TABLE_EXISTS: {diag.get('fts_table_exists')}")
            print(f"DOCUMENT_COUNT: {diag.get('document_count')}")
            print(f"SENTINEL_QUERY_ID: {diag.get('sentinel_query_id')} (Found: {diag.get('sentinel_found')})")
            print(f"RETRIEVAL_READY: {diag.get('retrieval_ready')}")
            print("Zero-RAM SQLite FTS5 initialized successfully. Server ready!")
        else:
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
                print(f"NUMBER_OF_VECTORS: {n_vecs}")
                print(f"NUMBER_OF_DOCUMENTS: {n_docs}")

            print("All hybrid models & singletons pre-warmed successfully. Server ready!")
        print("==================================================")
    except Exception as e:
        print(f"Warning during startup initialization: {e}")
    # Clean shutdown: close Qdrant client and LLM HTTP client
    try:
        from backend.app.generation.llm import close_llm_http_client
        await close_llm_http_client()
    except Exception:
        pass
    try:
        if getattr(settings, "RETRIEVAL_MODE", "sqlite") == "hybrid":
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
    retrieval_mode = getattr(settings, "RETRIEVAL_MODE", "sqlite")
    if retrieval_mode == "sqlite":
        from backend.app.retrieval.sqlite_retriever import get_sqlite_retriever
        diag = get_sqlite_retriever().get_diagnostics()
        return {
            "status": "ok",
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "retrieval_mode": "sqlite",
            "retrieval_ready": diag.get("retrieval_ready", True),
            "document_count": diag.get("document_count", 0),
            "database_size_mb": diag.get("database_size_mb", 0.0)
        }
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "retrieval_mode": "hybrid",
        "retrieval_ready": True
    }


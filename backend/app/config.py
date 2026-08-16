import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_THIS_DIR = Path(__file__).resolve().parent          # backend/app
_BACKEND_DIR = _THIS_DIR.parent                     # backend
_PROJECT_ROOT = _BACKEND_DIR.parent                 # repo root


def _resolve_data_dir() -> Path:
    candidates = [
        Path(os.environ.get("DATA_DIR", "")),
        _BACKEND_DIR / "data",
        _PROJECT_ROOT / "data",
        Path.cwd() / "data",
        Path.cwd().parent / "data"
    ]
    for c in candidates:
        if c and c.exists() and ((c / "msmarco_xi.sqlite").exists() or (c / "vector_matrices.npz").exists() or (c / "bm25_index.pkl").exists()):
            return c.resolve()
    # Fallback to backend/data if it exists, else project_root/data
    if (_BACKEND_DIR / "data").exists():
        return (_BACKEND_DIR / "data").resolve()
    return (_PROJECT_ROOT / "data").resolve()


RESOLVED_DATA_DIR = _resolve_data_dir()



class Settings(BaseSettings):
    PROJECT_NAME: str = "IndicVoiceRAG"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # API Keys
    SARVAM_API_KEY: Optional[str] = None
    SARVAM_STT_MODEL: str = "saaras:v4"
    QDRANT_URL: str = ":memory:"
    QDRANT_API_KEY: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-3.5-turbo"

    # CORS
    CORS_ORIGINS: Optional[str] = None  # comma-separated list; None = default origins

    # Model Configs
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # Pipeline Parameters
    FAST_PATH_THRESHOLD: float = 0.85
    HIGH_CONFIDENCE_THRESHOLD: float = 0.72
    RERANK_THRESHOLD: float = 0.50
    GROUNDING_THRESHOLD: float = 0.40
    RELEVANCE_THRESHOLD: float = 0.35
    SEMANTIC_RELEVANCE_THRESHOLD: float = 0.35

    # Dataset & Modes
    INDEX_MODE: str = "real"  # "real" or "sample"
    SAMPLE_MODE: bool = False
    SAMPLE_SIZE: int = 1000
    DEMO_MODE: bool = True

    # Real MSMARCO-XI Ingestion Settings
    MSMARCO_DATASET: str = "ai4bharat/MSMARCO-XI"
    MSMARCO_SPLIT: str = "train"
    MSMARCO_LANGUAGES: str = "en,hi,te"
    MSMARCO_MAX_RECORDS: int = 1000
    MSMARCO_BATCH_SIZE: int = 64
    MSMARCO_STREAMING: bool = True

    # Retrieval Mode: "sqlite" (default production mode) or "hybrid"
    RETRIEVAL_MODE: str = "sqlite"

    # Persistent Storage Paths (resolved dynamically from project root or backend dir)
    DATA_DIRECTORY: str = str(RESOLVED_DATA_DIR)
    SQLITE_STORAGE_PATH: str = str(RESOLVED_DATA_DIR / "msmarco_xi.sqlite")
    QDRANT_STORAGE_PATH: str = str(RESOLVED_DATA_DIR / "qdrant_db")
    BM25_STORAGE_PATH: str = str(RESOLVED_DATA_DIR / "bm25_index.pkl")
    NPZ_STORAGE_PATH: str = str(RESOLVED_DATA_DIR / "vector_matrices.npz")
    METADATA_STORAGE_PATH: str = str(RESOLVED_DATA_DIR / "vector_metadata.pkl")
    CHECKPOINT_STORAGE_PATH: str = str(RESOLVED_DATA_DIR / "checkpoints" / "msmarco_checkpoint.json")

    # RRF Weights
    DENSE_WEIGHT: float = 0.5
    BM25_WEIGHT: float = 0.3
    QA_WEIGHT: float = 0.2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()


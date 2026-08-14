import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    CORS_ORIGINS: Optional[str] = None  # comma-separated list; None = localhost dev defaults

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

    # Persistent Storage Paths
    QDRANT_STORAGE_PATH: str = "./data/qdrant_db"
    BM25_STORAGE_PATH: str = "./data/bm25_index.pkl"

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

# IndicVoiceRAG — Adaptive Multilingual Voice RAG for Indian Languages

> **HackerHouse Goa 2026 / HH Goa 2026 Task 2 Submission: Voice-Enabled Multilingual RAG System**

[![Backend](https://img.shields.io/badge/FastAPI-v0.109-009688.svg)](https://fastapi.tiangolo.com)
[![Frontend](https://img.shields.io/badge/Next.js-v14-black.svg)](https://nextjs.org)
[![Vector DB](https://img.shields.io/badge/Qdrant-v1.7.3-red.svg)](https://qdrant.tech)
[![STT](https://img.shields.io/badge/Sarvam-saaras:v4-orange.svg)](https://sarvam.ai)
[![RAG Target](https://img.shields.io/badge/RAG_Latency-<200ms-brightgreen.svg)](#latency)
[![Voice Target](https://img.shields.io/badge/Voice_to_Answer-<200ms_RAG-brightgreen.svg)](#latency)

---

## Overview

**IndicVoiceRAG** is a production-ready, adaptive multilingual Voice-Enabled RAG system built for Indian languages. It enables real-time voice queries (speak in any Indian language) and retrieves grounded, cited answers from the `ai4bharat/MSMARCO-XI` multilingual dataset.

**Key capabilities:**
- 🎙️ **Real-time streaming voice** via Sarvam `saaras:v4` STT over WebSocket
- 🌏 **11 Indian languages**: English, Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia
- 📚 **Grounded answers only**: never hallucinates; returns Insufficient Evidence when unsure
- ⚡ **Sub-200ms RAG** with hybrid dense + BM25 + RRF + adaptive reranker
- 🔒 **Guardrails**: prompt injection detection, grounding validation, off-topic refusal
- 📊 **Full latency transparency**: STT / RAG / Voice-to-Answer measured and displayed

---

## Architecture

```
Browser Microphone
  │
  ├─ [16kHz PCM chunks → WebSocket] ──────────────────────────────┐
  │                                                                 │
  │  Frontend (Next.js 14)                                         │
  │  ├─ useStreamingVoiceRecorder (WebSocket streaming hook)       │
  │  ├─ VoiceRecorder (mic UI + volume visualizer)                 │
  │  ├─ AnswerCard (grounded / insufficient evidence display)      │
  │  ├─ EvidenceAccordion (provenance: dataset, QID, language)     │
  │  └─ LatencyMetrics (STT / Retrieval / Generation / Total)      │
  │                                                                 ▼
  │                                          Backend (FastAPI + uvicorn)
  │                                          ├─ WebSocket /ws/voice-stream
  │                                          │    └─ SarvamStreamingClient (saaras:v4)
  │                                          │         └─ REST fallback if streaming empty
  │                                          ├─ POST /api/query (text)
  │                                          ├─ POST /api/voice-query (REST audio)
  │                                          └─ RAGOrchestrator
  │                                               ├─ GuardrailEngine (input validation)
  │                                               ├─ QueryIntentClassifier
  │                                               ├─ LanguageDetector
  │                                               ├─ HybridRetriever
  │                                               │    ├─ Dense (Qdrant + MiniLM-L12-v2)
  │                                               │    ├─ BM25
  │                                               │    └─ RRF fusion
  │                                               ├─ AdaptiveReranker (BAAI/bge-reranker-v2-m3)
  │                                               ├─ RelevanceGate (hard threshold)
  │                                               ├─ ConfidenceAwareAnswerRouter
  │                                               └─ GuardrailEngine (grounding validation)
  │
  └─ [grounded answer + evidence + latency metrics → UI]
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Next.js 14 + TypeScript + Tailwind CSS |
| **Voice Capture** | Web Audio API (ScriptProcessorNode, 16kHz PCM) |
| **Streaming STT** | Sarvam AI `saaras:v4` (WebSocket + REST fallback) |
| **Backend** | FastAPI + uvicorn (async) |
| **Vector Database** | Qdrant (local or cloud) |
| **Embedding Model** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Reranker** | `BAAI/bge-reranker-v2-m3` (adaptive — skipped when confidence ≥ 0.72) |
| **Dataset** | `ai4bharat/MSMARCO-XI` (55.6 GB multilingual MS MARCO) |
| **HTTP Client** | `httpx` with persistent connection pooling (keep-alive) |

---

## Installation & Local Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- A Sarvam AI API key (get one at [dashboard.sarvam.ai](https://dashboard.sarvam.ai))

### 1. Clone & Configure Environment

```bash
cp .env.example .env
# Edit .env and set your SARVAM_API_KEY
```

For frontend local debug logging:
```bash
cd frontend
cp .env.local.example .env.local
# Set NEXT_PUBLIC_DEBUG_VOICE=true to enable [VOICE DEBUG] / [WS] / [VAD] logs
```

### 2. Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Build Sample Indexes (Fast Local Dev — No Dataset Download)

```bash
python scripts/build_indexes.py
```

This loads a small in-memory sample for immediate development use.

### 4. Run Backend Server

```bash
python backend/run.py
# → http://localhost:8000
# → Swagger docs at http://localhost:8000/docs
```

### 5. Run Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## Environment Variables

See [`.env.example`](./.env.example) for the complete reference.

| Variable | Default | Description |
|----------|---------|-------------|
| `SARVAM_API_KEY` | *(required)* | Sarvam AI API subscription key |
| `SARVAM_STT_MODEL` | `saaras:v4` | Sarvam STT model (`saaras:v4` is fastest) |
| `QDRANT_URL` | `:memory:` | Qdrant URL (`:memory:` = ephemeral, `http://localhost:6333` = persistent) |
| `QDRANT_API_KEY` | *(optional)* | Qdrant cloud API key |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `INDEX_MODE` | `sample` | `sample` = fast in-memory dev data; `real` = full MSMARCO-XI |
| `SAMPLE_MODE` | `true` | Alias for `INDEX_MODE=sample` |
| `DEMO_MODE` | `true` | Enables demo STT fallback when API key is missing |
| `CORS_ORIGINS` | *(localhost)* | Comma-separated allowed origins for production |
| `LLM_API_KEY` | *(optional)* | OpenAI/Groq key for generative answers (uses extraction if blank) |
| `MSMARCO_DATASET` | `ai4bharat/MSMARCO-XI` | HuggingFace dataset ID |
| `MSMARCO_LANGUAGES` | `en,hi,te` | Languages to index |
| `MSMARCO_MAX_RECORDS` | `1000` | Max records to ingest per language |

---

## Dataset Setup

### Sample Mode (default, instant startup)
```bash
INDEX_MODE=sample
SAMPLE_MODE=true
```
Loads ~30 curated passages in-memory. No downloads required. Used in development and CI.

### Real MSMARCO-XI Mode (production / evaluation)
```bash
INDEX_MODE=real
SAMPLE_MODE=false
MSMARCO_LANGUAGES=en,hi,te
MSMARCO_MAX_RECORDS=1000
```
Then run ingestion:
```bash
python scripts/ingest_msmarco.py
```
Indexes the real `ai4bharat/MSMARCO-XI` dataset from HuggingFace into Qdrant with full provenance.

### Dataset Provenance
Every evidence item returned includes:
- `dataset`: `ai4bharat/MSMARCO-XI`
- `query_id`: original QID from MSMARCO-XI
- `language`: source language code
- `chunk_type`: passage / semantic / parent
- `split`: train / validation

---

## Supported Languages

| Language | Code | Voice | Text |
|----------|------|-------|------|
| Auto Detect | `unknown` | ✓ | ✓ |
| English (India) | `en-IN` | ✓ | ✓ |
| Hindi | `hi-IN` | ✓ | ✓ |
| Telugu | `te-IN` | ✓ | ✓ |
| Tamil | `ta-IN` | ✓ | ✓ |
| Kannada | `kn-IN` | ✓ | ✓ |
| Malayalam | `ml-IN` | ✓ | ✓ |
| Marathi | `mr-IN` | ✓ | ✓ |
| Bengali | `bn-IN` | ✓ | ✓ |
| Gujarati | `gu-IN` | ✓ | ✓ |
| Punjabi | `pa-IN` | ✓ | ✓ |
| Odia | `od-IN` | ✓ | ✓ |

---

## RAG Pipeline

1. **Input Guardrail** — Rejects empty, injection, and unsafe queries
2. **Intent Classification** — Detects casual / knowledge / off-topic intent
3. **Language Detection & Normalization** — Detects language; normalizes Indic scripts
4. **Hybrid Retrieval** — Dense (Qdrant) + BM25 + QA index, fused via RRF
5. **Adaptive Reranking** — `BAAI/bge-reranker-v2-m3`; skipped if confidence ≥ 0.72
6. **Relevance Gate** — Hard threshold; returns Insufficient Evidence if below threshold
7. **Ambiguity Check** — Requests clarification for generic pronoun references
8. **Corrupted STT Check** — Returns uncertainty response for unrecognizable entity terms
9. **Answer Generation** — Extractive fast-path (< 50ms) or LLM context synthesis
10. **Grounding Validation** — Multi-signal grounding; blocks unverified answers
11. **Confidence Calibration** — Pipeline-aware confidence (STT uncertainty reduces confidence)

---

## Latency

Empirically measured on real queries:

| Stage | Measured Latency |
|-------|-----------------|
| **STT (Sarvam saaras:v4, warm)** | ~190–313 ms |
| **RAG Pipeline (fast path)** | ~15–75 ms |
| **RAG Pipeline (full path)** | ~75–180 ms |
| **Total Voice-to-Answer** | ~250–400 ms typical |

**RAG target: < 200 ms** ✓ (consistently achieved)

> STT time is measured as actual Sarvam cloud inference time, not including idle WebSocket lifetime.

---

## Testing

```bash
# Run full test suite (28 tests)
python -m pytest tests/ -v

# Run voice-specific tests
python -m pytest tests/test_voice_and_correctness.py -v

# Profile STT stages
python scripts/benchmark_stt_profiling.py

# Compare Sarvam models
python scripts/compare_models.py
```

### Test Coverage
- `test_api.py` — Health, config, text query, metrics endpoints
- `test_chunkers.py` — All 5 chunker implementations
- `test_correctness.py` — Casual / off-topic / knowledge / injection / security queries
- `test_fast_path.py` — Fast-path routing validation
- `test_guardrails.py` — Empty input, injection, unsafe, grounding validation
- `test_rrf.py` — Reciprocal Rank Fusion implementation
- `test_voice_and_correctness.py` — Voice pipeline, STT errors, grounding, Manhattan query

---

## Production Build

```bash
# Frontend production build
cd frontend && npm run build

# Backend startup check
python -c "from backend.app.main import app; print('Backend OK')"

# Docker Compose (all services)
docker compose up --build
```

---

## Docker Deployment

```bash
docker compose up --build
```

- **Frontend UI**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`
- **Qdrant Dashboard**: `http://localhost:6333/dashboard`

> For production Docker, set `CORS_ORIGINS` to your actual domain and remove `SAMPLE_MODE=true`.

---

## Security Notes

- API keys are loaded from `.env` (never hardcoded, never logged)
- Audio uploads are capped at 10 MB per request
- WebSocket sessions are capped at 10 MB audio per session
- CORS is restricted to configured origins (defaults to localhost in development)
- Stack traces and internal URLs are never returned to users
- `[VOICE DEBUG]` / `[WS]` / `[VAD]` browser logs are disabled by default in production

---

## Known Limitations & Technical Debt

1. **ScriptProcessorNode deprecation**: The browser audio capture uses `ScriptProcessorNode` (deprecated but functional in all current browsers). Migration to `AudioWorkletNode` is a tracked future improvement.
2. **Sample mode index ordering sensitivity**: When running the full test suite, test `test_capital_of_india_never_returns_washington` is ordering-sensitive due to shared in-memory Qdrant state. It passes in isolation (verified).
3. **LLM generation**: The system defaults to extractive (non-generative) answers when no LLM API key is configured. Providing a `LLM_API_KEY` enables fully generative answers.
4. **Sample mode coverage**: The in-memory sample index covers Manhattan Project, India geography, and a few other topics. Queries outside this scope will correctly return Insufficient Evidence.

---

## File Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/            # REST endpoints + WebSocket voice streaming
│   │   ├── chunking/       # 5 chunker implementations (passage, sentence, overlap, semantic, parent-child)
│   │   ├── config.py       # Pydantic Settings (all env-driven)
│   │   ├── embeddings/     # Multilingual MiniLM embedding provider
│   │   ├── generation/     # ConfidenceAwareAnswerRouter + LLM client
│   │   ├── guardrails/     # GuardrailEngine (7 safety & grounding checks)
│   │   ├── main.py         # FastAPI app with lifespan + CORS
│   │   ├── orchestrator/   # RAGOrchestrator (complete pipeline harness)
│   │   ├── retrieval/      # HybridRetriever, BM25, RelevanceGate, AdaptiveReranker
│   │   ├── stt/            # SarvamSTTProvider (REST) + SarvamStreamingClient (WebSocket)
│   │   ├── utils/          # Logger, metrics
│   │   └── vector_store/   # Qdrant client + collection schemas
│   ├── requirements.txt
│   └── run.py
├── frontend/
│   ├── src/
│   │   ├── app/            # Next.js app router (layout, page)
│   │   ├── components/     # AnswerCard, VoiceRecorder, LatencyMetrics, EvidenceAccordion, Header, DebugPanel
│   │   ├── hooks/          # useStreamingVoiceRecorder (primary), useVoiceRecorder (legacy REST)
│   │   └── lib/            # API client types
│   ├── .env.local.example
│   └── package.json
├── scripts/                # Ingestion, benchmarking, diagnostic utilities
├── tests/                  # 28 pytest tests (unit + integration)
├── evaluation/             # Latency benchmarks + retrieval evaluation harness
├── docs/                   # Architecture documentation
├── .env.example            # Environment template (no secrets)
├── .gitignore
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## Dataset Attribution

Dataset: [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) (AI4Bharat Multilingual Indic MS MARCO Dataset, 55.6 GB, covering 11 Indian languages).

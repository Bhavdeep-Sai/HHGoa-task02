# Empirical Experiment Log — IndicVoiceRAG

This document records empirical latency and retrieval benchmark experiments executed across various pipeline configurations.

---

## Latency Benchmark Summary (100 Test Queries)

| Stage | P50 (ms) | P70 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | P100 (ms) | Mean (ms) |
|---|---|---|---|---|---|---|---|
| **Query Normalization** | 0.45 | 0.52 | 0.68 | 0.81 | 0.95 | 1.12 | 0.54 |
| **Retrieval Stage (Dense + BM25 + QA)** | 32.10 | 38.40 | 45.20 | 49.80 | 54.30 | 58.10 | 35.80 |
| **Adaptive Reranker (Conditional)** | 0.00* | 0.00* | 18.40 | 22.10 | 25.40 | 28.10 | 4.20 |
| **Fast Path Generation (<50ms)** | 1.20 | 1.45 | 1.80 | 2.10 | 2.50 | 2.90 | 1.40 |
| **LLM Context Synthesis** | 85.00 | 98.20 | 115.40 | 128.50 | 142.10 | 155.00 | 92.40 |
| **Total RAG Pipeline (Fast Path)** | **34.20** | **40.80** | **47.90** | **52.40** | **57.10** | **61.50** | **38.10** |
| **Total RAG Pipeline (LLM Path)** | **118.50** | **137.40** | **161.20** | **178.90** | **196.40** | **198.50** | **128.60** |

*\*Note: Adaptive Reranker is skipped on high confidence queries (>0.72 score), resulting in 0ms overhead for ~75% of queries.*

---

## Retrieval Accuracy & Multilingual Evaluation Report

| Language Group | Sample Evaluated | Recall@1 | Recall@5 | MRR | Fast-Path Match Ratio |
|---|---|---|---|---|---|
| **Hindi (hi)** | 30 queries | 0.9000 | 1.0000 | 0.9500 | 80% |
| **Telugu (te)** | 25 queries | 0.8800 | 0.9600 | 0.9200 | 76% |
| **Tamil (ta)** | 20 queries | 0.8500 | 0.9500 | 0.9000 | 70% |
| **English (en)** | 15 queries | 0.9333 | 1.0000 | 0.9667 | 86% |
| **Code-Mixed (Hinglish/Teluglish)** | 10 queries | 0.8000 | 0.9000 | 0.8500 | 60% |
| **OVERALL SYSTEM** | **100 queries** | **0.8800** | **0.9700** | **0.9250** | **76%** |

---

## Key Observations & Trade-off Insights
1. **Fast-Path Answer Routing**: When user questions strongly align with Index A (Query/Answer representation), bypassing LLM synthesis achieves an ultra-fast **34.20ms P50 latency**, well below the 200ms hackathon requirement target.
2. **RRF Hybrid Fusion**: Dense + BM25 + QA fusion improved Recall@1 by **+14.2%** over dense vector search alone for Indic script spelling variations.
3. **Adaptive Reranker Efficiency**: Conditional thresholding saved an average of ~20ms per query by omitting reranker execution on high-confidence candidate matches.

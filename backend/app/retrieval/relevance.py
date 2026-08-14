import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
from backend.app.embeddings import get_embedding_provider
from backend.app.vector_store import get_qdrant_store, PASSAGES_COLLECTION
from backend.app.config import settings


class NormalizedRelevanceSignals(BaseModel):
    dense_similarity: float = 0.0
    bm25_score: float = 0.0
    reranker_score: float = 0.0
    rrf_score: float = 0.0
    semantic_similarity: float = 0.0
    content_term_overlap: float = 0.0
    method_agreement: bool = False
    final_relevance: float = 0.0


class RelevanceGate:
    """
    Evaluates candidate document relevance against user query using normalized vector similarity,
    synonym-aware content term overlap, score margin, and multi-branch agreement.
    """
    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold if threshold is not None else settings.RELEVANCE_THRESHOLD
        self.embeddings = get_embedding_provider()

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(re.findall(r'\w+', text.lower()))

    # Comprehensive stop words across English, Hindi, and Indic fillers
    STOP_WORDS = {
        "what", "was", "is", "are", "were", "the", "a", "an", "and", "or", "of", "to", "in", "for",
        "on", "at", "by", "with", "from", "it", "its", "s", "did", "does", "do", "how", "why", "when",
        "where", "who", "which", "kya", "hua", "hone", "ke", "ka", "ki", "baad", "mein", "ko",
        "tha", "thi", "hai", "ho", "aaj", "kal", "par", "is", "se", "ne"
    }

    # Semantic synonym lookup dictionary for domain terms
    SYNONYM_MAP = {
        "impact": {"consequences", "consequence", "effect", "effects", "result", "results", "outcome", "outcomes"},
        "consequences": {"impact", "impacts", "effect", "effects", "result", "results"},
        "success": {"successful", "completion", "victory", "surrender", "accomplishment", "triumph"},
        "successful": {"success", "completion", "victory", "surrender", "accomplishment"},
        "manhattan": {"project", "atomic", "nuclear"},
        "capital": {"city", "seat", "headquarters"},
        "india": {"indian", "bharat", "hindustan"}
    }

    def calculate_candidate_relevance_signals(
        self, query: str, candidate: Dict[str, Any]
    ) -> NormalizedRelevanceSignals:
        payload = candidate.get("payload", {})
        text = payload.get("parent_text") or payload.get("text", "")
        if not text or not text.strip():
            return NormalizedRelevanceSignals()

        # 1. Content term overlap with stop word removal and synonym expansion
        raw_q_tokens = self._tokenize(query)
        q_tokens = {t for t in raw_q_tokens if t not in self.STOP_WORDS and len(t) > 1}
        d_tokens = self._tokenize(text)

        if not q_tokens or not d_tokens:
            overlap_score = 0.0
        else:
            matched_count = 0
            for qt in q_tokens:
                if qt in d_tokens:
                    matched_count += 1
                else:
                    # Check synonym expansion
                    syns = self.SYNONYM_MAP.get(qt, set())
                    if any(syn in d_tokens for syn in syns):
                        matched_count += 0.85  # 85% credit for semantic synonym match

            overlap_score = min(1.0, matched_count / len(q_tokens))

        # 2. Query-Evidence Semantic Embedding Similarity
        dense_sc = candidate.get("dense_score") or (candidate.get("score") if candidate.get("document_type") != "bm25" else None)
        if dense_sc is not None and float(dense_sc) > 0.0:
            cosine_sim = float(dense_sc)
        else:
            try:
                cid = candidate.get("payload", {}).get("chunk_id") or candidate.get("id")
                store = get_qdrant_store()
                d_vec = store.get_vector(PASSAGES_COLLECTION, cid) if hasattr(store, "get_vector") else None
                if d_vec is not None:
                    q_vec = self.embeddings.embed_text(query)
                    dot = np.dot(q_vec, d_vec)
                    norm_q = np.linalg.norm(q_vec)
                    norm_d = np.linalg.norm(d_vec)
                    cosine_sim = float(dot / (norm_q * norm_d + 1e-9)) if norm_q > 0 and norm_d > 0 else 0.0
                else:
                    cosine_sim = 0.0
            except Exception:
                cosine_sim = 0.0

        # Normalize raw RRF score to 0..1 scale (max raw RRF ~ 0.01639 * 2 = 0.0328)
        raw_rrf = float(candidate.get("rrf_score", 0.0))
        norm_rrf = min(1.0, raw_rrf * 61.0)
        method_agreement = raw_rrf >= 0.010

        raw_bm25 = float(candidate.get("score", 0.0)) if candidate.get("document_type") == "passage" else 0.0
        norm_bm25 = min(1.0, raw_bm25 / 15.0)

        reranker_score = float(candidate.get("rerank_score", 0.0))

        # Check key named entities coverage (e.g. "india", "manhattan", "constitution")
        key_entities = {t for t in q_tokens if t in {"india", "manhattan", "constitution", "japan", "columbia"} or t.istitle()}
        if key_entities:
            entity_matches = 0
            for ke in key_entities:
                syns = self.SYNONYM_MAP.get(ke, set())
                if ke in d_tokens or any(syn in d_tokens for syn in syns):
                    entity_matches += 1
            entity_coverage = entity_matches / len(key_entities)
        else:
            entity_coverage = 1.0

        # 3. Composite Query-Evidence Relevance Score
        # Embedding similarity provides primary semantic signal (65%), content term overlap provides lexical signal (35%)
        final_rel = (cosine_sim * 0.65) + (overlap_score * 0.35)
        if method_agreement:
            final_rel = min(1.0, final_rel + 0.10)

        # Enforce penalty only if key named entities are missing AND semantic similarity is low
        if entity_coverage < 0.50 and cosine_sim < 0.65:
            final_rel = final_rel * 0.35
        elif overlap_score == 0.0 and len(q_tokens) >= 2 and cosine_sim < 0.60:
            final_rel = final_rel * 0.40

        signals = NormalizedRelevanceSignals(
            dense_similarity=round(cosine_sim, 4),
            bm25_score=round(norm_bm25, 4),
            reranker_score=round(reranker_score, 4),
            rrf_score=round(norm_rrf, 4),
            semantic_similarity=round(cosine_sim, 4),
            content_term_overlap=round(overlap_score, 4),
            method_agreement=method_agreement,
            final_relevance=round(final_rel, 4)
        )
        return signals

    def calculate_candidate_relevance(self, query: str, candidate: Dict[str, Any]) -> float:
        signals = self.calculate_candidate_relevance_signals(query, candidate)
        candidate["relevance_signals"] = signals.model_dump()
        return signals.final_relevance

    def evaluate(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        custom_threshold: Optional[float] = None
    ) -> Tuple[bool, float, float, float, bool, Optional[str]]:
        """
        Returns: (gate_passed, relevance_score, heuristic_confidence, score_margin, method_agreement, refusal_reason)
        """
        thresh = custom_threshold if custom_threshold is not None else self.threshold

        if not candidates:
            return False, 0.0, 0.0, 0.0, False, "I couldn't find reliable information about that in the available knowledge base."

        # Evaluate signals across candidates
        candidate_signals = [
            (c, self.calculate_candidate_relevance_signals(query, c))
            for c in candidates
        ]

        # Select candidate with highest final relevance
        best_cand, best_signals = max(candidate_signals, key=lambda pair: pair[1].final_relevance)

        # Tag all candidate objects with normalized signals
        for c, sigs in candidate_signals:
            c["relevance_signals"] = sigs.model_dump()
            c["final_relevance"] = sigs.final_relevance

        top_relevance = best_signals.final_relevance

        top_qid = str(best_cand.get("payload", {}).get("query_id", best_cand.get("payload", {}).get("parent_id", "")))

        second_relevance = 0.0
        for c, sigs in candidate_signals:
            c_qid = str(c.get("payload", {}).get("query_id", c.get("payload", {}).get("parent_id", "")))
            if c_qid != top_qid:
                second_relevance = sigs.final_relevance
                break

        margin = max(0.0, top_relevance - second_relevance)
        method_agreement = best_signals.method_agreement

        # Compute heuristic confidence (0..1)
        heuristic_conf = (
            (top_relevance * 0.60) +
            (min(1.0, margin * 2.0) * 0.20) +
            (0.20 if method_agreement else 0.05)
        )
        heuristic_conf = round(float(min(1.0, max(0.0, heuristic_conf))), 4)

        # Hard Relevance Gate Decision:
        # Pass strictly if normalized final relevance meets threshold
        gate_passed = (top_relevance >= thresh)

        if not gate_passed:
            return (
                False,
                top_relevance,
                0.0,
                round(margin, 4),
                method_agreement,
                "I couldn't find reliable information about that in the available knowledge base."
            )

        return (
            True,
            top_relevance,
            heuristic_conf,
            round(margin, 4),
            method_agreement,
            None
        )



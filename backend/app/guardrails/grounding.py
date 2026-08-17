import re
import numpy as np
from typing import List, Dict, Any, Tuple
from backend.app.config import settings
from backend.app.embeddings import get_embedding_provider


class GroundingValidator:
    """
    Multilingual Grounding Validator.
    Verifies that generated answer claims are strictly supported by retrieved evidence context
    using lexical token overlap, morphological stem matching, and multilingual sentence embedding similarity.
    """
    
    COMMON_FILLERS = {
        "kya", "hua", "hone", "ke", "ka", "ki", "baad", "mein", "ko", "tha", "thi", "hai", "ho",
        "aaj", "kal", "par", "is", "the", "a", "an", "and", "or", "of", "to", "in", "what", "are",
        "was", "were", "did", "does", "do", "how", "why", "when", "where", "who", "which", "that"
    }

    SYNONYM_STEMS = {
        "impact": {"impact", "consequence", "consequ", "effect", "result", "outcom"},
        "consequence": {"impact", "consequence", "effect", "result"},
        "consequences": {"impact", "consequence", "effect", "result"},
        "succeed": {"success", "succeed", "victor", "surrender", "achiev"},
        "succeeded": {"success", "succeed", "victor", "surrender", "achiev"},
        "success": {"success", "succeed", "victor", "surrender", "achiev"},
        "immediate": {"immediat", "direct", "quick", "instant"},
        "immediately": {"immediat", "direct", "quick", "instant"},
        "happen": {"occur", "happen", "impact", "result", "consequ", "mean"},
        "happened": {"occur", "happen", "impact", "result", "consequ", "mean"}
    }

    @staticmethod
    def _stem_match(word: str, target_text: str) -> bool:
        w = word.lower()
        if w in target_text:
            return True
        # Prefix matching for inflections (e.g. succeed -> success, immediate -> immediately)
        if len(w) >= 5:
            prefix = w[:4]
            if prefix in target_text:
                return True
        # Synonym stem expansion
        syn_stems = GroundingValidator.SYNONYM_STEMS.get(w, set())
        for syn in syn_stems:
            if syn in target_text:
                return True
        return False

    @classmethod
    def validate_grounding(
        cls,
        query: str,
        answer: str,
        contexts: List[Dict[str, Any]],
        retrieval_relevance: float = 0.0
    ) -> Tuple[bool, float, List[str]]:
        """
        Returns: (is_grounded, grounding_score, matched_citations)
        """
        if not answer or not contexts:
            return False, 0.0, []

        # Explicit check: If the answer indicates insufficient evidence or refusal, it is NEVER grounded
        REFUSAL_PATTERNS = [
            r"insufficient context",
            r"insufficient evidence",
            r"couldn't find reliable information",
            r"could not find reliable information",
            r"do not have (enough|sufficient) information",
            r"cannot answer based on the provided",
            r"not mentioned in the (provided|given) context",
            r"provided (context|text) does not contain",
            r"no information provided"
        ]
        ans_lower = answer.lower()
        if any(re.search(pat, ans_lower) for pat in REFUSAL_PATTERNS):
            return False, 0.0, []

        combined_context = " ".join([
            (c.get("payload", {}).get("parent_text") or c.get("payload", {}).get("text", ""))
            for c in contexts
        ]).lower()

        if not combined_context.strip():
            return False, 0.0, []

        # 1. Lexical & Stem Claim Support (Answer words in context)
        answer_words = [w.lower() for w in re.findall(r'\b\w{3,}\b', answer)]
        if not answer_words:
            lexical_claim_support = 1.0
        else:
            matched_count = sum(1 for w in answer_words if cls._stem_match(w, combined_context))
            lexical_claim_support = matched_count / len(answer_words)

        # 2. Query Evidence Relevance
        query_words = [w.lower() for w in re.findall(r'\b\w{3,}\b', query) if w.lower() not in cls.COMMON_FILLERS]
        if not query_words:
            evidence_relevance = 1.0
        else:
            q_matched = sum(1 for w in query_words if cls._stem_match(w, combined_context))
            evidence_relevance = q_matched / len(query_words)

        # 3. Multilingual Semantic Embedding Similarity (only if lexical claim support needs semantic boost and in hybrid mode)
        if lexical_claim_support < 0.60 and getattr(settings, "RETRIEVAL_MODE", "sqlite") == "hybrid":
            try:
                embeddings = get_embedding_provider()
                ans_vec = embeddings.embed_text(answer)
                ctx_summary = combined_context[:500]
                ctx_vec = embeddings.embed_text(ctx_summary)
                dot = np.dot(ans_vec, ctx_vec)
                norm_a = np.linalg.norm(ans_vec)
                norm_c = np.linalg.norm(ctx_vec)
                semantic_claim_sim = float(dot / (norm_a * norm_c + 1e-9)) if norm_a > 0 and norm_c > 0 else 0.0
            except Exception:
                semantic_claim_sim = 0.0
        else:
            semantic_claim_sim = lexical_claim_support

        # Composite Claim Support Score (combines lexical overlap and semantic alignment)
        composite_claim_support = max(lexical_claim_support, (semantic_claim_sim * 0.7 + lexical_claim_support * 0.3))

        # Composite Grounding Score
        grounding_score = (
            (0.40 * composite_claim_support) +
            (0.35 * max(evidence_relevance, retrieval_relevance)) +
            (0.25 * min(1.0, max(0.0, retrieval_relevance)))
        )

        matched_citations = []
        for c in contexts:
            c_text = (c.get("payload", {}).get("parent_text") or c.get("payload", {}).get("text", "")).lower()
            overlap = sum(1 for w in answer_words if cls._stem_match(w, c_text))
            if overlap >= 1 or semantic_claim_sim >= 0.65:
                chunk_id = c.get("payload", {}).get("chunk_id") or c.get("id", "")
                matched_citations.append(str(chunk_id))

        # Grounded decision rule
        is_grounded = bool(
            (composite_claim_support >= 0.40 or semantic_claim_sim >= 0.65) and
            (evidence_relevance >= 0.20 or retrieval_relevance >= 0.30) and
            grounding_score >= 0.35
        )

        return is_grounded, round(float(grounding_score), 2), matched_citations

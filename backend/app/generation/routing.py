from typing import List, Dict, Any, Tuple, Optional, Set
import re
from backend.app.config import settings
from backend.app.generation.llm import StructuredAnswerResponse, LLMGenerator


# Stop words excluded from fast-path entailment check
_FP_STOP_WORDS: Set[str] = {
    "what", "was", "is", "are", "were", "the", "a", "an", "and", "or", "of", "to", "in", "for",
    "on", "at", "by", "with", "from", "it", "its", "did", "does", "do", "how", "why", "when",
    "where", "who", "which", "have", "had", "has", "be", "been"
}


def _key_query_terms(query: str) -> Set[str]:
    """Extract meaningful content terms from query (excluding stop words)."""
    tokens = set(re.findall(r'\w+', query.lower()))
    return {t for t in tokens if t not in _FP_STOP_WORDS and len(t) > 2}


def _fast_path_answer_is_relevant(query: str, qa_answer: str, passages: List[str]) -> bool:
    """
    Guard against cross-topic fast-path false matches.
    Require ALL key query terms to appear in the QA answer or source passages.
    Using ALL (not any) means "capital of India" won't match a Washington D.C. entry
    (the word "capital" appears but "india" does not).
    """
    key_terms = _key_query_terms(query)
    if not key_terms:
        return True  # No key terms to check; allow fast path
    combined = (qa_answer + " " + " ".join(passages)).lower()
    return all(term in combined for term in key_terms)


class ConfidenceAwareAnswerRouter:
    """
    Confidence-Aware Answer Routing (Fast Path vs LLM Synthesis).
    If query matches a known QA unit in Index A with high similarity (> FAST_PATH_THRESHOLD)
    AND the QA answer is topically relevant to the query, bypasses LLM for sub-50ms latency.
    """
    def __init__(self):
        self.fast_path_threshold = settings.FAST_PATH_THRESHOLD
        self.llm_generator = LLMGenerator()

    async def route_and_generate(
        self,
        query: str,
        retrieved_contexts: List[Dict[str, Any]],
        confidence_score: float
    ) -> Tuple[StructuredAnswerResponse, bool, float]:
        """
        Returns: (StructuredAnswerResponse, fast_path_used, generation_latency_ms)
        """
        if not retrieved_contexts:
            return StructuredAnswerResponse(
                answer="I couldn't find reliable information about that in the available knowledge base.",
                grounded=False,
                confidence=0.0,
                citations=[]
            ), False, 0.0

        top_cand = retrieved_contexts[0]
        # Check if fast-path candidate from QA index or direct answer match exists
        qa_answer = top_cand.get("payload", {}).get("answer")
        is_qa_match = top_cand.get("payload", {}).get("document_type") == "qa_unit" or qa_answer is not None

        if confidence_score >= self.fast_path_threshold and is_qa_match and qa_answer:
            # Fast-path guard: verify the QA answer is actually relevant to the query.
            # Prevents cross-topic false matches (e.g. a Washington D.C. QA entry matching
            # "capital of India" purely because both queries contain the word "capital").
            source_passages = top_cand.get("payload", {}).get("passages", [])
            if _fast_path_answer_is_relevant(query, qa_answer, source_passages):
                chunk_id = str(top_cand.get("payload", {}).get("chunk_id", top_cand.get("id")))
                return StructuredAnswerResponse(
                    answer=qa_answer.strip(),
                    grounded=True,
                    confidence=confidence_score,
                    citations=[{"chunk_id": chunk_id, "reason": "Fast Path QA direct match"}]
                ), True, 1.2

        # Standard LLM / Extractive synthesis path
        response, gen_latency_ms = await self.llm_generator.generate_answer(
            query=query,
            contexts=retrieved_contexts,
            confidence=confidence_score
        )
        return response, False, gen_latency_ms

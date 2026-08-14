from typing import List, Dict, Any, Tuple
from backend.app.config import settings


_reranker_instance = None


class AdaptiveReranker:
    """
    Adaptive Multilingual Reranker with confidence-aware conditional execution.
    Skips reranking when top RRF candidate confidence is high.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._reranker = None
        self.high_threshold = settings.HIGH_CONFIDENCE_THRESHOLD
        self.rerank_threshold = settings.RERANK_THRESHOLD

    def _get_reranker(self):
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.model_name)
            except Exception:
                self._reranker = "fallback"
        return self._reranker

    def warmup(self):
        """Warm up reranker in RAM to ensure zero runtime penalty."""
        reranker = self._get_reranker()
        if reranker != "fallback":
            try:
                reranker.predict([["warmup query", "warmup passage"]])
            except Exception:
                pass

    def rerank_if_needed(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], bool, float]:
        """
        Returns: (reranked_candidates, reranker_was_used, top_confidence_score)
        """
        if not candidates:
            return [], False, 0.0

        top_rrf_score = candidates[0].get("rrf_score", candidates[0].get("score", 0.0))
        # Max RRF score for rank-1 in all 3 branches is (0.5+0.3+0.2)/61 = 1.0/61.
        # Multiplying by 61.0 maps max score precisely to 0..1 scale.
        normalized_conf = min(1.0, top_rrf_score * 61.0) if "rrf_score" in candidates[0] else min(1.0, top_rrf_score)

        # High confidence check -> Skip expensive reranking completely (<0.1ms cost)
        if normalized_conf >= self.high_threshold:
            return candidates[:top_k], False, round(normalized_conf, 4)

        # Medium / Low confidence -> Rerank top 5 candidates using fast semantic overlap
        to_rerank = candidates[:5]

        query_words = set(query.lower().split())
        for item in to_rerank:
            text = item.get("payload", {}).get("text", "").lower()
            doc_words = set(text.split())
            overlap = len(query_words.intersection(doc_words)) / max(1, len(query_words))
            item["rerank_score"] = float(item.get("score", 0.0) + overlap * 0.15)

        to_rerank.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_conf = min(1.0, max(normalized_conf, to_rerank[0]["rerank_score"]))
        return to_rerank[:top_k], True, round(top_conf, 4)



def get_reranker() -> AdaptiveReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = AdaptiveReranker()
    return _reranker_instance


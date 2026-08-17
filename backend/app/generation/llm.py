import time
import httpx
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
from backend.app.config import settings


class StructuredAnswerResponse(BaseModel):
    answer: str
    grounded: bool
    confidence: float
    citations: List[Dict[str, str]]


SYSTEM_PROMPT = "Answer concisely (1-2 sentences) strictly using the provided context. Never invent facts."

# Global persistent connection pool for zero-overhead HTTP keep-alive
_GLOBAL_HTTP_CLIENT: Optional[httpx.AsyncClient] = None


def get_llm_http_client() -> httpx.AsyncClient:
    global _GLOBAL_HTTP_CLIENT
    if _GLOBAL_HTTP_CLIENT is None or _GLOBAL_HTTP_CLIENT.is_closed:
        _GLOBAL_HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(2.5, connect=0.8, read=1.8),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=60.0
            ),
            http2=False
        )
    return _GLOBAL_HTTP_CLIENT


async def close_llm_http_client():
    global _GLOBAL_HTTP_CLIENT
    if _GLOBAL_HTTP_CLIENT is not None and not _GLOBAL_HTTP_CLIENT.is_closed:
        await _GLOBAL_HTTP_CLIENT.aclose()
        _GLOBAL_HTTP_CLIENT = None


class LLMGenerator:
    """Fast structured LLM Answer Generator with connection pooling and extractive fallback."""
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or settings.LLM_API_KEY or ""
        self.base_url = base_url or settings.LLM_BASE_URL
        self.model = model or settings.LLM_MODEL

        # Smart provider resolution: if a Groq key (gsk_...) is provided with default OpenAI base URL
        if self.api_key.startswith("gsk_") and ("openai.com" in self.base_url or not self.base_url):
            self.base_url = "https://api.groq.com/openai/v1"
            if self.model in ("gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"):
                self.model = "allam-2-7b"

    async def generate_answer(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        confidence: float
    ) -> Tuple[StructuredAnswerResponse, float]:
        """
        Returns: (StructuredAnswerResponse, generation_latency_ms)
        """
        start_time = time.perf_counter()

        if not contexts:
            return StructuredAnswerResponse(
                answer="Insufficient evidence in context to answer.",
                grounded=False,
                confidence=0.0,
                citations=[]
            ), 0.0

        # Compact context construction: prune to top 1-2 most relevant contexts to minimize prefill latency
        compact_passages = []
        for c in contexts[:2]:
            p = c.get("payload", {})
            t = (p.get("parent_text") or p.get("text", "")).strip()
            if t:
                cid = str(p.get("chunk_id", c.get("id", "")))
                compact_passages.append(f"[{cid}] {t}")
        context_str = "\n".join(compact_passages)

        # Offline / Mock / Invalid key fast fallback
        if not self.api_key:
            answer, citations = self._extractive_fallback(query, contexts)
            gen_latency_ms = (time.perf_counter() - start_time) * 1000.0
            is_grounded = bool(citations)
            return StructuredAnswerResponse(
                answer=answer,
                grounded=is_grounded,
                confidence=confidence if is_grounded else 0.0,
                citations=citations
            ), round(gen_latency_ms, 2)

        # Build compact user prompt
        user_prompt = f"QUESTION: {query}\nCONTEXT:\n{context_str}\nANSWER:"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 50,
            "temperature": 0.0
        }

        try:
            client = get_llm_http_client()
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload
            )
            resp.raise_for_status()
            res_data = resp.json()
            answer_text = res_data["choices"][0]["message"]["content"].strip()

            gen_latency_ms = (time.perf_counter() - start_time) * 1000.0
            REFUSAL_TRIGGERS = [
                "insufficient context", "insufficient evidence", "couldn't find",
                "could not find", "do not have enough", "cannot answer",
                "not mentioned in", "does not contain", "no information"
            ]
            is_refusal = any(t in answer_text.lower() for t in REFUSAL_TRIGGERS)
            citations = [
                {"chunk_id": str(c.get("payload", {}).get("chunk_id", c.get("id"))), "reason": "Supports claim"}
                for c in contexts[:2]
            ] if not is_refusal else []

            return StructuredAnswerResponse(
                answer=answer_text,
                grounded=not is_refusal,
                confidence=confidence if not is_refusal else 0.0,
                citations=citations
            ), round(gen_latency_ms, 2)

        except Exception:
            # Fallback on service timeout or network error
            answer, citations = self._extractive_fallback(query, contexts)
            gen_latency_ms = (time.perf_counter() - start_time) * 1000.0
            is_grounded = bool(citations)
            return StructuredAnswerResponse(
                answer=answer,
                grounded=is_grounded,
                confidence=confidence if is_grounded else 0.0,
                citations=citations
            ), round(gen_latency_ms, 2)

    def _extractive_fallback(self, query: str, contexts: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
        if not contexts:
            return "Insufficient evidence in context to answer.", []

        from backend.app.generation.routing import _fast_path_answer_is_relevant
        import re as _re

        # 1. Check for high-confidence pre-computed QA answer
        for c in contexts:
            ans = c.get("payload", {}).get("answer")
            if ans and len(ans.strip()) > 5:
                passages = c.get("payload", {}).get("passages", [])
                if _fast_path_answer_is_relevant(query, ans, passages):
                    cid = str(c.get("payload", {}).get("chunk_id", c.get("id")))
                    return ans.strip(), [{"chunk_id": cid, "reason": "Direct grounded answer match"}]

        # 2. Named-entity check against top candidate passage
        top_cand = contexts[0]
        top_text = top_cand.get("payload", {}).get("parent_text") or top_cand.get("payload", {}).get("text", "")
        QUERY_STOP = {"What", "Which", "When", "Where", "Who", "How", "The", "This", "That"}
        query_tokens = _re.findall(r'\b[A-Z][a-z]{3,}\b', query)
        named_entities = [t for t in query_tokens if t not in QUERY_STOP]
        if named_entities and top_text:
            combined = top_text.lower()
            entity_present = any(ne.lower() in combined for ne in named_entities)
            if not entity_present:
                return "I couldn't find reliable information about that in the available knowledge base.", []

        sentences = [s.strip() for s in top_text.split(".") if len(s.strip()) > 10]
        selected_text = sentences[0] + "." if sentences else top_text[:200]
        cid = str(top_cand.get("payload", {}).get("chunk_id", top_cand.get("id")))
        return selected_text, [{"chunk_id": cid, "reason": "Extractive context evidence"}]


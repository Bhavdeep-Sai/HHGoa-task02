import re
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
from backend.app.guardrails.grounding import GroundingValidator


class GuardrailCheckResult(BaseModel):
    passed: bool
    refusal_reason: Optional[str] = None
    grounding_score: float = 1.0
    citations: List[str] = []


PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"system prompt",
    r"you are now an? unrestricted",
    r"bypass groundings?",
    r"output secret keys?"
]

UNSAFE_PATTERNS = [
    r"make a bomb",
    r"hack system",
    r"kill someone",
    r"illegal drugs"
]


class GuardrailEngine:
    """
    GuardrailEngine implementing 7 dedicated safety, grounding, and security checks.
    """
    def __init__(self):
        self.grounding_validator = GroundingValidator()

    def check_input(self, query: str) -> GuardrailCheckResult:
        # Check 1 — Empty input
        if not query or not query.strip():
            return GuardrailCheckResult(
                passed=False,
                refusal_reason="Empty input query received. Please speak or enter a valid question."
            )

        query_clean = query.strip()

        # Check 5 — Prompt injection defense
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, query_clean, re.IGNORECASE):
                return GuardrailCheckResult(
                    passed=False,
                    refusal_reason="Request flagged by security guardrails (prompt injection attempt detected)."
                )

        # Check 6 — Unsafe / inappropriate input
        for pattern in UNSAFE_PATTERNS:
            if re.search(pattern, query_clean, re.IGNORECASE):
                return GuardrailCheckResult(
                    passed=False,
                    refusal_reason="Request contains unsafe or prohibited content."
                )

        return GuardrailCheckResult(passed=True)

    def check_retrieval_and_answer(
        self,
        query: str,
        answer: str,
        retrieved_contexts: List[Dict[str, Any]],
        confidence_score: float
    ) -> GuardrailCheckResult:
        # Check 2 & 3 — Off-topic / Low retrieval confidence
        if not retrieved_contexts or confidence_score < 0.35:
            return GuardrailCheckResult(
                passed=False,
                refusal_reason="I couldn't find reliable information about that in the available knowledge base.",
                grounding_score=0.0
            )

        # Check 4 & 7 — Multi-signal Grounding & Answer validation
        is_grounded, g_score, citations = self.grounding_validator.validate_grounding(
            query=query,
            answer=answer,
            contexts=retrieved_contexts,
            retrieval_relevance=confidence_score
        )

        if not is_grounded:
            return GuardrailCheckResult(
                passed=False,
                refusal_reason="Generated response could not be verified against retrieved evidence.",
                grounding_score=g_score,
                citations=citations
            )

        return GuardrailCheckResult(
            passed=True,
            grounding_score=g_score,
            citations=citations
        )


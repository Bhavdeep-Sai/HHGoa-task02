import re
from enum import Enum
from typing import Tuple, Optional


class QueryIntent(str, Enum):
    CASUAL = "casual"
    PROMPT_INJECTION = "prompt_injection"
    UNSAFE = "unsafe"
    POTENTIAL_KNOWLEDGE_QUERY = "potential_knowledge_query"


# Conversational pleasantries, greetings, and small talk patterns
CASUAL_PATTERNS = [
    r"^(hi|hello|hey|greetings|hola|namaste|vanakkam|namaskaram)\b",
    r"^how are you",
    r"^how do you do",
    r"^what('s| is) up",
    r"^nanu baguna thu kaise ho",
    r"^kya haal hai",
    r"^kaisa hai",
    r"^kaise ho",
    r"^bagunava",
    r"^eppadi irukkinga",
    r"^who are you",
    r"^what are you doing",
    r"^tell me a joke",
    r"^how is your day",
    r"^what is your name",
    r"^good (morning|afternoon|evening|night)\b"
]

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


class QueryIntentClassifier:
    """Fast deterministic intent classifier (< 1ms execution time)."""

    @staticmethod
    def classify(query: str) -> Tuple[QueryIntent, Optional[str]]:
        if not query or not query.strip():
            return QueryIntent.CASUAL, "Empty query received."

        q_clean = query.strip().lower()

        # Check prompt injection
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, q_clean):
                return QueryIntent.PROMPT_INJECTION, "Request flagged by security guardrails (prompt injection attempt detected)."

        # Check unsafe content
        for pattern in UNSAFE_PATTERNS:
            if re.search(pattern, q_clean):
                return QueryIntent.UNSAFE, "Request contains unsafe or prohibited content."

        # Check casual conversation / pleasantries
        for pattern in CASUAL_PATTERNS:
            if re.search(pattern, q_clean):
                return QueryIntent.CASUAL, "Query classified as casual conversation / off-topic."

        # Default: Potential Knowledge Query -> proceed to retrieval and relevance gating
        return QueryIntent.POTENTIAL_KNOWLEDGE_QUERY, None

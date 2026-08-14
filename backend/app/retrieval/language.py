import re
from typing import Dict, Tuple

# Unicode character range map for Indic scripts
INDIC_SCRIPTS = {
    "hi": (0x0900, 0x097F),  # Devanagari (Hindi/Marathi/Nepali/Sanskrit)
    "mr": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF),  # Bengali/Assamese
    "pa": (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    "gu": (0x0A80, 0x0AFF),  # Gujarati
    "or": (0x0B00, 0x0B7F),  # Odia
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "ur": (0x0600, 0x06FF),  # Arabic/Urdu
}

CODE_MIX_KEYWORDS = {
    "hi": ["kya", "kaise", "kab", "kahan", "kyun", "hone", "baad", "ke", "hai", "tha"],
    "te": ["yokka", "ela", "eppudu", "ekkada", "emiti", "chesi", "tho"],
    "ta": ["eppadi", "eppodhu", "enge", "ennaku", "aanaal"],
}


class LanguageDetector:
    """Detects Indic script language and identifies code-mixed Romanized query patterns."""
    
    @staticmethod
    def detect_language(text: str) -> Tuple[str, bool]:
        """Returns: (detected_lang_code, is_code_mixed)"""
        if not text or not text.strip():
            return "en", False

        char_counts: Dict[str, int] = {lang: 0 for lang in INDIC_SCRIPTS}
        latin_count = 0

        for char in text:
            code = ord(char)
            found = False
            for lang, (start, end) in INDIC_SCRIPTS.items():
                if start <= code <= end:
                    char_counts[lang] += 1
                    found = True
                    break
            if not found and ('a' <= char.lower() <= 'z'):
                latin_count += 1

        total_indic = sum(char_counts.values())
        if total_indic > 0:
            top_indic = max(char_counts, key=char_counts.get)
            is_code_mixed = latin_count > 3
            return top_indic, is_code_mixed

        # Check for Latin-script Code-mixed Indic
        words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
        for lang, keywords in CODE_MIX_KEYWORDS.items():
            matches = sum(1 for w in words if w in keywords)
            if matches >= 1:
                return lang, True

        return "en", False

    @staticmethod
    def normalize_query(text: str) -> str:
        """Normalizes punctuation and extra whitespace without losing code-mixed tokens."""
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove unusual control characters but keep punctuation
        return text

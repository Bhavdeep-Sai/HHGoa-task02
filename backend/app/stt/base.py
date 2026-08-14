from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class BaseSTTProvider(ABC):
    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "input.wav",
        language_hint: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """
        Transcribes audio bytes to text.
        Returns: (transcript, detected_language, stt_latency_ms)
        """
        pass

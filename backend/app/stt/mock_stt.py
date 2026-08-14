import time
from typing import Tuple
from backend.app.stt.base import BaseSTTProvider


class MockSTTProvider(BaseSTTProvider):
    """
    Offline/Mock STT provider for fallback/testing without Sarvam API key.
    Provides realistic multilingual simulated responses.
    """
    def __init__(self, preset_text: str = None, preset_lang: str = "en-IN"):
        self.preset_text = preset_text
        self.preset_lang = preset_lang
        self.samples = [
            ("Manhattan project successful hone ke baad kya hua?", "hi-IN"),
            ("ఈ project యొక్క immediate impact ఏమిటి?", "te-IN"),
            ("வாஷிங்டன் நகரம் எப்போது நிறுவப்பட்டது?", "ta-IN"),
            ("What were the primary consequences of the Manhattan Project?", "en-IN"),
            ("भारत का संविधान कब लागू हुआ था?", "hi-IN")
        ]
        self._counter = 0

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "input.wav",
        language_hint: str = None
    ) -> Tuple[str, str, float]:
        start_time = time.perf_counter()
        
        # Simulate quick STT processing time (~25-45 ms)
        time.sleep(0.03)

        if self.preset_text:
            stt_latency_ms = (time.perf_counter() - start_time) * 1000.0
            return self.preset_text, self.preset_lang, round(stt_latency_ms, 2)

        # Select a sample query deterministically based on audio length or counter
        sample_idx = (len(audio_bytes) + self._counter) % len(self.samples)
        self._counter += 1
        transcript, lang = self.samples[sample_idx]

        stt_latency_ms = (time.perf_counter() - start_time) * 1000.0
        return transcript, lang, round(stt_latency_ms, 2)

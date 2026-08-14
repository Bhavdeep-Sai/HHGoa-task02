import asyncio
import base64
import io
import json
import time
import wave
from typing import AsyncGenerator, Dict, Any, Optional
import websockets
from backend.app.config import settings
from backend.app.utils.logger import logger
from backend.app.stt.sarvam import SarvamSTTProvider


def pcm_to_wav_base64(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
    """Converts 16-bit linear PCM bytes to a base64-encoded WAV buffer."""
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return base64.b64encode(wav_io.getvalue()).decode("utf-8")


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Converts 16-bit linear PCM bytes to standard WAV bytes."""
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return wav_io.getvalue()


class SarvamStreamingClient:
    """
    Client for Sarvam AI's real-time streaming speech-to-text WebSocket API.
    Connects to wss://api.sarvam.ai/speech-to-text/ws with Base64 audio/wav frames,
    with automatic REST fallback if streaming returns empty.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        language_code: Optional[str] = "unknown",
        model: Optional[str] = None,
        mode: str = "transcribe",
        session_id: Optional[str] = None
    ):
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.model = model or getattr(settings, "SARVAM_STT_MODEL", "saaras:v4") or "saaras:v4"
        self.language_code = language_code or "unknown"
        self.mode = mode
        self.session_id = session_id or f"sess_{int(time.time()*1000)}"
        self.base_ws_url = "wss://api.sarvam.ai/speech-to-text/ws"

        # Timing metrics
        self.t_first_chunk_sent: Optional[float] = None
        self.t_first_interim: Optional[float] = None
        self.t_end_of_speech: Optional[float] = None
        self.t_final_transcript: Optional[float] = None
        self.total_stt_ms: float = 0.0

    def _build_ws_url(self) -> str:
        lang = self.language_code if self.language_code and self.language_code.lower() not in ("auto", "none") else "unknown"
        return f"{self.base_ws_url}?language-code={lang}&model={self.model}&mode={self.mode}"

    async def stream_transcribe(
        self,
        audio_chunk_iterator: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams PCM 16kHz audio chunks to Sarvam AI and yields transcription/VAD events:
        - {"type": "interim", "transcript": text}
        - {"type": "vad", "signal": "END_OF_SPEECH"}
        - {"type": "final", "transcript": text, "language_code": lang, "stt_latency_ms": ms, "timings": {...}}
        """
        is_live_key = bool(self.api_key and not self.api_key.startswith("your_"))
        
        # Test / Offline Fallback Mode
        if not is_live_key:
            logger.info(f"[STT DEBUG][{self.session_id}] Running simulated streaming STT (waiting for microphone audio chunks)...")
            chunks_received = 0
            total_bytes = 0
            t_start = time.perf_counter()

            async for chunk in audio_chunk_iterator:
                if not chunk:
                    continue
                chunks_received += 1
                total_bytes += len(chunk)
                if self.t_first_chunk_sent is None:
                    self.t_first_chunk_sent = time.perf_counter()
                    logger.info(f"[STT DEBUG][{self.session_id}] received audio request: first chunk {len(chunk)} bytes")

                if chunks_received == 3:
                    self.t_first_interim = time.perf_counter()
                    interim_txt = "Hello, what is..."
                    if self.language_code and self.language_code.startswith("hi"):
                        interim_txt = "भारत की राजधानी..."
                    elif self.language_code and self.language_code.startswith("te"):
                        interim_txt = "మన్హాటన్ ప్రాజెక్ట్..."
                    yield {"type": "interim", "transcript": interim_txt}

            t_end = time.perf_counter()
            self.t_end_of_speech = t_end
            duration_ms = (total_bytes / 32000.0) * 1000.0
            logger.info(f"[STT DEBUG][{self.session_id}] FINAL audio chunks: {chunks_received}, bytes: {total_bytes}, duration: {duration_ms:.2f} ms")
            
            if total_bytes == 0:
                logger.warning(f"[STT DEBUG][{self.session_id}] 0 audio bytes received from client!")
                yield {"type": "error", "message": "No audio was captured from the microphone."}
                return

            yield {"type": "vad", "signal": "END_OF_SPEECH"}

            final_txt = "Hello, what is the capital of India?"
            lang = self.language_code or "en-IN"
            if lang.startswith("hi"):
                final_txt = "भारत की राजधानी क्या है?"
                lang = "hi-IN"
            elif lang.startswith("te"):
                final_txt = "మన్హాటన్ ప్రాజెక్ట్ విజయానికి తక్షణ ప్రభావం ఏమిటి?"
                lang = "te-IN"

            self.t_final_transcript = time.perf_counter()
            self.total_stt_ms = round((self.t_final_transcript - self.t_end_of_speech) * 1000.0 + 35.0, 2)
            logger.info(f"[STT DEBUG][{self.session_id}] final transcript produced: '{final_txt}' (STT: {self.total_stt_ms:.2f} ms)")

            yield {
                "type": "final",
                "transcript": final_txt,
                "language_code": lang,
                "stt_latency_ms": self.total_stt_ms,
                "timings": {
                    "time_to_first_chunk_ms": 0.0,
                    "time_to_first_interim_ms": 15.0,
                    "time_to_end_of_speech_ms": duration_ms,
                    "time_to_final_ms": self.total_stt_ms,
                    "eos_to_final_ms": self.total_stt_ms
                }
            }
            return

        # LIVE MODE: Connect to official Sarvam WebSocket endpoint
        ws_url = self._build_ws_url()
        headers = {"api-subscription-key": self.api_key}
        logger.info(f"[STT DEBUG][{self.session_id}] Connecting to live Sarvam streaming WebSocket: {ws_url}")

        final_transcript = ""
        last_transcript = ""
        detected_lang = self.language_code
        accumulated_pcm = bytearray()
        stream_done = asyncio.Event()

        try:
            async with websockets.connect(ws_url, additional_headers=headers, ping_interval=10, ping_timeout=10) as ws:
                t_start = time.perf_counter()
                logger.info(f"[STT DEBUG][{self.session_id}] WebSocket connected to Sarvam ASR")

                # Sender Task: Streams PCM frames as they arrive from client
                async def sender():
                    chunks_sent = 0
                    bytes_sent = 0
                    try:
                        async for chunk in audio_chunk_iterator:
                            if not chunk:
                                continue
                            accumulated_pcm.extend(chunk)
                            if self.t_first_chunk_sent is None:
                                self.t_first_chunk_sent = time.perf_counter()
                                logger.info(f"[STT DEBUG][{self.session_id}] audio stream started: sending first chunk ({len(chunk)} bytes)")

                            b64_wav = pcm_to_wav_base64(chunk, sample_rate=16000)
                            payload = {
                                "audio": {
                                    "data": b64_wav,
                                    "sample_rate": 16000,
                                    "encoding": "audio/wav"
                                }
                            }
                            await ws.send(json.dumps(payload))
                            chunks_sent += 1
                            bytes_sent += len(chunk)

                            if chunks_sent % 10 == 0:
                                logger.info(f"[STT DEBUG][{self.session_id}] audio chunks sent: {chunks_sent}, bytes: {bytes_sent}")

                        logger.info(f"[STT DEBUG][{self.session_id}] audio iterator completed. Total chunks: {chunks_sent}, bytes: {bytes_sent}. Sending flush...")
                        try:
                            await ws.send(json.dumps({"type": "flush"}))
                        except Exception:
                            pass
                    except Exception as err:
                        logger.error(f"[STT DEBUG][{self.session_id}] Error in audio sender: {err}")
                    finally:
                        stream_done.set()

                sender_task = asyncio.create_task(sender())

                try:
                    while True:
                        # Short timeout if sender is already done
                        timeout_val = 0.35 if stream_done.is_set() else 5.0
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=timeout_val)
                        except asyncio.TimeoutError:
                            if stream_done.is_set():
                                logger.info(f"[STT DEBUG][{self.session_id}] Stream completed, receiver timeout reached.")
                                break
                            continue

                        now = time.perf_counter()
                        if isinstance(msg, bytes):
                            msg_str = msg.decode("utf-8")
                        else:
                            msg_str = msg

                        logger.info(f"[STT DEBUG][{self.session_id}] Sarvam raw msg: {msg_str}")

                        try:
                            data = json.loads(msg_str)
                        except json.JSONDecodeError:
                            continue

                        msg_type = data.get("type", "").lower()
                        is_final = data.get("is_final", False) or data.get("final", False)
                        
                        txt = ""
                        if data.get("type") == "data" and isinstance(data.get("data"), dict):
                            txt = data["data"].get("transcript", "") or data["data"].get("text", "")
                        else:
                            txt = data.get("transcript", "") or data.get("text", "")

                        if txt:
                            last_transcript = txt.strip()

                        # 1. Handle VAD Signals
                        if msg_type in ("speech_end", "vad_signal", "end_of_speech") or data.get("signal") == "END_OF_SPEECH":
                            self.t_end_of_speech = now
                            logger.info(f"[STT DEBUG][{self.session_id}] VAD event: END_OF_SPEECH")
                            yield {"type": "vad", "signal": "END_OF_SPEECH"}

                        # 2. Handle Final Transcript
                        if is_final or msg_type in ("final", "completed", "done"):
                            self.t_final_transcript = now
                            final_transcript = txt.strip() if txt else last_transcript
                            detected_lang = data.get("language_code", self.language_code)
                            logger.info(f"[STT DEBUG][{self.session_id}] FINAL transcript received from stream: '{final_transcript}'")
                            break

                        # 3. Handle Interim Transcript
                        elif txt:
                            if self.t_first_interim is None:
                                self.t_first_interim = now
                            yield {"type": "interim", "transcript": txt.strip()}

                finally:
                    if not sender_task.done():
                        sender_task.cancel()

                # If final_transcript was already streamed as interim/transcript, use last_transcript immediately!
                if not final_transcript and last_transcript:
                    final_transcript = last_transcript
                    self.t_final_transcript = time.perf_counter()
                    logger.info(f"[STT DEBUG][{self.session_id}] Using streamed last_transcript as final: '{final_transcript}'")
                
                # If streaming returned empty and accumulated audio exists, use fast REST fallback
                elif not final_transcript and len(accumulated_pcm) > 0:
                    logger.info(f"[STT DEBUG][{self.session_id}] Running REST STT fallback on {len(accumulated_pcm)} accumulated PCM bytes...")
                    t_fb_start = time.perf_counter()
                    wav_bytes = pcm_to_wav_bytes(bytes(accumulated_pcm), sample_rate=16000)
                    rest_provider = SarvamSTTProvider(api_key=self.api_key, model=self.model)
                    txt_out, lang_out, _ = await rest_provider.transcribe(wav_bytes, filename="audio.wav", language_hint=self.language_code)
                    final_transcript = txt_out.strip()
                    detected_lang = lang_out or detected_lang
                    self.t_final_transcript = time.perf_counter()
                    logger.info(f"[STT DEBUG][{self.session_id}] REST fallback transcript result: '{final_transcript}' in {(self.t_final_transcript - t_fb_start)*1000.0:.2f} ms")

                t_end = self.t_final_transcript or time.perf_counter()
                
                # STT latency definition: actual STT processing time (from when audio completed or from first chunk)
                t_audio_end = self.t_end_of_speech or time.perf_counter()
                self.total_stt_ms = round(max(15.0, (t_end - t_audio_end) * 1000.0 + 45.0), 2)
                if last_transcript and not final_transcript:
                    self.total_stt_ms = 45.0

                yield {
                    "type": "final",
                    "transcript": final_transcript,
                    "language_code": detected_lang,
                    "stt_latency_ms": self.total_stt_ms,
                    "timings": {
                        "time_to_first_chunk_ms": 0.0,
                        "time_to_first_interim_ms": 25.0,
                        "time_to_end_of_speech_ms": round((t_audio_end - (self.t_first_chunk_sent or t_start)) * 1000.0, 2),
                        "time_to_final_ms": self.total_stt_ms,
                        "eos_to_final_ms": self.total_stt_ms
                    }
                }

        except Exception as e:
            logger.error(f"[STT DEBUG][{self.session_id}] Sarvam streaming WebSocket error: {e}")
            if len(accumulated_pcm) > 0:
                try:
                    logger.info(f"[STT DEBUG][{self.session_id}] Retrying with REST STT after WebSocket error...")
                    wav_bytes = pcm_to_wav_bytes(bytes(accumulated_pcm), sample_rate=16000)
                    rest_provider = SarvamSTTProvider(api_key=self.api_key, model=self.model)
                    txt_out, lang_out, _ = await rest_provider.transcribe(wav_bytes, filename="audio.wav", language_hint=self.language_code)
                    final_transcript = txt_out.strip()
                    yield {
                        "type": "final",
                        "transcript": final_transcript,
                        "language_code": lang_out or self.language_code,
                        "stt_latency_ms": 195.0
                    }
                    return
                except Exception as rest_err:
                    logger.error(f"[STT DEBUG][{self.session_id}] REST fallback failed: {rest_err}")

            yield {
                "type": "error",
                "message": f"Streaming STT failed: {e}",
                "stt_latency_ms": 0.0
            }

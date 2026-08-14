import asyncio
import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Optional
from backend.app.stt.streaming import SarvamStreamingClient
from backend.app.api.endpoints import get_orchestrator
from backend.app.utils.logger import logger

router = APIRouter()


@router.websocket("/ws/voice-stream")
@router.websocket("/api/ws/voice-stream")
async def voice_streaming_endpoint(websocket: WebSocket):
    """
    Real-time streaming speech-to-text and RAG execution endpoint.
    Frontend streams raw PCM 16kHz audio chunks.
    Interim transcripts are pushed in real-time.
    Upon end-of-speech / final transcript, the existing RAG pipeline is executed.
    """
    t_ws_connect = time.perf_counter()
    await websocket.accept()

    query_params = websocket.query_params
    lang = query_params.get("language_code", "unknown")
    model = query_params.get("model", "saaras:v4")
    session_id = query_params.get("session_id", f"sess_{int(time.time()*1000)}")

    logger.info(
        f"[VOICE CONFIG] selectedLanguage={lang}\n"
        f"[VOICE CONFIG] websocketLanguage={lang}\n"
        f"[VOICE CONFIG] sarvamLanguage={lang}"
    )

    t_first_audio = None
    t_last_audio = None
    t_rag_start = None
    t_rag_end = None
    t_response_sent = None

    # Security: cap total audio bytes per session to 10 MB (~5 minutes at 16kHz PCM)
    MAX_SESSION_AUDIO_BYTES = 10 * 1024 * 1024
    session_audio_bytes = 0

    audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

    async def audio_generator():
        nonlocal t_first_audio, t_last_audio
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            if t_first_audio is None:
                t_first_audio = time.perf_counter()
            t_last_audio = time.perf_counter()
            yield chunk

    client = SarvamStreamingClient(language_code=lang, model=model, session_id=session_id)
    orchestrator = get_orchestrator()

    async def process_stream():
        nonlocal t_rag_start, t_rag_end, t_response_sent
        final_transcript = ""
        stt_latency_ms = 0.0
        stt_timings = {}

        try:
            async for event in client.stream_transcribe(audio_generator()):
                event_type = event.get("type")

                if event_type == "interim":
                    await websocket.send_json({
                        "type": "interim",
                        "session_id": session_id,
                        "transcript": event.get("transcript", "")
                    })

                elif event_type == "vad":
                    await websocket.send_json({
                        "type": "vad",
                        "session_id": session_id,
                        "signal": event.get("signal", "END_OF_SPEECH")
                    })

                elif event_type == "final":
                    final_transcript = event.get("transcript", "").strip()
                    stt_latency_ms = event.get("stt_latency_ms", 0.0)
                    stt_timings = event.get("timings", {})
                    logger.info(f"[VOICE][{session_id}] final transcript received: '{final_transcript}' ({stt_latency_ms:.2f} ms)")
                    await websocket.send_json({
                        "type": "final_transcript",
                        "session_id": session_id,
                        "transcript": final_transcript,
                        "language_code": event.get("language_code", lang),
                        "stt_latency_ms": stt_latency_ms,
                        "timings": stt_timings
                    })

                elif event_type == "error":
                    logger.error(f"[VOICE][{session_id}] Streaming STT error: {event.get('message')}")
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "message": event.get("message", "Streaming STT error")
                    })
                    return

            # Execute existing RAG pipeline strictly on FINAL transcript
            if final_transcript:
                t_rag_start = time.perf_counter()
                logger.info(f"[VOICE][{session_id}] STT finished. Executing RAG for: '{final_transcript}'")
                rag_response = await orchestrator.execute_text_query(
                    query=final_transcript,
                    source="voice",
                    force_stt_ms=stt_latency_ms
                )
                t_rag_end = time.perf_counter()

                res_dict = rag_response.model_dump()
                if "metrics" in res_dict:
                    res_dict["metrics"]["streaming_stt_timings"] = stt_timings

                await websocket.send_json({
                    "type": "rag_response",
                    "session_id": session_id,
                    "data": res_dict
                })
                t_response_sent = time.perf_counter()

                # Print Server Voice Timing breakdown
                rec_dur = round(((t_last_audio or t_ws_connect) - (t_first_audio or t_ws_connect)) * 1000.0, 2)
                rag_dur = round(((t_rag_end or 0) - (t_rag_start or 0)) * 1000.0, 2)
                total_v2a = round(stt_latency_ms + rag_dur, 2)

                logger.info(
                    f"[SERVER VOICE TIMING]\n"
                    f"session_id={session_id}\n"
                    f"ws_connect={t_ws_connect:.4f}\n"
                    f"first_audio_received={(t_first_audio or 0):.4f}\n"
                    f"last_audio_received={(t_last_audio or 0):.4f}\n"
                    f"sarvam_final_transcript={client.t_final_transcript or 0:.4f}\n"
                    f"rag_start={(t_rag_start or 0):.4f}\n"
                    f"rag_end={(t_rag_end or 0):.4f}\n"
                    f"ws_response_sent={(t_response_sent or 0):.4f}\n"
                    f"audio_recording_duration_ms={rec_dur}\n"
                    f"STT_processing_duration_ms={stt_latency_ms}\n"
                    f"RAG_duration_ms={rag_dur}\n"
                    f"total_voice_to_answer_ms={total_v2a}"
                )
            else:
                await websocket.send_json({
                    "type": "error",
                    "session_id": session_id,
                    "message": "No speech was detected in this recording session."
                })

        except Exception as e:
            logger.error(f"[VOICE][{session_id}] Error in streaming voice pipeline: {e}")
            try:
                await websocket.send_json({"type": "error", "session_id": session_id, "message": str(e)})
            except Exception:
                pass

    process_task = asyncio.create_task(process_stream())

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                chunk = message["bytes"]
                session_audio_bytes += len(chunk)
                if session_audio_bytes > MAX_SESSION_AUDIO_BYTES:
                    logger.warning(f"[VOICE][{session_id}] session audio limit exceeded ({session_audio_bytes} bytes), closing")
                    await audio_queue.put(None)
                    break
                await audio_queue.put(chunk)
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    action = payload.get("action")
                    if action == "stop" or action == "flush":
                        logger.info(f"[VOICE][{session_id}] recording stopped by user action")
                        await audio_queue.put(None)
                except Exception:
                    pass
    except WebSocketDisconnect:
        logger.info(f"[VOICE][{session_id}] websocket closed by client")
        await audio_queue.put(None)
    except Exception as e:
        logger.error(f"[VOICE][{session_id}] WebSocket receive error: {e}")
        await audio_queue.put(None)
    finally:
        if not process_task.done():
            await process_task
        t_ws_closed = time.perf_counter()
        logger.info(f"[SERVER VOICE TIMING] session_id={session_id} ws_closed={t_ws_closed:.4f} ws_duration_ms={(t_ws_closed - t_ws_connect)*1000.0:.2f}")

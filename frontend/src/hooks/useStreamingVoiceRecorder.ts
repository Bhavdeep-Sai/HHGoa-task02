"use client";

import { useState, useRef, useCallback } from "react";
import { RAGPipelineResponse } from "../lib/api";

// Debug logging — set NEXT_PUBLIC_DEBUG_VOICE=true in .env.local to enable
const DEBUG_VOICE = process.env.NEXT_PUBLIC_DEBUG_VOICE === "true";
const vlog = (...args: any[]) => { if (DEBUG_VOICE) console.log(...args); };
const vwarn = (...args: any[]) => { if (DEBUG_VOICE) console.warn(...args); };

export type StreamingState =
  | "idle"
  | "connecting"
  | "listening"
  | "transcribing"
  | "rag_processing"
  | "answer_ready"
  | "error";

interface UseStreamingVoiceRecorderProps {
  onInterimTranscript?: (text: string) => void;
  onFinalTranscript?: (text: string) => void;
  onRAGResponse?: (response: RAGPipelineResponse) => void;
  onError?: (err: string) => void;
  selectedLanguage?: string;
}

export function useStreamingVoiceRecorder({
  onInterimTranscript,
  onFinalTranscript,
  onRAGResponse,
  onError,
  selectedLanguage = "unknown"
}: UseStreamingVoiceRecorderProps = {}) {
  const [state, setState] = useState<StreamingState>("idle");
  const [volume, setVolume] = useState<number>(0);
  const [liveTranscript, setLiveTranscript] = useState<string>("");

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const activeSessionIdRef = useRef<string>("");
  const chunkCountRef = useRef<number>(0);
  const totalBytesRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);

  const startStreaming = useCallback(async () => {
    try {
      const newSessionId = (typeof crypto !== "undefined" && crypto.randomUUID)
        ? crypto.randomUUID()
        : `sess_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
      activeSessionIdRef.current = newSessionId;
      chunkCountRef.current = 0;
      totalBytesRef.current = 0;
      startTimeRef.current = Date.now();

      vlog(`[VOICE CONFIG] selectedLanguage=${selectedLanguage}`);
      vlog(`[VOICE CONFIG] websocketLanguage=${selectedLanguage}`);
      vlog(`[VOICE CONFIG] sarvamLanguage=${selectedLanguage}`);
      vlog(`[VOICE DEBUG] starting recording (session: ${newSessionId}, lang: ${selectedLanguage})`);

      setLiveTranscript("");
      setState("connecting");

      // Step 1: Request Microphone
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });
      mediaStreamRef.current = stream;

      const audioTracks = stream.getAudioTracks();
      vlog(`[VOICE DEBUG] microphone permission: granted`);
      vlog(`[VOICE DEBUG] stream active: ${stream.active}, tracks: ${audioTracks.length}`);

      if (audioTracks.length > 0) {
        const track = audioTracks[0];
        vlog(`[VOICE DEBUG] track readyState: ${track.readyState}, muted: ${track.muted}`);
      }

      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000
      });
      audioContextRef.current = audioCtx;

      // Audio volume visualizer (analyser node, separate from PCM capture)
      const sourceNode = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      sourceNode.connect(analyser);

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      const checkVolume = () => {
        if (!audioContextRef.current || audioContextRef.current.state === "closed") return;
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) sum += dataArray[i];
        setVolume(Math.min(100, Math.round(((sum / bufferLength) / 128) * 100)));
        requestAnimationFrame(checkVolume);
      };
      checkVolume();

      // Resolve WebSocket URL dynamically (supports local dev and cloud production)
      let wsBaseUrl = process.env.NEXT_PUBLIC_WS_URL;
      if (!wsBaseUrl) {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL;
        if (apiUrl) {
          wsBaseUrl = apiUrl.replace(/^http(s)?:\/\//i, (_match, s) => (s ? "wss://" : "ws://")).replace(/\/api\/?$/i, "");
        } else {
          const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
          const host = window.location.hostname || "localhost";
          wsBaseUrl = `${wsProtocol}//${host}:8000`;
        }
      }
      wsBaseUrl = wsBaseUrl.replace(/\/+$/, "");
      const wsUrl = `${wsBaseUrl}/ws/voice-stream?language_code=${encodeURIComponent(selectedLanguage)}&session_id=${encodeURIComponent(newSessionId)}`;


      vlog(`[WS] connecting to ${wsUrl}`);
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        if (activeSessionIdRef.current !== newSessionId) {
          vwarn(`[WS] ignoring onopen for obsolete session ${newSessionId}`);
          ws.close();
          return;
        }

        vlog(`[WS] connected — recording started`);
        setState("listening");

        // ScriptProcessor: converts Float32 to 16-bit linear PCM and streams continuously.
        // NOTE: ScriptProcessorNode is deprecated but remains functional in all current browsers.
        // Migration to AudioWorkletNode is tracked as a future improvement.
        const processor = audioCtx.createScriptProcessor(4096, 1, 1);
        processorRef.current = processor;

        processor.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return;
          const inputData = e.inputBuffer.getChannelData(0);

          // Diagnostic RMS (debug only — does NOT gate audio sending)
          if (DEBUG_VOICE) {
            let sumSq = 0;
            for (let i = 0; i < inputData.length; i++) sumSq += inputData[i] * inputData[i];
            const rms = Math.sqrt(sumSq / inputData.length);
            const chunkN = chunkCountRef.current + 1;
            if (chunkN === 1 || chunkN % 4 === 0 || rms > 0.005) {
              vlog(`[VAD] chunk #${chunkN} rms=${rms.toFixed(4)} speechDetected=${rms > 0.005}`);
            }
          }

          // Convert Float32Array to 16-bit linear PCM (Int16Array)
          const pcm16 = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }

          // Audio is ALWAYS sent regardless of VAD state
          ws.send(pcm16.buffer);
          chunkCountRef.current += 1;
          totalBytesRef.current += pcm16.buffer.byteLength;
        };

        sourceNode.connect(processor);
        processor.connect(audioCtx.destination);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.session_id && data.session_id !== activeSessionIdRef.current) {
            vwarn(`[WS] discarded stale message from session ${data.session_id}`);
            return;
          }

          vlog(`[WS] received message:`, data.type);

          if (data.type === "interim") {
            const transcript = data.transcript || "";
            vlog(`[WS] interim transcript: "${transcript}"`);
            setLiveTranscript(transcript);
            onInterimTranscript?.(transcript);
          } else if (data.type === "final_transcript") {
            const transcript = data.transcript || "";
            vlog(`[WS] final transcript: "${transcript}" (STT: ${data.stt_latency_ms} ms)`);
            setLiveTranscript(transcript);
            onFinalTranscript?.(transcript);
            setState("rag_processing");
          } else if (data.type === "vad") {
            vlog(`[WS] END_OF_SPEECH received`);
            setState("transcribing");
          } else if (data.type === "rag_response") {
            setState("answer_ready");
            onRAGResponse?.(data.data);
          } else if (data.type === "error") {
            console.error(`[WS] STT error:`, data.message);
            setState("error");
            onError?.(data.message || "Streaming voice processing failed.");
          }
        } catch (e) {
          console.error("Error parsing WebSocket message:", e);
        }
      };

      ws.onerror = (err) => {
        console.error("[WS] websocket error:", err);
        setState("error");
        onError?.("Voice streaming connection error.");
      };

      ws.onclose = () => {
        vlog(`[WS] websocket closed (session: ${newSessionId})`);
      };

    } catch (err: any) {
      // Microphone permission denied or device error — surface to user
      console.error("[VOICE] Microphone error:", err);
      setState("error");
      const userMsg = err.name === "NotAllowedError"
        ? "Microphone permission was denied. Please allow microphone access and try again."
        : err.name === "NotFoundError"
        ? "No microphone was found. Please connect a microphone and try again."
        : err.message || "Could not access microphone.";
      onError?.(userMsg);
    }
  }, [selectedLanguage, onInterimTranscript, onFinalTranscript, onRAGResponse, onError]);

  const stopStreaming = useCallback(() => {
    const finalDurationMs = Date.now() - startTimeRef.current;
    vlog(`[VOICE DEBUG] stopping: chunks=${chunkCountRef.current} bytes=${totalBytesRef.current} duration=${finalDurationMs}ms`);

    setState("transcribing");
    setVolume(0);

    // Disconnect ScriptProcessor first to stop audio processing
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    // Stop all microphone tracks
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    // Close AudioContext
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Signal backend to flush and finalize transcript
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "stop" }));
    }
  }, []);

  return {
    state,
    setState,
    volume,
    liveTranscript,
    startStreaming,
    stopStreaming
  };
}

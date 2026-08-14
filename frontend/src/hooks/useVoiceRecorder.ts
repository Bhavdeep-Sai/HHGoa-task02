"use client";

import { useState, useRef, useCallback } from "react";

export type RecordingState =
  | "idle"
  | "requesting_permission"
  | "listening"
  | "stopping"
  | "transcribing"
  | "transcript_ready"
  | "submitting"
  | "answer_ready"
  | "error";

export function useVoiceRecorder() {
  const [state, setState] = useState<RecordingState>("idle");
  const [volume, setVolume] = useState<number>(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const animFrameRef = useRef<number | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

  const startRecording = useCallback(async () => {
    try {
      setState("requesting_permission");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      // Audio volume visualizer analyzer
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const updateVolume = () => {
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          sum += dataArray[i];
        }
        const avg = sum / bufferLength;
        setVolume(Math.min(100, Math.round((avg / 128) * 100)));
        animFrameRef.current = requestAnimationFrame(updateVolume);
      };
      updateVolume();

      mediaRecorder.start(100);
      setState("listening");
    } catch (err) {
      console.error("Microphone permission denied or error:", err);
      setState("error");
    }
  }, []);

  const stopRecording = useCallback((): Promise<Blob> => {
    return new Promise((resolve) => {
      if (!mediaRecorderRef.current) {
        setState("idle");
        resolve(new Blob());
        return;
      }

      setState("stopping");
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close();
      }

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        // Stop all tracks
        mediaRecorderRef.current?.stream.getTracks().forEach((track) => track.stop());
        setVolume(0);
        resolve(audioBlob);
      };

      mediaRecorderRef.current.stop();
    });
  }, []);

  return {
    state,
    setState,
    volume,
    startRecording,
    stopRecording
  };
}


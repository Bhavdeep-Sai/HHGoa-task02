"use client";

import React from "react";
import { Mic, MicOff, Send, Loader2, Sparkles, Volume2 } from "lucide-react";

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

interface VoiceRecorderProps {
  state: RecordingState;
  volume: number;
  onStartRecording: () => void;
  onStopRecording: () => void;
  onTextSubmit: (text: string) => void;
  isLoading: boolean;
  voiceTranscript: string;
  textQuery: string;
  setTextQuery: (val: string) => void;
  inputMode: "text" | "voice";
}

export function VoiceRecorder({
  state,
  volume,
  onStartRecording,
  onStopRecording,
  onTextSubmit,
  isLoading,
  voiceTranscript,
  textQuery,
  setTextQuery,
  inputMode
}: VoiceRecorderProps) {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (textQuery.trim() && !isLoading) {
      onTextSubmit(textQuery.trim());
    }
  };

  const isListening = state === "listening" || state === "requesting_permission";
  const isTranscribing = state === "transcribing" || state === "stopping";

  return (
    <div className="bg-surface/70 border border-border rounded-3xl p-8 shadow-xl shadow-slate-100/20 dark:shadow-black/30 text-center max-w-2xl mx-auto my-6 space-y-6 transition-all duration-300">
      {/* State Animated Status Badge */}
      <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-bold border bg-bgDarker/60 border-border text-textSecondary shadow-sm transition-all duration-200">
        {isListening ? (
          <>
            <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
            <span className="text-red-500 font-semibold">Listening... Speak now</span>
          </>
        ) : isTranscribing ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin text-accent" />
            <span className="text-accent">Transcribing audio (Sarvam STT)...</span>
          </>
        ) : isLoading ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin text-accent" />
            <span className="text-accent">Retrieving evidence &amp; validating...</span>
          </>
        ) : (
          <>
            <Sparkles className="w-3.5 h-3.5 text-accent" />
            <span>Ready for Voice or Text Query</span>
          </>
        )}
      </div>

      {/* Display Spoken Voice Transcript immediately if available */}
      {inputMode === "voice" && voiceTranscript && (
        <div className="bg-accent/10 border border-accent/25 text-accent px-4 py-3 rounded-2xl text-sm font-semibold flex items-center justify-center gap-2 shadow-sm animate-in fade-in slide-in-from-top-2 duration-300">
          <Volume2 className="w-4 h-4 shrink-0" />
          <span>Transcribed Question: &ldquo;{voiceTranscript}&rdquo;</span>
        </div>
      )}

      {/* Large Microphone Button */}
      <div className="relative flex justify-center items-center my-6">
        {/* Pulsing Audio Ripples */}
        {isListening ? (
          <div
            className="absolute rounded-full bg-red-500/20 dark:bg-red-500/10 border border-red-500/30 animate-pulse transition-all duration-75"
            style={{
              width: `${110 + volume * 1.4}px`,
              height: `${110 + volume * 1.4}px`,
            }}
          />
        ) : (
          /* Silent breathing glow ring */
          <div className="absolute w-[110px] h-[110px] rounded-full border border-accent/20 bg-accent/5 animate-pulse -z-0" />
        )}
        <button
          onClick={isListening ? onStopRecording : onStartRecording}
          disabled={isLoading || isTranscribing}
          className={`relative z-10 w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 shadow-xl ${
            isListening
              ? "bg-red-500 hover:bg-red-600 text-white ring-4 ring-red-500/20 scale-105"
              : "bg-gradient-to-br from-orange-500 to-orange-600 hover:scale-105 text-white shadow-lg shadow-orange-500/20 hover:shadow-orange-500/35 disabled:opacity-50 cursor-pointer"
          }`}
        >
          {isListening ? <MicOff className="w-8 h-8" /> : <Mic className="w-8 h-8" />}
        </button>
      </div>

      <p className="text-xs text-textTertiary font-medium tracking-wide transition-all">
        {isListening ? "Click microphone again to stop & process audio" : "Click microphone to speak or type question below"}
      </p>

      {/* Integrated Capsule Text Form */}
      <form onSubmit={handleSubmit} className="max-w-lg mx-auto pt-2">
        <div className="relative flex items-center bg-bgDarker/60 border border-border rounded-full p-1.5 focus-within:ring-4 focus-within:ring-accent/15 focus-within:border-accent shadow-inner hover:border-border/80 transition-all duration-300">
          <input
            type="text"
            value={textQuery}
            onChange={(e) => setTextQuery(e.target.value)}
            placeholder="Or type question in Hindi, Telugu, Tamil, English..."
            disabled={isLoading || isListening || isTranscribing}
            className="flex-1 bg-transparent px-4 py-2 text-sm text-textPrimary placeholder-textTertiary/60 focus:outline-none disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isLoading || !textQuery.trim() || isListening || isTranscribing}
            className="bg-orange-500 hover:bg-orange-600 disabled:bg-textTertiary/20 text-white font-medium w-9 h-9 rounded-full flex items-center justify-center disabled:opacity-40 transition-all duration-200 shadow-md cursor-pointer"
            title="Send query"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </form>
    </div>
  );
}

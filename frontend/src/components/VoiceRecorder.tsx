"use client";

import React from "react";
import { Mic, MicOff, Send, Loader2, Sparkles, Volume2 } from "lucide-react";

// States passed in from page.tsx after mapping StreamingState → UI state
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
    <div className="bg-surface/90 border border-gray-800 rounded-2xl p-6 shadow-xl backdrop-blur-md text-center max-w-2xl mx-auto my-6 space-y-4">
      {/* State Animated Status Badge */}
      <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold border bg-gray-900 border-gray-700 text-gray-300">
        {isListening ? (
          <>
            <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
            <span className="text-red-400">Listening... Speak now</span>
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
        <div className="bg-accent/10 border border-accent/30 text-accent px-4 py-2 rounded-xl text-sm font-medium flex items-center justify-center gap-2 animate-in fade-in">
          <Volume2 className="w-4 h-4" />
          <span>Transcribed Question: &ldquo;{voiceTranscript}&rdquo;</span>
        </div>
      )}

      {/* Large Microphone Button */}
      <div className="relative flex justify-center items-center my-4">
        {isListening && (
          <div
            className="absolute rounded-full bg-accent/30 transition-all duration-75"
            style={{
              width: `${120 + volume * 1.2}px`,
              height: `${120 + volume * 1.2}px`,
            }}
          />
        )}
        <button
          onClick={isListening ? onStopRecording : onStartRecording}
          disabled={isLoading || isTranscribing}
          className={`relative z-10 w-24 h-24 rounded-full flex items-center justify-center transition-all duration-200 shadow-2xl ${
            isListening
              ? "bg-red-600 hover:bg-red-700 text-white ring-4 ring-red-500/50 scale-105"
              : "bg-gradient-to-br from-accent to-orange-600 hover:scale-105 text-white shadow-accentGlow disabled:opacity-50"
          }`}
        >
          {isListening ? <MicOff className="w-10 h-10" /> : <Mic className="w-10 h-10" />}
        </button>
      </div>

      <p className="text-xs text-gray-400">
        {isListening ? "Click microphone again to stop & process audio" : "Click microphone to speak or type question below"}
      </p>

      {/* Text Form */}
      <form onSubmit={handleSubmit} className="flex gap-2 max-w-lg mx-auto pt-2">
        <input
          type="text"
          value={textQuery}
          onChange={(e) => setTextQuery(e.target.value)}
          placeholder="Or type question in Hindi, Telugu, Tamil, English, Code-Mixed..."
          disabled={isLoading || isListening || isTranscribing}
          className="flex-1 bg-surfaceHover border border-gray-700 rounded-xl px-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={isLoading || !textQuery.trim() || isListening || isTranscribing}
          className="bg-accent hover:bg-orange-600 text-white font-medium px-4 py-2.5 rounded-xl text-sm flex items-center gap-1.5 disabled:opacity-50 transition-all"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}


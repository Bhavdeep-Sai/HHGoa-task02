"use client";

import React from "react";
import { Mic, MicOff, Send, Loader2, Sparkles, Volume2, Server, AlertCircle, RefreshCw, Radio } from "lucide-react";
import { BackendConnectionStatus } from "../hooks/useBackendStatus";

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
  backendStatus?: BackendConnectionStatus;
  elapsedSeconds?: number;
  onRetryBackend?: () => void;
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
  inputMode,
  backendStatus = "checking",
  elapsedSeconds = 0,
  onRetryBackend
}: VoiceRecorderProps) {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (textQuery.trim() && !isLoading && backendStatus === "connected") {
      onTextSubmit(textQuery.trim());
    }
  };

  const isConnected = backendStatus === "connected";
  const isWakingUp = backendStatus === "checking";
  const isDisconnected = backendStatus === "disconnected";

  const isListening = state === "listening" || state === "requesting_permission";
  const isTranscribing = state === "transcribing" || state === "stopping";
  const isMicDisabled = !isConnected || isLoading || isTranscribing;
  const isInputDisabled = !isConnected || isLoading || isListening || isTranscribing;

  return (
    <div className="bg-surface/70 border border-border rounded-3xl p-8 shadow-xl shadow-slate-100/20 dark:shadow-black/30 text-center max-w-2xl mx-auto my-6 space-y-6 transition-all duration-300">
      {/* State Animated Status Badge */}
      <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-bold border bg-bgDarker/60 border-border text-textSecondary shadow-sm transition-all duration-200">
        {isWakingUp ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-500" />
            <span className="text-amber-500 font-semibold">
              Waking up backend server... {elapsedSeconds > 0 ? `(${elapsedSeconds}s)` : ""}
            </span>
          </>
        ) : isDisconnected ? (
          <>
            <AlertCircle className="w-3.5 h-3.5 text-red-500" />
            <span className="text-red-500 font-semibold">Backend Unreachable</span>
          </>
        ) : isListening ? (
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

      {/* Backend Wake-up Card Banner */}
      {isWakingUp && (
        <div className="bg-amber-500/10 border border-amber-500/25 dark:bg-amber-950/40 dark:border-amber-800/40 rounded-2xl p-4 text-left shadow-sm space-y-2 animate-in fade-in zoom-in-95 duration-300">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="relative flex items-center justify-center">
                <Server className="w-5 h-5 text-amber-500 animate-pulse" />
                <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-amber-400 animate-ping" />
              </div>
              <span className="text-sm font-bold text-amber-700 dark:text-amber-300">
                Waking Up Cloud Server ({elapsedSeconds}s)
              </span>
            </div>
            <div className="flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400 font-mono font-medium">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span>Connecting...</span>
            </div>
          </div>
          <p className="text-xs text-textSecondary leading-relaxed">
            Free-tier cloud backend spins down on inactivity. Cold start takes approximately <strong>30–50 seconds</strong>. Speech and text input will unlock automatically once live.
          </p>
          {/* Animated Progress Bar shimmer */}
          <div className="w-full bg-amber-500/20 rounded-full h-1.5 overflow-hidden">
            <div className="bg-gradient-to-r from-amber-500 to-orange-500 h-full w-2/3 rounded-full animate-[pulse_1.5s_ease-in-out_infinite]" />
          </div>
        </div>
      )}

      {/* Backend Disconnected / Error Banner with Retry */}
      {isDisconnected && (
        <div className="bg-red-500/10 border border-red-500/25 dark:bg-red-950/40 dark:border-red-800/40 rounded-2xl p-4 text-left shadow-sm space-y-3 animate-in fade-in zoom-in-95 duration-300">
          <div className="flex items-center gap-2.5">
            <AlertCircle className="w-5 h-5 text-red-500 shrink-0" />
            <div>
              <h4 className="text-sm font-bold text-red-600 dark:text-red-300">
                Backend Connection Timeout
              </h4>
              <p className="text-xs text-textSecondary mt-0.5">
                Could not connect to backend server. The server might still be waking up or restarting.
              </p>
            </div>
          </div>
          <div className="flex justify-end">
            <button
              onClick={onRetryBackend}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold bg-accent hover:bg-accent/90 text-white shadow-md transition-all cursor-pointer"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Wake Up / Retry Connection</span>
            </button>
          </div>
        </div>
      )}

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
        ) : isConnected ? (
          /* Silent breathing glow ring */
          <div className="absolute w-[110px] h-[110px] rounded-full border border-accent/20 bg-accent/5 animate-pulse -z-0" />
        ) : null}

        <button
          onClick={isListening ? onStopRecording : onStartRecording}
          disabled={isMicDisabled}
          aria-label="Voice Query Microphone"
          title={
            !isConnected
              ? `Waiting for backend server to wake up (${elapsedSeconds}s)...`
              : isListening
              ? "Click to stop recording"
              : "Click to speak"
          }
          className={`relative z-10 w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 shadow-xl ${
            isListening
              ? "bg-red-500 hover:bg-red-600 text-white ring-4 ring-red-500/20 scale-105"
              : !isConnected
              ? "bg-bgDarker/80 text-textTertiary/50 border border-border/80 cursor-not-allowed opacity-60"
              : "bg-gradient-to-br from-orange-500 to-orange-600 hover:scale-105 text-white shadow-lg shadow-orange-500/20 hover:shadow-orange-500/35 disabled:opacity-50 cursor-pointer"
          }`}
        >
          {isWakingUp ? (
            <Loader2 className="w-8 h-8 animate-spin text-amber-500/80" />
          ) : isListening ? (
            <MicOff className="w-8 h-8" />
          ) : (
            <Mic className="w-8 h-8" />
          )}
        </button>
      </div>

      <p className="text-xs text-textTertiary font-medium tracking-wide transition-all">
        {!isConnected
          ? `Microphone disabled while server is waking up (${elapsedSeconds}s)`
          : isListening
          ? "Click microphone again to stop & process audio"
          : "Click microphone to speak or type question below"}
      </p>

      {/* Integrated Capsule Text Form */}
      <form onSubmit={handleSubmit} className="max-w-lg mx-auto pt-2">
        <div className="relative flex items-center bg-bgDarker/60 border border-border rounded-full p-1.5 focus-within:ring-4 focus-within:ring-accent/15 focus-within:border-accent shadow-inner hover:border-border/80 transition-all duration-300">
          <input
            type="text"
            value={textQuery}
            onChange={(e) => setTextQuery(e.target.value)}
            placeholder={
              !isConnected
                ? `Waking up backend server (${elapsedSeconds}s)... Text input locked`
                : "Or type question in Hindi, Telugu, Tamil, English..."
            }
            disabled={isInputDisabled}
            className="flex-1 bg-transparent px-4 py-2 text-sm text-textPrimary placeholder-textTertiary/60 focus:outline-none disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isInputDisabled || !textQuery.trim()}
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

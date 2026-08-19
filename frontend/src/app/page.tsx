"use client";

import React, { useState, useEffect } from "react";
import { Header } from "../components/Header";
import { VoiceRecorder } from "../components/VoiceRecorder";
import { AnswerCard } from "../components/AnswerCard";
import { EvidenceAccordion } from "../components/EvidenceAccordion";
import { LatencyMetrics } from "../components/LatencyMetrics";
import { DebugPanel } from "../components/DebugPanel";
import { useStreamingVoiceRecorder } from "../hooks/useStreamingVoiceRecorder";
import { useBackendStatus } from "../hooks/useBackendStatus";
import { sendTextQuery, RAGPipelineResponse } from "../lib/api";

export default function Home() {
  const [debugMode, setDebugMode] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("light");

  // Backend connection status & cold-start wake-up monitor
  const {
    status: backendStatus,
    elapsedSeconds,
    latencyMs,
    isConnected,
    retryConnection
  } = useBackendStatus();

  // Load and apply theme
  useEffect(() => {
    const savedTheme = localStorage.getItem("theme") as "light" | "dark";
    if (savedTheme) {
      setTheme(savedTheme);
      document.documentElement.className = savedTheme;
    } else {
      setTheme("light");
      document.documentElement.className = "light";
    }
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "light" ? "dark" : "light";
    setTheme(nextTheme);
    localStorage.setItem("theme", nextTheme);
    document.documentElement.className = nextTheme;
  };
  const [selectedLanguage, setSelectedLanguage] = useState("unknown");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [response, setResponse] = useState<RAGPipelineResponse | null>(null);

  // Distinct query states per input mode
  const [textQuery, setTextQuery] = useState("");
  const [voiceTranscript, setVoiceTranscript] = useState("");
  const [inputMode, setInputMode] = useState<"text" | "voice">("text");

  const {
    state,
    volume,
    liveTranscript,
    startStreaming,
    stopStreaming
  } = useStreamingVoiceRecorder({
    selectedLanguage,
    onInterimTranscript: (interim) => {
      setVoiceTranscript(interim);
    },
    onFinalTranscript: (final) => {
      setVoiceTranscript(final);
    },
    onRAGResponse: (ragRes) => {
      setResponse(ragRes);
      setIsLoading(false);
    },
    onError: (err) => {
      setErrorMsg(err);
      setIsLoading(false);
    }
  });

  const handleStartVoiceRecording = async () => {
    if (!isConnected) {
      setErrorMsg("Please wait for the backend server to finish waking up.");
      return;
    }
    setVoiceTranscript("");
    setResponse(null);
    setErrorMsg(null);
    setInputMode("voice");
    await startStreaming();
  };

  const handleStopRecordingAndSubmit = () => {
    setIsLoading(true);
    stopStreaming();
  };

  const handleTextQuerySubmit = async (queryText: string) => {
    if (!isConnected) {
      setErrorMsg("Backend server is not connected yet. Please wait a moment.");
      return;
    }
    try {
      setInputMode("text");
      setTextQuery(queryText);
      setVoiceTranscript("");
      setIsLoading(true);
      setErrorMsg(null);
      setResponse(null);

      const result = await sendTextQuery(queryText);
      setResponse(result);
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to process query");
      // If network failed, trigger backend status check
      if (err.message && (err.message.includes("Failed to fetch") || err.message.includes("NetworkError"))) {
        retryConnection();
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Map streaming recorder state to UI recording state
  const mappedState: any = state === "connecting"
    ? "requesting_permission"
    : state === "listening"
    ? "listening"
    : state === "transcribing" || state === "rag_processing"
    ? "transcribing"
    : state === "answer_ready"
    ? "answer_ready"
    : state === "error"
    ? "error"
    : "idle";

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Background Mesh Gradient Blobs */}
      <div className="absolute top-[-10%] left-[-15%] w-[500px] h-[500px] rounded-full bg-gradient-to-br from-accent/20 to-orange-400/10 blur-[120px] -z-10 pointer-events-none transition-all duration-300" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[600px] h-[600px] rounded-full bg-gradient-to-br from-blue-400/10 to-indigo-500/10 blur-[130px] -z-10 pointer-events-none transition-all duration-300" />

      <Header
        debugMode={debugMode}
        setDebugMode={setDebugMode}
        selectedLanguage={selectedLanguage}
        setSelectedLanguage={setSelectedLanguage}
        theme={theme}
        toggleTheme={toggleTheme}
        backendStatus={backendStatus}
        elapsedSeconds={elapsedSeconds}
        latencyMs={latencyMs}
        onRetryBackend={retryConnection}
      />

      <main className="flex-1 max-w-5xl w-full mx-auto px-4 py-8 space-y-6">
        {/* Voice Recorder & Form */}
        <VoiceRecorder
          state={mappedState}
          volume={volume}
          onStartRecording={handleStartVoiceRecording}
          onStopRecording={handleStopRecordingAndSubmit}
          onTextSubmit={handleTextQuerySubmit}
          isLoading={isLoading || state === "rag_processing"}
          voiceTranscript={liveTranscript || voiceTranscript}
          textQuery={textQuery}
          setTextQuery={setTextQuery}
          inputMode={inputMode}
          backendStatus={backendStatus}
          elapsedSeconds={elapsedSeconds}
          onRetryBackend={retryConnection}
        />

        {/* Error Banner */}
        {errorMsg && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-600 dark:bg-red-950/80 dark:border-red-800 dark:text-red-300 p-4 rounded-xl text-sm text-center font-medium shadow-lg">
            {errorMsg}
          </div>
        )}

        {/* Results View */}
        {response && (
          <div className="space-y-6 animate-in fade-in duration-300">
            <AnswerCard response={response} />
            <LatencyMetrics
              stages={response.stage_latencies}
              fastPathUsed={response.fast_path_used}
            />
            <EvidenceAccordion evidence={response.evidence} />
            {debugMode && <DebugPanel response={response} />}
          </div>
        )}
      </main>

      <footer className="border-t border-border/80 text-center py-4 text-xs text-textTertiary">
        HH Goa 2026 Task 2 — IndicVoiceRAG (Adaptive Multilingual Voice RAG Model)
      </footer>
    </div>
  );
}

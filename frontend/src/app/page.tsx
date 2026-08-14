"use client";

import React, { useState } from "react";
import { Header } from "../components/Header";
import { VoiceRecorder } from "../components/VoiceRecorder";
import { AnswerCard } from "../components/AnswerCard";
import { EvidenceAccordion } from "../components/EvidenceAccordion";
import { LatencyMetrics } from "../components/LatencyMetrics";
import { DebugPanel } from "../components/DebugPanel";
import { useStreamingVoiceRecorder } from "../hooks/useStreamingVoiceRecorder";
import { sendTextQuery, RAGPipelineResponse } from "../lib/api";

export default function Home() {
  const [debugMode, setDebugMode] = useState(false);
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
    <div className="min-h-screen flex flex-col">
      <Header
        debugMode={debugMode}
        setDebugMode={setDebugMode}
        selectedLanguage={selectedLanguage}
        setSelectedLanguage={setSelectedLanguage}
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
        />

        {/* Error Banner */}
        {errorMsg && (
          <div className="bg-red-950/80 border border-red-800 text-red-300 p-4 rounded-xl text-sm text-center font-medium shadow-lg">
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

      <footer className="border-t border-gray-800/80 text-center py-4 text-xs text-gray-500">
        HH Goa 2026 Task 2 — IndicVoiceRAG (Adaptive Multilingual Voice RAG Model)
      </footer>
    </div>
  );
}

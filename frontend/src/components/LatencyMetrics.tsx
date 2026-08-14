"use client";

import React from "react";
import { PipelineStageMetrics } from "../lib/api";
import { Timer } from "lucide-react";

interface LatencyMetricsProps {
  stages: PipelineStageMetrics;
  fastPathUsed: boolean;
}

export function LatencyMetrics({ stages, fastPathUsed }: LatencyMetricsProps) {
  const isVoice = stages.stt_latency_ms > 0;
  const ragLatency = stages.rag_latency_ms || (stages.query_norm_latency_ms + stages.retrieval_latency_ms + stages.generation_latency_ms + (stages.grounding_ms || 0));
  const totalVoice = stages.voice_to_answer_latency_ms || (stages.stt_latency_ms + ragLatency);
  const total = isVoice ? totalVoice : (stages.total_latency_ms || ragLatency || 1);

  const getPct = (val: number) => Math.min(100, Math.max(2, Math.round((val / total) * 100)));

  return (
    <div className="bg-surface border border-border rounded-2xl p-5 shadow-lg space-y-4 transition-all">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h3 className="text-sm font-semibold text-textPrimary flex items-center gap-2 transition-all">
          <Timer className="w-4 h-4 text-accent" />
          Pipeline Latency Breakdown
        </h3>
        <div className="flex gap-2 text-xs font-bold flex-wrap">
          <div className={`px-3 py-1 rounded-lg border transition-all ${
            ragLatency <= 200 
              ? "text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 dark:bg-emerald-950/80 border-emerald-500/20 dark:border-emerald-800" 
              : "text-amber-600 dark:text-amber-400 bg-amber-500/10 dark:bg-amber-950/80 border-amber-500/20 dark:border-amber-800"
          }`}>
            RAG Pipeline: {ragLatency.toFixed(2)} ms ({ragLatency <= 200 ? "Under target" : `Above target by ${(ragLatency - 200).toFixed(2)} ms`})
          </div>
          {isVoice && (
            <div className={`px-3 py-1 rounded-lg border transition-all ${
              totalVoice <= 500 
                ? "text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 dark:bg-emerald-950/80 border-emerald-500/20 dark:border-emerald-800" 
                : "text-blue-600 dark:text-blue-400 bg-blue-500/10 dark:bg-blue-950/80 border-blue-500/20 dark:border-blue-800"
            }`}>
              Voice-to-Answer: {totalVoice.toFixed(2)} ms ({totalVoice <= 500 ? "Under target" : `Above target by ${(totalVoice - 500).toFixed(2)} ms`})
            </div>
          )}
        </div>
      </div>

      {/* Latency Bar Stack */}
      <div className="h-3 w-full bg-bgDarker rounded-full overflow-hidden flex gap-0.5 transition-all">
        {isVoice && (
          <div
            style={{ width: `${getPct(stages.stt_latency_ms)}%` }}
            className="bg-blue-500 h-full"
            title={`STT: ${stages.stt_latency_ms}ms`}
          />
        )}
        <div
          style={{ width: `${getPct(stages.query_norm_latency_ms)}%` }}
          className="bg-indigo-500 h-full"
          title={`Query Norm: ${stages.query_norm_latency_ms}ms`}
        />
        <div
          style={{ width: `${getPct(stages.retrieval_latency_ms)}%` }}
          className="bg-amber-500 h-full"
          title={`Retrieval: ${stages.retrieval_latency_ms}ms`}
        />
        <div
          style={{ width: `${getPct(stages.generation_latency_ms)}%` }}
          className="bg-emerald-500 h-full"
          title={`Generation: ${stages.generation_latency_ms}ms`}
        />
        {(stages.grounding_ms || 0) > 0 && (
          <div
            style={{ width: `${getPct(stages.grounding_ms || 0)}%` }}
            className="bg-purple-500 h-full"
            title={`Grounding: ${stages.grounding_ms}ms`}
          />
        )}
      </div>

      {/* Individual Stage Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        <div className="bg-surfaceHover/50 p-2.5 rounded-xl border border-border transition-all">
          <span className="text-textTertiary text-[11px] block transition-all">STT Latency</span>
          <span className="text-sm font-semibold text-blue-600 dark:text-blue-400 transition-all">
            {isVoice ? `${stages.stt_latency_ms.toFixed(2)} ms` : "0 ms (Typed Query)"}
          </span>
        </div>
        <div className="bg-surfaceHover/50 p-2.5 rounded-xl border border-border transition-all">
          <span className="text-textTertiary text-[11px] block transition-all">Retrieval Latency</span>
          <span className="text-sm font-semibold text-amber-600 dark:text-amber-400 transition-all">{stages.retrieval_latency_ms.toFixed(2)} ms</span>
        </div>
        <div className="bg-surfaceHover/50 p-2.5 rounded-xl border border-border transition-all">
          <span className="text-textTertiary text-[11px] block transition-all">Generation Latency</span>
          <span className="text-sm font-semibold text-emerald-600 dark:text-emerald-400 transition-all">{stages.generation_latency_ms.toFixed(2)} ms</span>
        </div>
        <div className="bg-surfaceHover/50 p-2.5 rounded-xl border border-border transition-all">
          <span className="text-textTertiary text-[11px] block transition-all">
            {isVoice ? "Total Voice-to-Answer" : "RAG Total Latency"}
          </span>
          <span className="text-sm font-semibold text-textPrimary transition-all">{total.toFixed(2)} ms</span>
        </div>
      </div>
    </div>
  );
}



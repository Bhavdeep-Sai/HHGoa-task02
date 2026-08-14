"use client";

import React from "react";
import { RAGPipelineResponse } from "../lib/api";
import { Bug, Terminal, Code2 } from "lucide-react";

interface DebugPanelProps {
  response: RAGPipelineResponse;
}

export function DebugPanel({ response }: DebugPanelProps) {
  return (
    <div className="bg-surface/95 border border-accent/40 rounded-2xl p-6 shadow-2xl space-y-4">
      <div className="flex items-center gap-2 text-accent font-semibold text-sm border-b border-gray-800 pb-3">
        <Terminal className="w-4 h-4" />
        <span>Developer Mode — Execution Diagnostics</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        <div className="bg-surfaceHover p-3 rounded-xl border border-gray-800">
          <span className="text-gray-400 block">Detected Language</span>
          <p className="text-sm font-semibold text-white uppercase">{response.detected_language}</p>
        </div>
        <div className="bg-surfaceHover p-3 rounded-xl border border-gray-800">
          <span className="text-gray-400 block">Code-Mixed Query</span>
          <p className="text-sm font-semibold text-white">{response.is_code_mixed ? "Yes" : "No"}</p>
        </div>
        <div className="bg-surfaceHover p-3 rounded-xl border border-gray-800">
          <span className="text-gray-400 block">Adaptive Reranker</span>
          <p className="text-sm font-semibold text-white">{response.reranker_used ? "Executed" : "Skipped (High Conf)"}</p>
        </div>
        <div className="bg-surfaceHover p-3 rounded-xl border border-gray-800">
          <span className="text-gray-400 block">Fast Path Router</span>
          <p className="text-sm font-semibold text-emerald-400">{response.fast_path_used ? "Active (<50ms)" : "Standard Synthesis"}</p>
        </div>
      </div>

      {/* Granular Sub-Stage Diagnostics Panel */}
      <div className="bg-surfaceHover/60 p-4 rounded-xl border border-gray-800 space-y-2">
        <p className="text-xs font-semibold text-accent flex items-center gap-1.5">
          <Bug className="w-3.5 h-3.5" /> Granular Stage Timing Diagnostics (Target &lt; 200 ms)
        </p>
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 text-[11px]">
          <div className="bg-gray-900/80 p-2 rounded-lg border border-gray-800">
            <span className="text-gray-400 block">Embedding</span>
            <span className="font-mono font-semibold text-indigo-400">{response.stage_latencies.embedding_ms ?? 0} ms</span>
          </div>
          <div className="bg-gray-900/80 p-2 rounded-lg border border-gray-800">
            <span className="text-gray-400 block">Dense ANN</span>
            <span className="font-mono font-semibold text-amber-400">{response.stage_latencies.dense_search_ms ?? 0} ms</span>
          </div>
          <div className="bg-gray-900/80 p-2 rounded-lg border border-gray-800">
            <span className="text-gray-400 block">BM25</span>
            <span className="font-mono font-semibold text-orange-400">{response.stage_latencies.bm25_ms ?? 0} ms</span>
          </div>
          <div className="bg-gray-900/80 p-2 rounded-lg border border-gray-800">
            <span className="text-gray-400 block">RRF</span>
            <span className="font-mono font-semibold text-purple-400">{response.stage_latencies.rrf_ms ?? 0} ms</span>
          </div>
          <div className="bg-gray-900/80 p-2 rounded-lg border border-gray-800">
            <span className="text-gray-400 block">Reranker</span>
            <span className="font-mono font-semibold text-pink-400">{response.stage_latencies.reranker_ms ?? 0} ms</span>
          </div>
          <div className="bg-gray-900/80 p-2 rounded-lg border border-gray-800">
            <span className="text-gray-400 block">Grounding</span>
            <span className="font-mono font-semibold text-cyan-400">{response.stage_latencies.grounding_ms ?? 0} ms</span>
          </div>
        </div>
      </div>

      <div>
        <p className="text-xs font-semibold text-gray-400 mb-1 flex items-center gap-1">
          <Code2 className="w-3.5 h-3.5 text-accent" /> Raw Pipeline Diagnostics Payload
        </p>
        <pre className="bg-black/90 p-4 rounded-xl text-[11px] font-mono text-emerald-400 overflow-x-auto max-h-60 border border-gray-800">
          {JSON.stringify(response, null, 2)}
        </pre>
      </div>
    </div>
  );
}

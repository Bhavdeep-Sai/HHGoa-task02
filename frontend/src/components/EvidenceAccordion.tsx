"use client";

import React, { useState } from "react";
import { EvidenceItem } from "../lib/api";
import { ChevronDown, ChevronUp, FileText, Layers } from "lucide-react";

interface EvidenceAccordionProps {
  evidence: EvidenceItem[];
}

export function EvidenceAccordion({ evidence }: EvidenceAccordionProps) {
  const [isOpen, setIsOpen] = useState(true);

  if (!evidence || evidence.length === 0) return null;

  return (
    <div className="bg-surface border border-gray-800 rounded-2xl p-5 shadow-lg">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between text-left text-sm font-semibold text-gray-200"
      >
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-accent" />
          <span>Retrieved Context Evidence ({evidence.length} Passages)</span>
        </div>
        {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {isOpen && (
        <div className="mt-4 space-y-3">
          {evidence.map((item, idx) => (
            <div
              key={idx}
              className="bg-surfaceHover/80 border border-gray-800 rounded-xl p-4 text-xs space-y-2 transition-all hover:border-gray-700"
            >
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-semibold text-accent uppercase tracking-wider text-[11px]">
                    Passage #{idx + 1}
                  </span>
                  <span className="bg-emerald-950/80 text-emerald-300 border border-emerald-800 px-2 py-0.5 rounded text-[10px] font-mono">
                    ai4bharat/MSMARCO-XI
                  </span>
                  {item.relevance_signals?.query_id && (
                    <span className="bg-gray-800 text-gray-300 px-2 py-0.5 rounded text-[10px] font-mono">
                      QID: {item.relevance_signals.query_id}
                    </span>
                  )}
                  <span className="bg-gray-800 text-gray-300 px-2 py-0.5 rounded text-[10px] font-mono">
                    {item.chunk_type}
                  </span>
                  <span className="bg-gray-800 text-gray-300 px-2 py-0.5 rounded text-[10px] uppercase">
                    {item.language}
                  </span>
                </div>
                <span className="text-gray-400 font-mono text-[11px]">
                  Relevance Score: <strong className="text-white">{item.score}</strong>
                </span>
              </div>
              <p className="text-gray-300 leading-relaxed font-sans">{item.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

"use client";

import React, { useState } from "react";
import { EvidenceItem } from "../lib/api";
import { ChevronDown, ChevronUp, Layers } from "lucide-react";

interface EvidenceAccordionProps {
  evidence: EvidenceItem[];
}

export function EvidenceAccordion({ evidence }: EvidenceAccordionProps) {
  const [isOpen, setIsOpen] = useState(true);

  if (!evidence || evidence.length === 0) return null;

  return (
    <div className="bg-surface border border-border rounded-2xl p-5 shadow-lg transition-all">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between text-left text-sm font-semibold text-textPrimary transition-all"
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
              className="bg-surfaceHover/80 border border-border rounded-xl p-4 text-xs space-y-2 transition-all hover:border-accent/40"
            >
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-semibold text-accent uppercase tracking-wider text-[11px]">
                    Passage #{idx + 1}
                  </span>
                  <span className="bg-emerald-500/10 dark:bg-emerald-950/80 text-emerald-600 dark:text-emerald-300 border border-emerald-500/20 dark:border-emerald-800 px-2 py-0.5 rounded text-[10px] font-mono transition-all">
                    ai4bharat/MSMARCO-XI
                  </span>
                  {item.relevance_signals?.query_id && (
                    <span className="bg-bgDarker text-textSecondary border border-border px-2 py-0.5 rounded text-[10px] font-mono transition-all">
                      QID: {item.relevance_signals.query_id}
                    </span>
                  )}
                  <span className="bg-bgDarker text-textSecondary border border-border px-2 py-0.5 rounded text-[10px] font-mono transition-all">
                    {item.chunk_type}
                  </span>
                  <span className="bg-bgDarker text-textSecondary border border-border px-2 py-0.5 rounded text-[10px] uppercase transition-all">
                    {item.language}
                  </span>
                </div>
                <span className="text-textTertiary font-mono text-[11px] transition-all">
                  Relevance Score: <strong className="text-textPrimary">{item.score}</strong>
                </span>
              </div>
              <p className="text-textSecondary leading-relaxed font-sans transition-all">{item.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


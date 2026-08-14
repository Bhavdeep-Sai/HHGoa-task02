"use client";

import React from "react";
import { RAGPipelineResponse } from "../lib/api";
import { CheckCircle2, AlertTriangle, ShieldCheck, Zap, Globe, HelpCircle, ShieldAlert } from "lucide-react";

interface AnswerCardProps {
  response: RAGPipelineResponse;
}

export function AnswerCard({ response }: AnswerCardProps) {
  const isGrounded = response.grounded && response.confidence > 0;
  const isCasual = response.intent === "casual" || response.intent === "off_topic";
  const isSecurity = response.intent === "prompt_injection" || response.intent === "unsafe";

  const confidencePct = isGrounded ? Math.round(response.confidence * 100) : 0;

  // Header Title & Icon logic
  let headerTitle = "Grounded Answer";
  let HeaderIcon = ShieldCheck;
  let headerColor = "text-accent";

  if (isSecurity) {
    headerTitle = "Request Rejected";
    HeaderIcon = ShieldAlert;
    headerColor = "text-red-600 dark:text-red-400";
  } else if (isCasual) {
    headerTitle = "Off-Topic Query";
    HeaderIcon = HelpCircle;
    headerColor = "text-amber-600 dark:text-amber-400";
  } else if (!isGrounded) {
    headerTitle = "Insufficient Evidence";
    HeaderIcon = AlertTriangle;
    headerColor = "text-amber-600 dark:text-amber-400";
  }

  return (
    <div className="bg-surface/70 border border-border rounded-3xl p-8 shadow-xl shadow-slate-100/20 dark:shadow-black/30 space-y-6 transition-all duration-300">
      {/* Top Bar: Language & Badges */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/40 pb-4 transition-all">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 bg-bgDarker/60 border border-border px-3.5 py-1.5 rounded-full text-xs font-semibold text-textSecondary transition-all">
            <Globe className="w-3.5 h-3.5 text-accent" />
            <span className="capitalize">{response.detected_language}</span>
            {response.is_code_mixed && (
              <span className="text-[10px] bg-accent/15 text-accent px-2 py-0.5 rounded-full font-bold">
                Code-Mixed
              </span>
            )}
          </div>

          {response.fast_path_used && (
            <div className="flex items-center gap-1 bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 px-3 py-1.5 rounded-full text-xs font-bold transition-all">
              <Zap className="w-3.5 h-3.5" />
              <span>Fast Path (&lt;50ms)</span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Grounding / State Tag */}
          {isGrounded ? (
            <div className="flex items-center gap-1.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 px-4 py-1.5 rounded-full text-xs font-bold transition-all shadow-sm">
              <CheckCircle2 className="w-4 h-4 text-emerald-555" />
              <span>Grounded ✓</span>
            </div>
          ) : isCasual ? (
            <div className="flex items-center gap-1.5 bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400 px-4 py-1.5 rounded-full text-xs font-bold transition-all shadow-sm">
              <HelpCircle className="w-4 h-4 text-amber-555" />
              <span>Off-Topic ⚠️</span>
            </div>
          ) : isSecurity ? (
            <div className="flex items-center gap-1.5 bg-red-500/10 border border-red-500/20 text-red-600 dark:text-red-400 px-4 py-1.5 rounded-full text-xs font-bold transition-all shadow-sm">
              <ShieldAlert className="w-4 h-4 text-red-500 dark:text-red-400" />
              <span>Security Guardrail</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400 px-4 py-1.5 rounded-full text-xs font-bold transition-all shadow-sm">
              <AlertTriangle className="w-4 h-4 text-amber-555" />
              <span>Insufficient Evidence</span>
            </div>
          )}

          {/* Confidence Badge */}
          <div className="bg-surface/50 border border-border px-3.5 py-1.5 rounded-full text-xs font-extrabold text-textPrimary transition-all shadow-sm">
            {confidencePct}% Confidence
          </div>
        </div>
      </div>

      {/* User Transcript */}
      <div>
        <p className="text-xs font-bold text-textTertiary uppercase tracking-wider mb-2 transition-all">
          Transcribed User Question
        </p>
        <p className="text-base text-textSecondary italic bg-surfaceHover/30 p-4 rounded-2xl border border-border/60 shadow-sm leading-relaxed transition-all">
          &ldquo;{response.query}&rdquo;
        </p>
      </div>

      {/* Answer Output */}
      <div>
        <p className={`text-xs font-bold ${headerColor} uppercase tracking-wider mb-2 flex items-center gap-1.5 transition-all`}>
          <HeaderIcon className="w-4 h-4" /> {headerTitle}
        </p>
        <div className="text-lg text-textPrimary font-medium bg-bgDarker/40 p-5 rounded-2xl border border-border/80 leading-relaxed shadow-sm transition-all">
          {response.answer}
        </div>
      </div>
    </div>
  );
}

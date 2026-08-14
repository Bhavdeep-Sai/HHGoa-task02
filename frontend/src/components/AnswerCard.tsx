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
    headerColor = "text-red-400";
  } else if (isCasual) {
    headerTitle = "Off-Topic Query";
    HeaderIcon = HelpCircle;
    headerColor = "text-amber-400";
  } else if (!isGrounded) {
    headerTitle = "Insufficient Evidence";
    HeaderIcon = AlertTriangle;
    headerColor = "text-amber-400";
  }

  return (
    <div className="bg-surface border border-gray-800 rounded-2xl p-6 shadow-2xl space-y-4">
      {/* Top Bar: Language & Badges */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-gray-800/80 pb-4">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 bg-gray-900 border border-gray-700 px-3 py-1 rounded-lg text-xs font-medium text-gray-300">
            <Globe className="w-3.5 h-3.5 text-accent" />
            <span className="capitalize">{response.detected_language}</span>
            {response.is_code_mixed && (
              <span className="text-[10px] bg-accent/20 text-accent px-1.5 py-0.2 rounded">
                Code-Mixed
              </span>
            )}
          </div>

          {response.fast_path_used && (
            <div className="flex items-center gap-1 bg-emerald-950/60 border border-emerald-800/80 text-emerald-400 px-2.5 py-1 rounded-lg text-xs font-semibold">
              <Zap className="w-3.5 h-3.5" />
              <span>Fast Path (&lt;50ms)</span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Grounding / State Tag */}
          {isGrounded ? (
            <div className="flex items-center gap-1.5 bg-emerald-950/80 text-emerald-400 border border-emerald-800 px-3 py-1 rounded-lg text-xs font-medium">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Grounded ✓</span>
            </div>
          ) : isCasual ? (
            <div className="flex items-center gap-1.5 bg-amber-950/80 text-amber-400 border border-amber-800 px-3 py-1 rounded-lg text-xs font-medium">
              <HelpCircle className="w-4 h-4 text-amber-400" />
              <span>Off-Topic ⚠️</span>
            </div>
          ) : isSecurity ? (
            <div className="flex items-center gap-1.5 bg-red-950/80 text-red-400 border border-red-800 px-3 py-1 rounded-lg text-xs font-medium">
              <ShieldAlert className="w-4 h-4 text-red-400" />
              <span>Security Guardrail</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 bg-amber-950/80 text-amber-400 border border-amber-800 px-3 py-1 rounded-lg text-xs font-medium">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              <span>Insufficient Evidence</span>
            </div>
          )}

          {/* Confidence Badge */}
          <div className="bg-surfaceHover border border-gray-700 px-3 py-1 rounded-lg text-xs font-bold text-white">
            {confidencePct}% Confidence
          </div>
        </div>
      </div>

      {/* User Transcript */}
      <div>
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
          Transcribed User Question
        </p>
        <p className="text-base text-gray-300 italic bg-surfaceHover/50 p-3 rounded-xl border border-gray-800">
          &ldquo;{response.query}&rdquo;
        </p>
      </div>

      {/* Answer Output */}
      <div>
        <p className={`text-xs font-semibold ${headerColor} uppercase tracking-wider mb-1 flex items-center gap-1`}>
          <HeaderIcon className="w-4 h-4" /> {headerTitle}
        </p>
        <div className="text-lg text-white font-medium bg-gray-900/80 p-4 rounded-xl border border-gray-800 leading-relaxed shadow-inner">
          {response.answer}
        </div>
      </div>
    </div>
  );
}


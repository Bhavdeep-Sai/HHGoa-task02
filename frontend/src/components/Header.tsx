"use client";

import React from "react";
import { Mic, Terminal, ShieldCheck, Zap } from "lucide-react";

interface HeaderProps {
  debugMode: boolean;
  setDebugMode: (val: boolean) => void;
  selectedLanguage: string;
  setSelectedLanguage: (lang: string) => void;
}

export const SUPPORTED_LANGUAGES = [
  { code: "unknown", name: "Auto Detect (Indic/English)" },
  { code: "en-IN", name: "English (India)" },
  { code: "hi-IN", name: "Hindi (हिन्दी)" },
  { code: "te-IN", name: "Telugu (తెలుగు)" },
  { code: "ta-IN", name: "Tamil (தமிழ்)" },
  { code: "kn-IN", name: "Kannada (ಕನ್ನಡ)" },
  { code: "ml-IN", name: "Malayalam (മലയാളം)" },
  { code: "mr-IN", name: "Marathi (मराठी)" },
  { code: "bn-IN", name: "Bengali (বাংলা)" },
  { code: "gu-IN", name: "Gujarati (ગુજરાતી)" },
  { code: "pa-IN", name: "Punjabi (ਪੰਜਾਬੀ)" },
  { code: "od-IN", name: "Odia (ଓଡ଼ିଆ)" }
];

export function Header({
  debugMode,
  setDebugMode,
  selectedLanguage,
  setSelectedLanguage
}: HeaderProps) {
  return (
    <header className="border-b border-gray-800 bg-surface/80 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between flex-wrap gap-4">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent to-orange-600 flex items-center justify-center shadow-lg shadow-accentGlow">
          <Mic className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-white tracking-wide flex items-center gap-2">
            IndicVoiceRAG
            <span className="text-xs px-2 py-0.5 rounded-full bg-accent/20 text-accent font-medium border border-accent/30">
              Streaming STT + RAG
            </span>
          </h1>
          <p className="text-xs text-gray-400">
            Ask. Stream. Retrieve. In any Indian language.
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3 flex-wrap">
        {/* Language Selector Dropdown */}
        <select
          value={selectedLanguage}
          onChange={(e) => setSelectedLanguage(e.target.value)}
          className="bg-surfaceHover text-gray-200 border border-gray-700 rounded-lg px-3 py-1.5 text-xs font-medium focus:outline-none focus:border-accent cursor-pointer"
        >
          {SUPPORTED_LANGUAGES.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.name}
            </option>
          ))}
        </select>

        <div className="flex items-center gap-2 text-xs text-emerald-400 bg-emerald-950/50 px-3 py-1.5 rounded-lg border border-emerald-800/50">
          <Zap className="w-3.5 h-3.5" />
          <span>Target &lt;200ms RAG</span>
        </div>

        <button
          onClick={() => setDebugMode(!debugMode)}
          className={`flex items-center gap-2 text-xs font-medium px-3.5 py-2 rounded-lg border transition-all ${
            debugMode
              ? "bg-accent text-white border-accent shadow-md shadow-accentGlow"
              : "bg-surfaceHover text-gray-300 border-gray-700 hover:text-white"
          }`}
        >
          <Terminal className="w-4 h-4" />
          <span>Developer Mode</span>
        </button>
      </div>
    </header>
  );
}

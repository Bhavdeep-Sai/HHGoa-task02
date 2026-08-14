"use client";

import React from "react";
import { Mic, Terminal, ShieldCheck, Zap, Sun, Moon } from "lucide-react";

interface HeaderProps {
  debugMode: boolean;
  setDebugMode: (val: boolean) => void;
  selectedLanguage: string;
  setSelectedLanguage: (lang: string) => void;
  theme: "light" | "dark";
  toggleTheme: () => void;
}

export const SUPPORTED_LANGUAGES = [
  { code: "unknown", name: "Auto Detect (Indic/English)" },
  { code: "en-IN", name: "English (India)" },
  { code: "hi-IN", name: "Hindi (हिन्दी)" },
  { code: "te-IN", name: "Telugu (తెలుగు)" },
  { code: "ta-IN", name: "Tamil (தமிழ்)" },
  { code: "kn-IN", name: "Kannada (ಕನ್ನಡ)" },
  { code: "ml-IN", name: "Malayalam (മലയാളം)" },
  { code: "mr-IN", name: "Marathi (మরাठी)" },
  { code: "bn-IN", name: "Bengali (বাংলা)" },
  { code: "gu-IN", name: "Gujarati (ગુજરાતી)" },
  { code: "pa-IN", name: "Punjabi (ਪੰਜਾਬੀ)" },
  { code: "od-IN", name: "Odia (ଓଡ଼ିଆ)" }
];

export function Header({
  debugMode,
  setDebugMode,
  selectedLanguage,
  setSelectedLanguage,
  theme,
  toggleTheme
}: HeaderProps) {
  return (
    <header className="border-b border-border/40 bg-surface/75 backdrop-blur-md sticky top-0 z-50 px-8 py-4 flex items-center justify-between flex-wrap gap-4 shadow-sm shadow-slate-100/10 dark:shadow-black/5 transition-all duration-300">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent to-orange-600 flex items-center justify-center shadow-lg shadow-accentGlow">
          <Mic className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-textPrimary tracking-wide flex items-center gap-2">
            IndicVoiceRAG
            <span className="text-xs px-2 py-0.5 rounded-full bg-accent/20 text-accent font-medium border border-accent/30">
              Streaming STT + RAG
            </span>
          </h1>
          <p className="text-xs text-textTertiary">
            Ask. Stream. Retrieve. In any Indian language.
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3 flex-wrap">
        {/* Language Selector Dropdown */}
        <select
          value={selectedLanguage}
          onChange={(e) => setSelectedLanguage(e.target.value)}
          className="bg-surface border border-border/60 hover:border-border text-textSecondary hover:text-textPrimary rounded-full px-4 py-2 text-xs font-semibold focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/15 cursor-pointer shadow-sm transition-all duration-200"
        >
          {SUPPORTED_LANGUAGES.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.name}
            </option>
          ))}
        </select>

        <div className="flex items-center gap-2 text-xs font-semibold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 dark:bg-emerald-950/40 px-4 py-2 rounded-full border border-emerald-500/20 dark:border-emerald-800/30 shadow-sm transition-all duration-200">
          <Zap className="w-3.5 h-3.5" />
          <span>Target &lt;200ms RAG</span>
        </div>

        <button
          onClick={() => setDebugMode(!debugMode)}
          className={`flex items-center gap-2 text-xs font-bold px-4 py-2 rounded-full border shadow-sm transition-all duration-200 ${
            debugMode
              ? "bg-accent text-white border-accent hover:bg-accent/90"
              : "bg-surface border-border/60 text-textSecondary hover:text-textPrimary hover:border-border"
          }`}
        >
          <Terminal className="w-4 h-4" />
          <span>Developer Mode</span>
        </button>

        {/* Theme Switcher Toggle */}
        <button
          onClick={toggleTheme}
          className="flex items-center justify-center w-9 h-9 rounded-full border border-border/60 bg-surface text-textSecondary hover:text-textPrimary hover:border-border hover:bg-surfaceHover shadow-sm transition-all duration-200 cursor-pointer"
          title={theme === "light" ? "Switch to Dark Mode" : "Switch to Light Mode"}
        >
          {theme === "light" ? (
            <Moon className="w-4.5 h-4.5" />
          ) : (
            <Sun className="w-4.5 h-4.5 text-amber-400" />
          )}
        </button>
      </div>
    </header>
  );
}

export interface PipelineStageMetrics {
  stt_latency_ms: number;
  query_norm_latency_ms: number;
  preprocessing_ms?: number;
  embedding_ms?: number;
  qdrant_connect_ms?: number;
  dense_search_ms?: number;
  bm25_ms?: number;
  rrf_ms?: number;
  reranker_ms?: number;
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  grounding_ms?: number;
  rag_latency_ms?: number;
  voice_to_answer_latency_ms?: number;
  total_latency_ms: number;
}

export interface EvidenceItem {
  chunk_id: string;
  chunk_type: string;
  language: string;
  score: number;
  relevance_signals?: Record<string, any>;
  text: string;
}

export interface RAGPipelineResponse {
  request_id?: string;
  source?: string;
  query: string;
  detected_language: string;
  is_code_mixed: boolean;
  answer: string;
  confidence: number;
  grounded: boolean;
  intent?: string;
  refusal_reason?: string;
  fast_path_used: boolean;
  reranker_used: boolean;
  audio_duration_ms?: number;
  evidence: EvidenceItem[];
  citations: Array<{ chunk_id: string; reason: string }>;
  stage_latencies: PipelineStageMetrics;
  metrics: Record<string, any>;
}

const RAW_API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
const API_BASE = RAW_API_URL.endsWith("/api")
  ? RAW_API_URL
  : `${RAW_API_URL.replace(/\/+$/, "")}/api`;

export async function sendTextQuery(query: string): Promise<RAGPipelineResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, source: "text" })
  });
  if (!res.ok) {
    throw new Error(`API Error: ${res.statusText}`);
  }
  return res.json();
}

export async function sendVoiceQuery(audioBlob: Blob, language?: string): Promise<RAGPipelineResponse> {
  const formData = new FormData();
  formData.append("file", audioBlob, "voice_input.wav");
  if (language && language !== "auto") {
    formData.append("language", language);
  }

  const res = await fetch(`${API_BASE}/voice-query`, {
    method: "POST",
    body: formData
  });
  if (!res.ok) {
    throw new Error(`API Voice Query Error: ${res.statusText}`);
  }
  return res.json();
}

export async function transcribeAudio(audioBlob: Blob, language?: string): Promise<{ transcript: string; language: string; stt_latency_ms: number }> {
  const formData = new FormData();
  formData.append("file", audioBlob, "voice_input.wav");
  if (language && language !== "auto") {
    formData.append("language", language);
  }

  const res = await fetch(`${API_BASE}/transcribe`, {
    method: "POST",
    body: formData
  });
  if (!res.ok) {
    throw new Error("Speech recognition failed. Please try speaking again.");
  }
  return res.json();
}

export async function getSystemMetrics(): Promise<any> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) return null;
  return res.json();
}


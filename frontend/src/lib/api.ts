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

export interface BackendHealthResponse {
  ok: boolean;
  status?: string;
  service?: string;
  version?: string;
  retrieval_mode?: string;
  retrieval_ready?: boolean;
  latencyMs?: number;
  error?: string;
}

export async function checkBackendHealth(timeoutMs = 8000): Promise<BackendHealthResponse> {
  const startTime = Date.now();
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  // Derive root URL (without trailing /api)
  const rootUrl = API_BASE.replace(/\/api\/?$/, "");
  const healthEndpoints = [`${rootUrl}/health`, `${API_BASE}/health`];

  for (const url of healthEndpoints) {
    try {
      const res = await fetch(url, {
        method: "GET",
        signal: controller.signal,
        cache: "no-store"
      });
      clearTimeout(timeoutId);
      const latencyMs = Date.now() - startTime;
      if (res.ok) {
        const data = await res.json();
        return {
          ok: true,
          status: data.status || "ok",
          service: data.service,
          version: data.version,
          retrieval_mode: data.retrieval_mode,
          retrieval_ready: data.retrieval_ready ?? true,
          latencyMs
        };
      }
    } catch (e: any) {
      // Continue to next endpoint attempt or abort
      if (e.name === "AbortError") {
        return { ok: false, error: "Health check timed out (backend sleeping)" };
      }
    }
  }

  clearTimeout(timeoutId);
  return { ok: false, error: "Backend server is not reachable yet" };
}


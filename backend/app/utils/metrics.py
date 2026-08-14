import statistics
from typing import List, Dict, Any


class MetricsCollector:
    """Telemetry and metric aggregation for latency and request stats."""
    def __init__(self):
        self.total_queries: int = 0
        self.fast_path_count: int = 0
        self.grounded_count: int = 0
        self.latencies: List[float] = []
        self.retrieval_latencies: List[float] = []
        self.generation_latencies: List[float] = []
        self.language_counts: Dict[str, int] = {}

    def record(
        self,
        language: str,
        total_latency: float,
        retrieval_latency: float,
        generation_latency: float,
        confidence: float,
        fast_path: bool,
        grounded: bool
    ):
        self.total_queries += 1
        if fast_path:
            self.fast_path_count += 1
        if grounded:
            self.grounded_count += 1

        self.latencies.append(total_latency)
        self.retrieval_latencies.append(retrieval_latency)
        self.generation_latencies.append(generation_latency)
        self.language_counts[language] = self.language_counts.get(language, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        if not self.latencies:
            return {
                "total_queries": 0,
                "fast_path_ratio": 0.0,
                "grounded_ratio": 0.0,
                "p50_ms": 0.0,
                "p90_ms": 0.0,
                "p99_ms": 0.0
            }

        sorted_lat = sorted(self.latencies)
        n = len(sorted_lat)

        def pct(p: float) -> float:
            idx = int(round((p / 100.0) * (n - 1)))
            return round(sorted_lat[min(idx, n - 1)], 2)

        return {
            "total_queries": self.total_queries,
            "fast_path_count": self.fast_path_count,
            "fast_path_ratio": round(self.fast_path_count / max(1, self.total_queries), 2),
            "grounded_ratio": round(self.grounded_count / max(1, self.total_queries), 2),
            "mean_total_ms": round(statistics.mean(self.latencies), 2),
            "p50_ms": pct(50),
            "p70_ms": pct(70),
            "p90_ms": pct(90),
            "p95_ms": pct(95),
            "p99_ms": pct(99),
            "p100_ms": round(max(sorted_lat), 2),
            "languages": self.language_counts
        }

import os
import sys
import json
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.retrieval import HybridRetriever, BM25Retriever, DenseRetriever
from scripts.build_indexes import build_sample_indexes, SAMPLE_INDIC_DATA


def evaluate_retrieval():
    print("=" * 60)
    print("IndicVoiceRAG — Retrieval Quality & Multilingual Evaluation")
    print("=" * 60)

    build_sample_indexes()

    dense_retriever = DenseRetriever()
    hybrid_retriever = HybridRetriever()

    languages = ["hi", "te", "ta", "en"]
    results_by_lang = {}

    total_rec1, total_rec5, total_mrr = 0.0, 0.0, 0.0
    total_evals = 0

    for item in SAMPLE_INDIC_DATA:
        query_txt = item["query"]
        expected_query_id = item["query_id"]
        lang = item["language"]

        # Run Hybrid Retrieval
        candidates, reranker_used, top_conf, ret_ms = hybrid_retriever.search_sync(
            query=query_txt,
            language=lang,
            top_k=10
        )

        rank = None
        for r_idx, c in enumerate(candidates, start=1):
            q_id = c.get("payload", {}).get("query_id")
            if q_id == expected_query_id:
                rank = r_idx
                break

        rec1 = 1.0 if rank == 1 else 0.0
        rec5 = 1.0 if rank and rank <= 5 else 0.0
        mrr = (1.0 / rank) if rank else 0.0

        total_rec1 += rec1
        total_rec5 += rec5
        total_mrr += mrr
        total_evals += 1

        if lang not in results_by_lang:
            results_by_lang[lang] = {"evals": 0, "rec1": 0.0, "rec5": 0.0, "mrr": 0.0}
        
        results_by_lang[lang]["evals"] += 1
        results_by_lang[lang]["rec1"] += rec1
        results_by_lang[lang]["rec5"] += rec5
        results_by_lang[lang]["mrr"] += mrr

    summary_by_lang = {}
    for lang, metrics in results_by_lang.items():
        n = metrics["evals"]
        summary_by_lang[lang] = {
            "Recall@1": round(metrics["rec1"] / n, 4),
            "Recall@5": round(metrics["rec5"] / n, 4),
            "MRR": round(metrics["mrr"] / n, 4)
        }

    overall_metrics = {
        "overall_Recall@1": round(total_rec1 / max(1, total_evals), 4),
        "overall_Recall@5": round(total_rec5 / max(1, total_evals), 4),
        "overall_MRR": round(total_mrr / max(1, total_evals), 4),
        "by_language": summary_by_lang
    }

    print("\nRETRIEVAL EVALUATION REPORT:")
    print(json.dumps(overall_metrics, indent=2))

    out_dir = os.path.join("evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "retrieval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    print(f"\nSaved retrieval report to {report_path}")


if __name__ == "__main__":
    evaluate_retrieval()

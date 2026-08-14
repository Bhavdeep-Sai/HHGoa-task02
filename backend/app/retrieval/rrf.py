from typing import List, Dict, Any


def reciprocal_rank_fusion(
    results_dict: Dict[str, List[Dict[str, Any]]],
    weights: Dict[str, float] = None,
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combines ranked candidate lists from multiple branches (dense, bm25, qa) using RRF.
    """
    if weights is None:
        weights = {"dense": 0.5, "bm25": 0.3, "qa": 0.2}

    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for branch_name, item_list in results_dict.items():
        w = weights.get(branch_name, 0.3)
        for rank, item in enumerate(item_list, start=1):
            doc_id = item.get("id") or str(item.get("payload", {}).get("chunk_id"))
            if not doc_id:
                continue

            doc_map[doc_id] = item
            score_contrib = w / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score_contrib

    fused_results = []
    for doc_id, score in rrf_scores.items():
        item = doc_map[doc_id].copy()
        item["rrf_score"] = float(score)
        fused_results.append(item)

    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_results

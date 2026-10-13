"""Retrieval metrics measured against independently supplied relevant chunk IDs."""

import math


def rank_metrics(ranked_ids, relevant_ids, k=5):
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Each question must have at least one relevant chunk")
    seen, ranked = set(), []
    for item in ranked_ids:
        if item not in seen:
            ranked.append(item)
            seen.add(item)
        if len(ranked) == k:
            break
    matches = [int(item in relevant) for item in ranked]
    dcg = sum(value / math.log2(i + 2) for i, value in enumerate(matches))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {
        "hit": int(any(matches)),
        "recall": sum(matches) / len(relevant),
        "mrr": next((1 / (i + 1) for i, value in enumerate(matches) if value), 0),
        "ndcg": dcg / ideal,
    }

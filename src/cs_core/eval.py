"""Ranking-quality evaluation — the honest number cruncher.

LACE-style: NDCG@k, Recall@k on held-out interactions + category coverage / novelty
for a single user. Small, torch-free, numpy-only (sklearn for the heavy lifting of
ranking metrics isn't needed — pure formulas here).
"""

from __future__ import annotations

import math

import numpy as np


def _dcg(ranks: list[int]) -> float:
    return sum(1.0 / math.log2(i + 2) for i in ranks)


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """relevance: (N,) binary judgements aligned to the returned list."""
    top = relevance[:k]
    if top.sum() == 0:
        return 0.0
    dcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(top))
    ideal = sorted(top, reverse=True)
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(relevance: np.ndarray, k: int, total_rel: int | None = None) -> float:
    top = relevance[:k]
    total_rel = total_rel if total_rel is not None else int(relevance.sum())
    if total_rel == 0:
        return 0.0
    return int(top.sum()) / total_rel


def category_coverage(recs: list[str]) -> float:
    """Share of *distinct* categories present in the recommendation list."""
    return len(set(recs)) / len(recs) if recs else 0.0


def novelty(items: list[str], seen: set[str]) -> float:
    """Share of recommended items the user has never interacted with."""
    if not items:
        return 0.0
    return sum(1 for x in items if x not in seen) / len(items)


def evaluate_ranking(
    scores: np.ndarray,
    relevance: np.ndarray,
    rec_items: list[str],
    seen: set[str],
    k_list: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """One-stop evaluation: NDCG/Recall at each k + category coverage + novelty."""
    order = np.argsort(-scores)  # descending score
    rel = relevance[order]
    recs = [rec_items[i] for i in order]
    out = {}
    for k in k_list:
        out[f"ndcg@{k}"] = ndcg_at_k(rel, k)
        out[f"recall@{k}"] = recall_at_k(rel, k)
    out["category_coverage"] = category_coverage(recs)
    out["novelty"] = novelty(recs, seen)
    return out
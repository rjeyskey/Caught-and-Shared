"""Ranking-quality evaluation — the honest number cruncher.

LACE-style: NDCG@k, Recall@k on held-out interactions + category coverage / novelty
for a single user. Tuple-ordered: (item_ids ordered by score desc, relevance per item).

Category/novelty now take *item IDs* + an item→category map, so callers stop
pre-cooking strings; the evaluation owns the category logic (deepens the seam).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


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


def category_coverage(categories: Sequence[str]) -> float:
    """Share of *distinct* categories present in the recommendation list."""
    if not categories:
        return 0.0
    return len(set(categories)) / len(categories)


def novelty(items: Sequence, seen: set) -> float:
    """Share of recommended items the user has never interacted with."""
    if not items:
        return 0.0
    return sum(1 for x in items if x not in seen) / len(items)


def evaluate_ranking(
    scores: np.ndarray,
    relevance: np.ndarray,
    item_ids: Sequence,
    seen: set,
    category_of: dict | None = None,
    k_list: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """One-stop evaluation: NDCG/Recall at each k + category coverage + novelty.

    item_ids: ordered parallel to `scores`/`relevance`.
    seen: set of item ids the user has already interacted with (for novelty).
    category_of: optional item→category map. When provided, category coverage is
    computed from the top-k categories; else coverage is omitted.
    """
    order = np.argsort(-scores)  # descending score
    rel = relevance[order]
    items = [item_ids[i] for i in order]
    out = {}
    for k in k_list:
        out[f"ndcg@{k}"] = ndcg_at_k(rel, k)
        out[f"recall@{k}"] = recall_at_k(rel, k)
    if category_of is not None:
        cats = [category_of.get(x, "?") for x in items]
        out["category_coverage"] = category_coverage(cats)
    out["novelty"] = novelty(items, seen)
    return out
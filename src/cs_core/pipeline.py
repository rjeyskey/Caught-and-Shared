"""Shared experiment pipeline — the single seam scripts call.

Realises cards 0-5 of the 2026-10-07 architecture review:
- One `run_blend_experiment(...)` for both MovieLens and synthetic sets.
- Mentor selection, blending, evaluation all live behind it; scripts become thin adapters.
- `debug_users=N` replaces the old duplicated `01b_debug_single_user.py` (delete it).

Pipeline (Avito-mirroring shape):
  1. Build Item2Vec embeddings (gensim, no torch).
  2. User emb = mean of their liked-item embs (retrieval-boundary query).
  3. Mentor = closest warm user by genre-overlap (crude but honest 'taste I admire').
  4. Blend (1-α)·user + α·mentor via LensOperator; evaluate NDCG/R/coverage/novelty.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cs_core.blend import EmbeddingBlend  # noqa: E402
from cs_core.eval import evaluate_ranking  # noqa: E402


def default_embedding_factory(
    ratings_path: Path, dim: int = 128, epochs: int = 5, min_count: int = 2
):
    """Item2Vec (gensim SGNS) from a ratings csv: userId,movieId,rating,timestamp."""
    import pandas as pd
    from gensim.models import Word2Vec

    ratings = pd.read_csv(ratings_path)
    pos = ratings[ratings["rating"] >= 4.0]
    seqs = pos.groupby("userId")["movieId"].apply(list).tolist()
    seqs = [s for s in seqs if len(s) >= 3]
    return Word2Vec(seqs, vector_size=dim, window=5, min_count=min_count, sg=1, epochs=epochs, workers=1, seed=42)


@dataclass
class BlendResult:
    """One evaluated blend for one test user."""

    user: str
    alpha: float
    mentor: str
    mentor_overlap: int
    base: dict
    blended: dict
    held_items: set = field(default_factory=set)

    def deltas(self) -> dict:
        out = {}
        for k in ("ndcg@10", "recall@10", "category_coverage", "novelty"):
            out[f"delta_{k}"] = self.blended.get(k, 0.0) - self.base.get(k, 0.0)
        return out


def run_blend_experiment(
    data_dir: Path,
    alpha_values: list[float],
    topk: int = 20,
    n_test_users: int = 120,
    debug_users: int | None = None,
    embedding_factory: Callable = default_embedding_factory,
) -> list[BlendResult]:
    """Run mentor-blend evaluation over warm users.

    Returns per-user results; the caller (script) aggregates into JSON.
    """
    ratings = __import__("pandas").read_csv(data_dir / "ratings.csv")
    movies = __import__("pandas").read_csv(data_dir / "movies.csv")
    model = embedding_factory(data_dir / "ratings.csv")
    pos = ratings[ratings["rating"] >= 4.0]
    npos = pos.groupby("userId").size()
    warm = npos[npos >= 10].index.tolist()

    rng = np.random.default_rng(0)
    if debug_users is not None:
        test_users = warm[: debug_users]
    else:
        test_users = rng.choice(warm, size=min(n_test_users, len(warm)), replace=False)

    movie_genre = dict(zip(movies["movieId"], movies["genres"].fillna("")))
    user_items = pos.groupby("userId")["movieId"].apply(set).to_dict()

    def genre_of(ids):
        g = set()
        for m in ids:
            g.update(movie_genre.get(m, "").split("|"))
        return g

    def user_emb(ids) -> np.ndarray:
        vecs = [model.wv[m] for m in ids if m in model.wv]
        if not vecs:
            return np.zeros(model.vector_size)
        return np.mean(vecs, axis=0)

    blend_op = EmbeddingBlend()
    results: list[BlendResult] = []

    for uid in test_users:
        held = set(user_items[uid])
        if len(held) < 2:
            continue
        all_ids = sorted(held)
        n_held = max(1, int(0.2 * len(all_ids)))
        held_ids = set(all_ids[-n_held:])
        train_ids = [m for m in all_ids if m not in held_ids]

        u_emb = user_emb(train_ids)
        if not np.any(u_emb):
            continue
        best = max(
            (m for m in warm if m != uid),
            key=lambda m: len(genre_of(user_items[m]) & genre_of(train_ids)),
        )
        mentor_ids = [m for m in user_items[best] if m not in held]
        m_emb = user_emb(mentor_ids[:60])
        if not np.any(m_emb):
            continue

        cand_ids = [k for k in model.wv.key_to_index if isinstance(k, int)][:3000]
        cand_ids = [m for m in cand_ids if m not in train_ids]
        if not cand_ids:
            continue
        C = np.array([model.wv[m] for m in cand_ids])
        base_scores = C @ u_emb
        rel = np.array([1 if m in held_ids else 0 for m in cand_ids])
        seen = set(train_ids)
        category_of = {m: (movie_genre.get(m, "?") or "?").split("|")[0] for m in cand_ids}

        base = evaluate_ranking(base_scores, rel, cand_ids, seen, category_of=category_of, k_list=(5, 10, 20))
        for a in alpha_values:
            blended_vec = blend_op.blend(u_emb, [m_emb], a)
            b_scores = C @ blended_vec
            b_eval = evaluate_ranking(b_scores, rel, cand_ids, seen, category_of=category_of, k_list=(5, 10, 20))
            results.append(BlendResult(
                user=str(uid), alpha=a, mentor=str(best),
                mentor_overlap=len(genre_of(user_items[best]) & genre_of(train_ids)),
                base=base, blended=b_eval, held_items=held_ids,
            ))

    return results
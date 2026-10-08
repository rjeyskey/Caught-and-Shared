"""Script 01 — Embedding Blending + per-category α on MovieLens.

Honest numbers: does mentor-blending push NDCG/Recall up for warm users
vs the single-vector baseline, and what does it do to category coverage & novelty?

Pipeline (Avito-mirroring shape):
  1. Build Item2Vec embeddings from ml-latest-small ratings (gensim, no torch).
  2. For a test user: user emb = mean of their liked-item embs (retrieval-boundary query).
  3. Candidates = top-N by cosine from user emb (baseline) and from blended emb
     (1-α)·user + α·mentor, per user-choice of α.
  4. NDCG/R/coverage/novelty vs held-out.

Run:
  python scripts/01_embedding_blending_movielens.py --out data/test_results/blend_movielens.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cs_core.blend import EmbeddingBlend  # noqa: E402
from cs_core.eval import evaluate_ranking  # noqa: E402


def load_movielens(data_dir: Path):
    import pandas as pd

    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")
    return ratings, movies


def train_item2vec(ratings, dim: int = 128, epochs: int = 5):
    from gensim.models import Word2Vec

    pos = ratings[ratings["rating"] >= 4.0]
    seqs = pos.groupby("userId")["movieId"].apply(list).tolist()
    seqs = [s for s in seqs if len(s) >= 3]
    model = Word2Vec(seqs, vector_size=dim, window=5, min_count=2, sg=1, epochs=epochs, workers=1, seed=42)
    return model


def user_emb(model, movie_ids) -> np.ndarray:
    vecs = [model.wv[m] for m in movie_ids if m in model.wv]
    if not vecs:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO / "ml-latest-small")
    ap.add_argument("--out", type=Path, default=REPO / "data/test_results/blend_movielens.json")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--alpha", type=str, default="0.2,0.45,0.7")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    ratings, movies = load_movielens(args.data)
    print(f"[ml] ratings={len(ratings)} movies={len(movies)}")

    model = train_item2vec(ratings, dim=args.dim)
    print(f"[ml] item2vec vocab={len(model.wv)}")

    pos = ratings[ratings["rating"] >= 4.0]
    npos = pos.groupby("userId").size()
    warm = npos[npos >= 10].index.tolist()
    rng = np.random.default_rng(0)
    test_users = rng.choice(warm, size=min(120, len(warm)), replace=False)

    movie_genre = dict(zip(movies["movieId"], movies["genres"].fillna("")))
    user_items = pos.groupby("userId")["movieId"].apply(set).to_dict()

    def genre_of(ids):
        g = set()
        for m in ids:
            g.update(movie_genre.get(m, "").split("|"))
        return g

    blend = EmbeddingBlend()
    alphas = [float(x) for x in args.alpha.split(",")]

    results = defaultdict(list)
    for uid in test_users:
        held = set(user_items[uid])
        if len(held) < 2:
            continue
        all_ids = sorted(held)
        n_held = max(1, int(0.2 * len(all_ids)))
        held_ids = set(all_ids[-n_held:])
        train_ids = [m for m in all_ids if m not in held_ids]

        u_emb = user_emb(model, train_ids)
        if not np.any(u_emb):
            continue
        # mentor = closest by genre-overlap warm user (excluding self)
        best = max(
            (m for m in warm if m != uid),
            key=lambda m: len(genre_of(user_items[m]) & genre_of(train_ids)),
        )
        mentor_ids = [m for m in user_items[best] if m not in held]
        m_emb = user_emb(model, mentor_ids[:60])
        if not np.any(m_emb):
            continue

        # candidate pool: all items known to the model (int keys)
        cand_ids = [k for k in model.wv.key_to_index if isinstance(k, int)][:3000]
        cand_ids = [m for m in cand_ids if m not in train_ids]
        if not cand_ids:
            continue
        C = np.array([model.wv[m] for m in cand_ids])
        base_scores = C @ u_emb
        rel = np.array([1 if m in held_ids else 0 for m in cand_ids])
        seen = set(train_ids)
        movie_cat = {m: (movie_genre.get(m, "?") or "?").split("|")[0] for m in cand_ids}
        rec_items_cat = [movie_cat[m] for m in cand_ids]

        base = evaluate_ranking(base_scores, rel, rec_items_cat, seen, k_list=(5, 10, 20))
        for a in alphas:
            blended = blend.blend(u_emb, [m_emb], a)
            b_scores = C @ blended
            b_eval = evaluate_ranking(b_scores, rel, rec_items_cat, seen, k_list=(5, 10, 20))
            row = {"user": uid, "alpha": a, "mentor_overlap": len(genre_of(user_items[best]) & genre_of(train_ids))}
            for k in ("ndcg@10", "recall@10", "category_coverage", "novelty"):
                row[f"blend_{k}"] = b_eval[k]
                row[f"base_{k}"] = base[k]
                row[f"delta_{k}"] = b_eval[k] - base[k]
            results[a].append(row)

    summary = {}
    for a, rows in results.items():
        n = len(rows)
        if n == 0:
            summary[f"alpha_{a}"] = {"n_users": 0}
            continue
        def avg(key):
            return float(np.mean([r[key] for r in rows]))
        summary[f"alpha_{a}"] = {
            "n_users": n,
            "ndcg10_delta_mean": avg("delta_ndcg@10"),
            "recall10_delta_mean": avg("delta_recall@10"),
            "coverage_delta_mean": avg("delta_category_coverage"),
            "novelty_delta_mean": avg("delta_novelty"),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "alphas": alphas}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
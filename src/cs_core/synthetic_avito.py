"""Synthetic Avito-like marketplace — the category/mentor storytelling set.

Fake but structurally faithful:
- 12 categories (sneakers, home, tech, work, gifts, …)
- ~300 users x ~600 items; each user has per-category taste mixtures (mainstream vs niche
  per-user-per-category, echoing Analytis' per-domain strategy finding)
- 10 'mentor' profiles: curated (hand-set) per-category taste centers + interaction history
- interaction: users rate items via a softmax over cosine similarity to their (item, category) taste

The point is NOT realism — it's controllability: we control the ground truth (who likes what,
who is a good mentor for which category), so honest A/B of blend vs no-blend is possible
without confounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORIES = [
    "sneakers", "home", "tech", "work", "gifts", "auto",
    "children", "sport", "beauty", "furniture", "books", "fashion",
]


def _sim(items: np.ndarray, q: np.ndarray) -> float:
    qn = np.linalg.norm(q)
    inn = items @ q
    denom = np.linalg.norm(items, axis=1) * qn + 1e-9
    return float(np.max(inn / denom))


def generate(
    rng: np.random.Generator | None = None,
    n_users: int = 300,
    n_items: int = 600,
    n_cats: int = 12,
    n_mentors: int = 10,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(42)
    cats = CATEGORIES[:n_cats]
    D = 16  # small embedding dim, readable

    # item embeddings: gaussian per category + global drift → each item belongs mostly to one category
    item_ids = [f"i{i:04d}" for i in range(n_items)]
    item_cat = rng.integers(0, n_cats, size=n_items)  # hard category membership
    item_emb = rng.standard_normal((n_items, D))
    # push items toward their category center for clean per-category retrieval
    cat_center = rng.standard_normal((n_cats, D))
    for c in range(n_cats):
        mask = item_cat == c
        item_emb[mask] = item_emb[mask] + 2.0 * cat_center[c]
    item_emb = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-9)

    # users: per-category taste vector; taste skew (mainstream 0 ↔ niche 1) per user per category
    users = {}
    cat_map = {c: i for i, c in enumerate(cats)}
    for u in range(n_users):
        uid = f"u{u:04d}"
        # per-category taste centers (in category-direction) with user noise
        taste = rng.standard_normal((n_cats, D))
        taste = taste / (np.linalg.norm(taste, axis=1, keepdims=True) + 1e-9)
        users[uid] = {"id": uid, "taste": taste, "category": cats}

    # mentors: extreme per-category taste — a mentor is 'the sneakers guy' or 'the home guy'
    mentors = {}
    for m in range(n_mentors):
        mid = f"m{m:02d}"
        fav = m % n_cats  # primary category
        secondary = (m + 3) % n_cats
        taste = rng.standard_normal((n_cats, D))
        taste = taste / (np.linalg.norm(taste, axis=1, keepdims=True) + 1e-9)
        # amplify their signature categories
        for c in (fav, secondary):
            taste[c] = taste[c] + 3.0 * cat_center[c]
            taste[c] = taste[c] / (np.linalg.norm(taste[c]) + 1e-9)
        mentors[mid] = {"id": mid, "taste": taste, "fav_cat": cats[fav], "secondary": cats[secondary]}

    # build interactions: per user, sample items across categories (some users heavy on some cats)
    rows = []
    for u, info in users.items():
        n = rng.integers(30, 90)
        chosen = rng.choice(n_items, size=n, replace=False)
        for idx in chosen:
            c = int(item_cat[idx])
            sim = _sim(item_emb[[idx]], info["taste"][c])
            # rating ~ logistic of similarity + noise; cap reasonable
            p = 1.0 / (1.0 + np.exp(-4.0 * (sim - 0.2)))
            rating = int(round(5 * p))
            rating = max(1, min(5, rating))
            rows.append((u, item_ids[idx], cats[c], rating))
    interactions = pd.DataFrame(rows, columns=["user_id", "item_id", "category", "rating"])

    return {
        "rng": rng,
        "categories": cats,
        "items": pd.DataFrame({"item_id": item_ids, "category": [cats[c] for c in item_cat], "emb": list(item_emb)}),
        "users": users,
        "mentors": mentors,
        "emb": item_emb,
        "interactions": interactions,
        "cat_map": cat_map,
    }
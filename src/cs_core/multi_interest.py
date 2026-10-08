"""Multi-interest user representation, honest & light.

Realizes ComiRec's two extractors (dynamic-routing capsule; self-attentive) in a
vector-math form that works with Item2Vec/any pretrained embeddings — no torch needed.

MIND dynamic-count: K' = max(1, min(K, log2(|behaviors|)))
ComiRec-SA: V_u = softmax(W2.T · tanh(W1·H))^T · H  (K attention heads)
"""

from __future__ import annotations

import numpy as np


def dynamic_interest_count(n_behaviors: int, k_max: int = 8) -> int:
    """MIND's dynamic interest count formula (CIKM'19)."""
    from math import log2

    return max(1, min(k_max, int(log2(max(1, n_behaviors)))))


def self_attentive_interests(
    behavior_embs: np.ndarray, k: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """ComiRec-SA multi-interest extractor.

    behavior_embs: (T, D) sequence of item embeddings (user's history).
    Returns (K, D) interest vectors (one per attention head), K = dynamic count.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    T, D = behavior_embs.shape
    K = dynamic_interest_count(T)
    # W1: (D,H) projection; W2: (H,K) → K attention maps (fixed dim mismatch)
    H = 64
    W1 = rng.standard_normal((D, H)) / np.sqrt(D)
    W2 = rng.standard_normal((H, K)) / np.sqrt(H)
    M = np.tanh(behavior_embs @ W1)  # (T,H)
    A = np.tanh(M @ W2).T  # (K,T) attention logits per head
    A = np.exp(A - A.max(axis=1, keepdims=True))
    A = A / A.sum(axis=1, keepdims=True)
    interests = A @ behavior_embs  # (K,D)
    # L2-normalize so cosine serves as the similarity kernel
    interests = interests / (np.linalg.norm(interests, axis=1, keepdims=True) + 1e-9)
    return interests


def capsule_routing_interests(
    behavior_embs: np.ndarray,
    k: int,
    iters: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """ComiRec-DR style dynamic routing (capsules). Vector-math approximation.

    behavior_embs: (T, D). Returns (K, D) interest capsules, K = dynamic count.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    T, D = behavior_embs.shape
    K = dynamic_interest_count(T)
    S = rng.standard_normal((D, D)) / np.sqrt(D)  # shared transformation matrix
    primary = behavior_embs @ S  # (T, D) votes
    b = rng.standard_normal((T, K)) * 0.01  # routing logits
    for _ in range(iters):
        c = np.exp(b - b.max(axis=1, keepdims=True))
        c = c / c.sum(axis=1, keepdims=True)  # (T,K) coupling
        v = primary.T @ c  # (D,K) capsules
        norm = np.linalg.norm(v, axis=0, keepdims=True) + 1e-9
        v = (norm**2 / (1 + norm**2)) * (v / norm)  # squash
        b = b + (primary @ v)  # agreement update
    # last routing iteration's capsules out
    c = np.exp(b - b.max(axis=1, keepdims=True))
    c = c / c.sum(axis=1, keepdims=True)
    out = primary.T @ c
    out = out / (np.linalg.norm(out, axis=0, keepdims=True) + 1e-9)
    return out.T  # (K,D)
"""Caught & Shared core — pure vector-math blending at the retrieval boundary.

Design constraints (ADR-0001, ADR-0002, AGENTS.md principles):
- Blend is a *layer*, never a replacement: the underlying model and ranker are untouched.
- α=0 ⇒ bit-identical output (built-in rollback); α=1 ⇒ pure mentor takeover (explicit UI state).
- Per-category / per-interest alpha supported; multi-mentor supported.
- No I/O, no model training here — pure numpy math, no torch/catboost dependency.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class LensOperator(Protocol):
    """Single seam every influence technique implements (ADR-0002).

    EmbeddingBlend is the MVP implementation. Future techniques (ProxyBlend via
    Virtual Proxy Items, contrastive-trained blends, graph-leveraged vecs) implement
    the same Protocol and can be swapped in without touching ingest/rank/respond.
    """

    def blend(
        self,
        user_emb: np.ndarray,
        mentor_embs: Sequence[np.ndarray],
        alpha: float | Sequence[float],
        domain: str = "global",
    ) -> np.ndarray:
        """Return the blended user embedding.

        user_emb: (D,) single vector or (K,D) per-interest vectors.
        mentor_embs: one or more mentor embeddings, each (D,) or (K,D).
        alpha: scalar (all mentors share it) or sequence over mentors.
        domain: category key — reserved for capacity routing / explainability.
        """
        ...


class EmbeddingBlend:
    """Embedding Blending — the MVP operator (paper §3.3, ADR-0001).

    final = (1 - Σα) * user + Σ(α_i * mentor_i)

    The class is the ONLY public surface of this module. Callers construct
    `EmbeddingBlend()` (stateless but honest); there is deliberately no module-level
    convenience function, so the α-clamp logic cannot be bypassed (ADR-0002).
    """

    def blend(
        self,
        user_emb: np.ndarray,
        mentor_embs: Sequence[np.ndarray],
        alpha: float | Sequence[float],
        domain: str = "global",
    ) -> np.ndarray:
        user = np.asarray(user_emb, dtype=np.float64)
        mentors = [np.asarray(m, dtype=np.float64) for m in mentor_embs]

        if user.ndim == 2:
            # per-interest (K,D): scalar-per-mentor or sequence; clamp to convex hull.
            return self._blend_vectors(user, mentors, alpha)
        return self._blend_vectors(user, mentors, alpha)

    @staticmethod
    def _blend_vectors(
        user: np.ndarray,
        mentors: list[np.ndarray],
        alpha: float | Sequence[float],
    ) -> np.ndarray:
        if np.isscalar(alpha):
            alphas = np.full(len(mentors), float(alpha))
        else:
            alphas = np.asarray(alpha, dtype=np.float64).reshape(-1)
        total = alphas.sum()
        if total > 1.0:
            alphas = alphas / total  # stay inside the convex hull of tastes
        result = (1.0 - alphas.sum()) * user
        for a, m in zip(alphas, mentors):
            result = result + a * m
        return result


def rollout_checkpoint(lens_state: dict) -> dict:
    """Return the previous lens state (mentors, per-category α, timestamps).

    A 'checkpoint' = a snapshot the user can return to. The MVP blends away from
    the record and back by swapping `lens_state` for `previous`. The *record* itself
    is the rollback unit; this helper documents the contract the surface relies on.
    """
    return lens_state.get("previous", {"alpha": 0.0, "mentors": []})
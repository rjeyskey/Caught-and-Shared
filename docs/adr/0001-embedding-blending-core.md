# ADR-0001 — Emb layering as the product core (not graph surgery)

Status: Accepted (2026-10-07) · supersedes prior README taxonomy emphasis

## Context
The code-ground-truth must realize the science that survived 11-paper review: Embedding
Blending + controllable per-category α, multi-category mentors, honest evaluation.
Prior taxonomy listed 3 universal strategies (Graph-based, Contrastive Alignment,
Embedding Blending) as peers; literature synthesis (ComiRec, Harper, LACE, Brun,
GemiRec, DisMIR …) shows the industry-viable, user-controllable, do-no-harm path is a
thin inference-time layer, not graph surgery.

## Decision
- **Embedding Blending + per-category α** is the implemented core.
- **Structural graph approaches** (Heterogeneous/Dual Graph) move to a research appendix.
- **Contrastive Alignment** remains an auxiliary-training option, not a user-facing mechanism.
- **Virtual Proxy Items** stays alive as the one soft structural experiment if a graph angle is ever needed.

## Consequences
+ Presentable to Avito in one sentence: "we don't retrain your model; we blend embeddings at inference with user-controlled weights."
+ Rollback by construction (α=0 → identical output).
+ Multi-category mentors remain the real differentiator vs all papers read.
− The graph/contrastive taxonomy notebooks stay as references, not shipped code.
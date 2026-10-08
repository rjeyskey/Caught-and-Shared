# ADR-0002 — LensOperator as the single blend seam

Status: Accepted (2026-10-07)

## Context
Six original techniques must survive the MVP without shipping all six. Future
techniques (contrastive, proxy, graph) should attach without touching ingest/rank/respond.

## Decision
Define one interface that ALL current and future influence types implement:

```python
class LensOperator(Protocol):
    def blend(
        self,
        user_emb: np.ndarray,      # (D,) or (K,D) per-interest
        mentor_embs: list[np.ndarray],
        alpha: float | list[float],  # scalar or per-interest
        domain: str = "global",      # category key
    ) -> np.ndarray: ...
```

MVP ships `EmbeddingBlend` (α-blend math). `ProxyBlend` (Virtual Proxy Items) can be added
later behind the same seam without touching the serving path.

## Consequences
+ The product cycle `set mentor → influence updates → α levels → multi-mentor result`
  maps 1:1 to calling `blend()` with a new (mentor, α, category) tuple.
+ The demo surface (Figma/Paper → HTML/JSON) drives the same calls — one code path.
− Abstraction must not leak: `blend()` stays pure vector math (no I/O, no model training).
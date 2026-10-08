<p align="center">
<img src="assets/trust-driven-recs.png">
</p>

# Caught & Shared

A research project exploring conscious control in recommender systems.

**Website**: https://rjeyskey.github.io/Caught-and-Shared/

**Paper**: [caught-and-shared-conscious-control-in-recommender-systems.pdf](caught-and-shared-conscious-control-in-recommender-systems.pdf)

**Pitch**:
eng: https://www.youtube.com/watch?v=QtJTm5fp5rY
rus: https://www.youtube.com/watch?v=O5XaG_wbHDI

---

## The Idea

Modern recommender systems learn from what users do.

Caught & Shared explores a complementary idea:

**What if users could explicitly choose whose taste influences their recommendations?**

Instead of relying solely on implicit signals, users may consciously select mentors — friends, creators, or people they admire — and borrow their perspective.

Choose who influences your recommendations.
Share taste.
Build recommendation systems together.

---

## Implementation (the short version)

The experimental seed that proves the mechanism lives in **`code-ground-truth/`** (this repository = research + pitch artifacts).

- **Core (importable, tested):** `src/cs_core/`
  - `blend.py` — `LensOperator` seam + `EmbeddingBlend`: `final = (1-Σα)·user + Σ(α_i·mentor_i)` at the retrieval boundary. α=0 ⇒ identical output (rollback by construction); α=1 ⇒ takeover (explicit UI state).
  - `multi_interest.py` — ComiRec-SA self-attention + ComiRec-DR capsule routing + MIND dynamic interest count (`K' = max(1,min(K,log₂|behaviors|))`).
  - `eval.py` — LACE-style NDCG@k / Recall@k + category coverage + novelty.
  - `synthetic_avito.py` — a small fake marketplace (12 life categories, 10 mentors with per-category taste) for storytelling experiments.
  - `pipeline.py` — one shared experiment path (data → item2vec → mentor select → blend → evaluate).
- **Experiments (thin CLI adapters):** `scripts/`
  - `01_embedding_blending_movielens.py` — honest numbers on MovieLens ml-latest-small (real ratings). Run `--debug-users N` to reproduce a single-user debug (replaces the old `01b_*`).
  - `02_multicategory_mentors_synthetic.py` — multi-category mentors on the synthetic set: coverage ↑ while relevance stays flat (do-no-harm + diversity proof).
- **Numbers (on disk):** `data/test_results/*.json`
  - MovieLens (120 warm users, mentor = closest genre-overlap user): NDCG@10 Δ **+0.019** (α=0.45) and **+0.025** (α=0.7); Recall@10 Δ **+0.007 / +0.016**. Negative deltas: none.
  - Synthetic (60 users): category coverage Δ **+0.083** with relevance flat (0.000) — blending adds categories without harming relevance.

### Repo layout (navigate)

```
code-ground-truth/
├── README.md                  ← you are here; the map
├── GLOSSARY.md                ← domain vocabulary (Mentor, Lens, α, Category, Blend, …)
├── AGENTS.md                  ← agent/engineering config (issue tracker, triage labels, domain docs)
├── docs/
│   ├── adr/                   ← architecture decisions (0001 blending-core, 0002 LensOperator)
│   └── agents/                ← github-auth notes, eval plan, issue-tracker, domain, triage
├── src/cs_core/               ← importable product core (torch-free, numpy only)
├── scripts/                   ← thin runnable experiment CLIs
├── applications/              ← research/taxonomy notebooks (torch/pyg) — the 6-technique story for the pitch
├── data/test_results/         ← honest JSON numbers
├── assets/                    ← figures for the site/slides
├── index.html                 ← public landing (GitHub Pages)
└── caught-and-shared-conscious-control-in-recommender-systems.pdf   ← the paper
```

**Why two code styles?** `applications/` are the research notebooks (torch + torch-geometric) that tell the *taxonomy* story (Graph-based, Contrastive Alignment, Embedding Blending, Multi-Category Mentors). The shipped product core is `src/cs_core/` — deliberately torch-free so the blending math runs anywhere and can be unit-tested without a GPU stack. Techniques that graduate get ported into `src/cs_core/` behind the `LensOperator` seam (see ADR-0002).

---

## Core Implementation Taxonomy

Regardless of the underlying architecture, the Caught & Shared concept can be realized through three universal strategies for injecting mentor influence:

## 1. Graph-based (Structural) Approaches

We explicitly add mentor relationships into the graph. This can be done by transforming the original user-item bipartite graph into a heterogeneous multi-relational graph or by using alternative techniques that preserve bipartiteness.

**Main variants**
- **Heterogeneous Graph** — Direct user → mentor edges (most powerful signal)
- **Virtual Proxy Items** — Represent each mentor as a special virtual item
- **Dual Graph** — Separate bipartite and social graphs with later fusion
- **MetaPath-based** — Information flows through predefined paths (e.g., `User → Mentor → Item`)

**Strengths:** Strong, direct influence and natural propagation.
**Weaknesses:** Varies by variant — from breaking bipartiteness to increased complexity.

> **Status (2026-10):** kept as research appendix & use-case for the taxonomy. Not the product core (ADR-0001).

## 2. Contrastive Alignment

Instead of changing the graph, we add a contrastive objective during training that pulls the user’s embedding closer to their chosen mentor’s embedding in latent space, while pushing it away from random users.

**Key idea**
- Positive pair: *(user, mentor)*
- Negative pairs: *(user, random users)*

**Strengths**
- Preserves original graph structure
- Works with any neural model (Two‑Tower, GNNs, Transformers)
- Creates meaningful semantic closeness between user and mentor

**Weaknesses:** Influence is softer and requires mentor choice data during training.

> **Status (2026-10):** useful as an *auxiliary training loss*, not a user-facing mechanism (ADR-0001).

## 3. Embedding Blending (Composite) — ✅ the product core

The simplest and most controllable method — directly blend user and mentor embeddings at inference time:

```
User_final = (1 - α) * User_emb + α * Mentor_emb
```

**Strengths**
- Maximum user control and transparency
- Extremely easy to implement and explain
- Works on top of any existing model
- Perfect for MVP and production

**Weaknesses:** Post-processing approach (model doesn’t learn mentor relationships during training).

> **Status (2026-10):** implemented in `src/cs_core/blend.py`, per-category α + multi-mentor + rollback-by-construction. This is THE thing we pitch: "we don't retrain your model; we blend embeddings at inference with user-controlled weights."

---

## Influence Control (Product Feature)

When choosing a mentor, users can select the depth of influence:

- **Light (20%)** — Gentle nudge toward the mentor’s taste
- **Balanced (45%)** — Nearly equal mix of both tastes
- **Deep (70%)** — Strong immersion into the mentor’s style

This gives users a clear sense of agency and reduces the fear of completely losing their own identity in recommendations.

Additionally, users can:
- Temporarily try different influence levels
- Revert to previous recommendation state ("Reset to my original taste")
- Fine‑tune alpha manually if desired

---

## Future Growth: Conscious Recommendations & Multi‑Category Mentors

**Caught & Shared** is the first building block of a broader vision — **Conscious Recommendations**.

In the future, users will be able to assign different mentors to different life categories (sneakers, outerwear, tech, home, workwear, etc.). This creates a personal **"Mentor Team"** where each sphere of life is guided by someone whose taste the user genuinely admires.

This approach transforms recommendations from passive algorithmic output into a deeply personal, human‑curated experience.
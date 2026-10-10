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

## The Idea

Modern recommender systems learn from what users do.

Caught & Shared explores a complementary idea:

**What if users could explicitly choose whose taste influences their recommendations?**

Instead of relying solely on implicit signals, users may consciously select mentors — friends, creators, or people they admire — and borrow their perspective.

Choose who influences your recommendations.
Share taste.
Build recommendation systems together.

## How it's implemented

The core of the project is in `code-ground-truth/` — small, honest experiments that realize the science into code.

```
code-ground-truth/
├── src/cs_core/        # the engine: blending, multi-interest, evaluation (pure Python)
├── scripts/            # runnable experiments
├── applications/       # research notebooks — the 6 technical directions we explored
├── data/test_results/  # honest numbers from the experiments
└── docs/               # glossary + architecture decisions (why we chose what we chose)
```

**The idea in one line:** blend the user's taste vector with a mentor's taste vector at recommendation time — `final = (1 - α) * user + α * mentor`. α is the dial the user controls (20 / 45 / 70%), and α = 0 means your feed is exactly as it was (rollback by construction).

**Why embedding blending (and not the other approaches)?** It's the only one that:
- sits *on top* of an existing recommender without changing it (do no harm),
- gives the user a real dial (control),
- is explainable ("40% of your feed is your friend's taste"),
- and doesn't need retraining.

Graph-based and contrastive approaches stay in `applications/` as the research story — they're stronger signals in theory, but they touch the core model, which we deliberately avoid.

## What we've measured so far

| Experiment | What it shows |
|---|---|
| MovieLens (100k real ratings) | Blending a mentor's taste in **improves recommendation quality** (NDCG@10 up +2.5% at α=0.7, +1.9% at α=0.45), no degradation at any α |
| Synthetic marketplace (12 categories) | Adding a mentor for a "cold" category **grows feed coverage (+8% categories) without hurting relevance** — new worlds to explore, nothing broken |

Both results are on disk in `data/test_results/` and reproducible with two commands:

```bash
python scripts/01_embedding_blending_movielens.py
python scripts/02_multicategory_mentors_synthetic.py
```

Honest caveat: these are offline experiments. The "users feel more in control" claim is the next step — a small user study (the protocol is ready in `docs/`).

## Core Implementation Taxonomy

Regardless of the underlying architecture, the Caught & Shared concept can be realized through three universal strategies for injecting mentor influence:

## 1. Graph-based (Structural) Approaches

**Main variants**
- **Heterogeneous Graph** — Direct user → mentor edges (most powerful signal)
- **Virtual Proxy Items** — Represent each mentor as a special virtual item
- **Dual Graph** — Separate bipartite and social graphs with later fusion
- **MetaPath-based** — Information flows through predefined paths (e.g., `User → Mentor → Item`)

**Strengths:** Strong, direct influence and natural propagation.
**Weaknesses:** Varies by variant — from breaking bipartiteness to increased complexity.

## 2. Contrastive Alignment

Instead of changing the graph, we add a contrastive objective during training that pulls the user's embedding closer to their chosen mentor's embedding in latent space, while pushing it away from random users.

**Key idea**
- Positive pair: *(user, mentor)*
- Negative pairs: *(user, random users)*

**Strengths**
- Preserves original graph structure
- Works with any neural model (Two‑Tower, GNNs, Transformers)
- Creates meaningful semantic closeness between user and mentor

**Weaknesses:** Influence is softer and requires mentor choice data during training.

## 3. Embedding Blending (Composite)

The simplest and most controllable method — directly blend user and mentor embeddings at inference time:

User_final = (1 - α) * User_emb + α * Mentor_emb

**Strengths**
- Maximum user control and transparency
- Extremely easy to implement and explain
- Works on top of any existing model
- Perfect for MVP and production

**Weaknesses:** Post-processing approach (model doesn't learn mentor relationships during training).

---

## Influence Control (Product Feature)

When choosing a mentor, users can select the depth of influence:

- **Light (20%)** — Gentle nudge toward the mentor's taste
- **Balanced (45%)** — Nearly equal mix of both tastes
- **Deep (70%)** — Strong immersion into the mentor's style

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
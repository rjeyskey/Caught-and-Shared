# Caught & Shared — Code Ground-Truth (Week 2)

Scientific spike + Avito-shaped scaffold for the mentor-blending core.

## What this repo holds

- **src/cs_core/** — the reusable core (LensOperator interface, blend math, multi-interest, evaluation).
- **scripts/** — runnable experiments (MovieLens + synthetic Avito-like marketplace).
- **data/test_results/** — JSON tables (honest numbers for the pitch).
- **docs/adr/** — architecture decisions (started by grill-with-docs 2026-10-07).

## Two datasets, two jobs

| Dataset | Job |
|---|---|
| MovieLens ml-latest-small | Quantitative honesty — real ratings, hard NDCG/Recall numbers |
| Synthetic Avito-like marketplace | Category storytelling — sneakers/home/tech mentors, per-category α |

## Principles (from AGENTS.md)

1. User agency first — influence visible/adjustable/reversible.
2. Do not harm the existing RecSys — C&S is a layer, not a replacement.
3. Comprehensibility — user understands mentor + α + rollback.
4. Rollback mandatory (checkpoints, previous α, "pure me").
5. Multi-category is the real product.
6. Code ⇄ Design loop stays simple and reusable.

## Ground truth posture (after 11 papers)

- **Embedding Blending + per-category α** = product core and Avito story.
- **Multi-category mentors** = differentiator vs all literature.
- Structural graph (hetero/dual) = research appendix; Contrastive = auxiliary loss.
- Virtual Proxy Items = the one soft structural live option.

## Non-goals (this week)
- No real two-tower training, no CatBoost, no Triton — that's Avito's stack, not our science.
- No UI/extension yet — design-facing hooks only.

## Run
```bash
# from repo root; python 3.11 (Hermes venv has pandas 3.0.6, numpy 2.4.3, gensim 4.4.0,
# sklearn 1.9.0, scipy, fastapi; NO torch/catboost — code must not require them)
python scripts/01_embedding_blending_movielens.py --help
```
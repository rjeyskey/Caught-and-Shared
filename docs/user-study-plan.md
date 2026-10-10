# User Study Plan — real mentor choice, not simulated similarity

Status: **DRAFT — the real evidence path for C&S.** Replaces the "similar-user proxy"
offline experiments as the source of the product claim. Lightweight, honest, runnable
in ~1–2 weeks with 15–30 people (LACE-style).

## Why (the hole the offline tests leave)

The product claim is: *a user who explicitly chooses a person they trust gets a better
or more satisfying feed*. The offline experiments matched *similar* users as mentors —
which is exactly what ordinary recommenders already do, so they cannot evidence the
product. We need human context: a real person + a real choice + a real sense of control.

## Core idea

Give a participant a **mentor pool of real people they actually know** (friends),
let them pick, let them tune α, and measure both objective ranking and subjective
control (LACE used the exact same shape — see ground-truth skill).

## Protocol (within-subject, 15–30 participants)

1. **Setup (10 min).** Each participant invites 1–3 friends they trust for taste in a
   domain (marketplace items, music, movies…) via a shareable link. Friends consent
   and their taste profile is built from their existing interactions (opt-in read).
2. **Baseline (5 min).** Participant browses their normal personalized feed; we log which
   items they save/like/open.
3. **Assignment (5 min).** From the pool of their friends (only people they know), they
   pick 1–2 mentors per category and set α (Light / Balanced / Deep).
4. **Treated (5–10 min).** They browse the blended feed; log saves/likes/opens again.
5. **Rollback check.** They can flip to "pure me" any time; we log every rollback.
6. **Survey (5 min).** The 10-item Likert (sense of control, clarity, interest,
   discovery, "more me", mentor-utility, α-utility, would-use, emotional, open).

## Metrics

- **Objective (from logged behavior):** NDCG@10/20 + Recall@20 on items saved *after*
  assignment vs the baseline feed's saved items; category coverage of saved items;
  novelty of saved items; rollback rate; CTR/engagement.
- **Subjective (survey):** 10 Likert items; success = ≥70% score control ≥4, average
  "more interesting" ≥3.8, majority would-use.

## Implementation (in this repo, minimal)

- `scripts/03_user_study_prepare.py` — generate the invite/link, collect the mentor
  profile consents, build per-category mentor taste vectors (same Item2Vec or
  synthetic embedding path as `src/cs_core/`).
- `src/cs_core/study/` — a tiny FastAPI surface the extension/UI can hit to
  (a) register a mentor assignment, (b) fetch a blended feed for (user, mentor, α),
  (c) log saves and rollbacks.
- `scripts/04_user_study_report.py` — aggregate logged events → the numbers table.

## What we need from real people (preliminaries)

1. A small pool of participants who actually use the same platform/domain (Avito,
   or our demo domain) and have real friends who also use it.
2. Friends consent to their taste being shared as a "lens" (the reciprocity model —
   open question in the skill).
3. The domain's existing feed (Avito API, or our MovieLens demo as a stand-in) to
   produce baseline + treated lists.

## Acceptance (what "it works" means)

- Users can find their real friends, assign them, and report that the feed changed in
  a direction they can articulate (survey Q2).
- Objective saved-items NDCG does not drop below baseline (do-no-harm) — ideally
  coverage and novelty rise.
- Rollback rate is low *except* when users experiment (exploratory rollback is fine);
  users can always return to baseline without loss.
- Qualitative gold: "this feels like my friend picked it" comments we can quote.

## Deferred

- Recruitment logistics (Avito internal pilot vs student group) — decide with Alexey.
- Privacy/consent wording for the invite (RU + EN).
- Whether the study runs on Avito real data or the MovieLens demo (both machine paths
  are the same once the lens is a real person's profile).
# User Study Plan

The real product claim can't be proven by matching similar users offline — that's what
ordinary recommenders already do. We need real people choosing real friends and telling
us how it feels. This is that test. Lightweight, LACE-style, ~15–30 people, ~1–2 weeks.

## The core move

A participant gets a pool of their **actual friends** (people they genuinely trust for
taste in a domain), picks 1–2 per category, sets the α, and we compare their feed before
and after — objectively and subjectively.

## Protocol

1. **Invite (10 min).** Participant invites 1–3 friends via a link. Friends opt in; their
   taste profile is built from their existing interactions.
2. **Baseline (5 min).** Participant browses their normal feed; we log saves/likes/opens.
3. **Assign (5 min).** They pick mentors from their friend pool, per category, and set
   α (Light / Balanced / Deep).
4. **Treated (5–10 min).** They browse the blended feed; we log the same events.
5. **Rollback.** They can flip to "pure me" anytime; we count rollbacks.
6. **Survey (5 min).** 10 Likert items: control, clarity, interest, discovery, "more me",
   mentor usefulness, α usefulness, would-use, emotional, open question.

## What we measure

- **Objective:** NDCG@10/20 + Recall@20 on items saved *after* assignment vs baseline's
  saved items; category coverage and novelty of saved items; rollback rate; engagement.
- **Subjective:** the 10 Likert items. Success = ≥70% score control ≥4, average
  "more interesting" ≥3.8, majority would use it.

## What "it works" means

- Users find their real friends, assign them, and can say how the feed changed.
- Saved-item NDCG doesn't drop below baseline (do no harm) — ideally coverage and
  novelty rise.
- Rollback is rare unless experimenting; returning to baseline is always possible.
- We get quotable moments like "this feels like my friend picked it."

## What we implement (minimal)

- `scripts/03_user_study_prepare.py` — invite link, consent, build mentor taste vectors
  (same embedding path as `src/cs_core/`).
- `src/cs_core/study/` — tiny FastAPI surface: register assignment, fetch blended feed,
  log saves and rollbacks.
- `scripts/04_user_study_report.py` — aggregate logged events into the numbers table.

## Open questions (decide before running)

1. Where do we run it — Avito real data (internal pilot) or the MovieLens demo? Both use
   the same machine path once the lens is a real person's profile.
2. Who are the participants — student group or friendly Avito users?
3. Privacy/consent wording for the invite (RU + EN) — needs real legal eyes, not ours.
4. Does a friend's profile need active consent each time, or once? (reciprocity model)
# GLOSSARY

Caught & Shared domain vocabulary, single-context. Terms defined as the design resolves (grill-with-docs).

## Mentor
A person whose taste the user explicitly chooses to influence their recommendations in a given category (or globally). Source of a *mentor profile*: a shareable, trained taste representation that pre-exists as a readable artifact (public demo profile or an explicit interaction history).

## Mentor profile
The artifact that represents a mentor's taste to the system: trained embedding(s) + the interaction history they derive from. C&S does not create this out of nothing — it consumes it (source-profile dependency).

## Lens
The combined influence configuration for a user: which mentors, for which categories, at which α (strength). "Borrow a lens" = apply someone's taste to your own feed.

## α (alpha) / influence level
Per-category strength of mentor influence on the user's blending vector. α=0 ⇒ pure own taste (bit-identical to the unblended feed); α=1 ⇒ pure mentor taste (takeover). Product presets: Light (~20%), Friendly-Half/Balanced (~45%), Big Piece/Deep (~70%).

## Category (life category)
A sphere of life the user distinguishes (sneakers, home, tech, work…). The unit at which mentor assignment, α, and rollback operate. Multi-category mentors = the product's core differentiator.

## Blend
The inference-time operation `final = (1-Σα)·own + Σ(α·mentor)` at the retrieval-boundary embedding. Pure post-processing: the underlying model and ranker are untouched.

## Cold start
The user state with little/no history, where own-taste is empty and the blend degenerates to mentor takeover. C&S surface must make this state explicit ("you're seeing your master's lens"), never silent.

## Rollback / checkpoint
Recoverable previous state of (mentors, per-category α, timestamps). Rollback returns the feed to a prior lens state; "pure me" resets all α to 0.

## Conscious recommendation
The declared product family: user-driven, inspectable, reversible control over recommendation influence — of which C&S (mentor + α layer) is the first building block.

## Retrieval boundary
The point in a production pipeline (two-tower / candidate-generation) where the user's embedding is computed and used to match candidates. C&S blends at this boundary (or re-ranks its output); all other stages stay untouched.
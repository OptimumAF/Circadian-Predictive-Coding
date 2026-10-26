# ADR-0525 — Compose low-rank updates through a small core

## Context

R5.4 native adapter consolidation remains missing. Matched native A/B controls
show acquisition/retention failures; cold C was already correct and does not
establish new acquisition. None of these results diagnoses rank deficiency or
proves compression useful. Cached native PEFT adapters use scaled B@A products
(with explicit fan-in-fan-out transpose semantics), not independent factor sums.

## Decision

Add pure NumPy exact product composition and optional thin QR/SVD projection.
Accept explicit A[r,in]/B[out,r], signed mixture coefficient and scaling; return
owned float64 factors with scaling1, spectral tail and bounded explicit work.
No dense original operator/base construction, model calls, native defaults or
head/optimizer changes. Caller establishes shared base and orientation; future
native integration must validate these and its float32/cost/reset boundaries.

Why this: product concatenation is exact without the cross terms of factorwise
averaging. Reduced QR preserves nonzero singular values in a small bounded core,
allowing an optimal-rank reference without materializing a large backbone matrix.
NumPy already exists; float64 gives an explicit algebra/reference boundary.

## Alternatives

- Averaging A/B separately changes the intended weighted sum with cross terms.
- Dense full updates need unnecessary out×in storage.
- Automatic rank/weight selection would introduce an unfrozen scientific choice.
- Installing factors in the learner now would mix algebra validation with native
  scaling/casting/state/policy changes before the math is qualified.

## Consequences

The core has no evidence/data/learner authority. Bounds precede scans and explicit
allocation; internal LAPACK workspace and FLOPs stay unknown. Tail residual is
spectral/truncation error, not rounded native-output error. Numerical rank uses
a stated relative threshold; cancellation may leave tiny numerical residuals.
Owner integration must permanently charge attempts and explicitly handle stale
AdamW moments after a coordinate change. No training-budget renewal or final/
probe-based choice is allowed. Full R5.4 retention/rank/calibration and repository
release remain open until their own evidence exists.

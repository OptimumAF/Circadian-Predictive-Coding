# ADR-0616 — Compose the existing affine TRAIN control by public cue

## Context

The original contextual game control uses two affine heads routed by an initial
visible cue. Its partial TRAIN fit and incomplete games do not establish useful
learning. The later readonly diagnosis found correctly aligned, conflict-free
inputs; a lookup ceiling does not establish affine attainability. The existing
pooled one-hot least-squares control already provides a fixed constructive fit.

## Decision

Add an explicit two-cue mode to that fitter and its existing CLI. Admit all
causal TRAIN sources first, route by the initial public image and fit once per
head using the original solver, intercept, rank cutoff and objective. Preserve
separate models, work and failures. Validate paid previous results before further
work and aggregate only after both complete, in original source order.

Why this: it directly tests the existing two-head affine hypothesis with an
existing solver. It adds no optimizer, orchestration framework or game variant.

## Alternatives

Increasing perceptron passes would introduce another dose question. Treating
conflict-free inputs as a fit proof would conflate lookup and affine capacity.
Injecting solver coefficients into perceptron snapshots would misstate their
learning history and objective.

## Consequences

This is an offline conventional TRAIN control. It has the contextual control's
numeric head capacity and twice the pooled control's capacity, with a different
learning objective and work. It transfers no optimizer, clock, budget or native
restoration authority. Full TRAIN fit is required before considering new games;
it does not prove live task success, chronological adaptation or generalization.
All existing controls, failures and scientific limits remain intact.

# ADR-0468: Use declared RGB range as an explicit transform

## Context

The default visual learner fails both native contexts, including after eight
corrective updates. Stored input diagnostics identify large empirical RGB
normalization excursions in B; this is not a proven cause. The existing frozen
public recipe already declares feature order and RGB range, while empirical
vector statistics and all accepted default behavior must remain reproducible.

## Decision

Add a pure exact-recipe option: fit the21 public vector coordinates using TRAIN-A
only, then use RGB mean.5/scale.5 from the declared[0,1] sensor range. Both complete
TRAIN episodes validate before fitting/model creation; terminal observations
validate but are not fitted. Expose the option through existing fit/result/JSON/
CLI boundaries. Keep default empirical calls/reports and policy schema unchanged;
report the effective modality provenance and validation/fitting costs explicitly.

## Alternatives

Tuning a variance floor or clipping would add outcome-sensitive parameters.
Fitting B, probes or final data would violate isolation. Changing the default
would alter accepted baselines. A new runner, learned representation, snapshot
format or optimizer is unnecessary for this controlled input-transform test.

## Consequences

RGB contribution is bounded for admitted public inputs; vector inputs remain
unbounded and shifted. The policy retains the effective flat transform, while
the explicit report carries fitting provenance. One small offline matched
control can test this hypothesis without native retries or a CPC benefit claim.
All previous negative results and unfinished parent requirements remain.

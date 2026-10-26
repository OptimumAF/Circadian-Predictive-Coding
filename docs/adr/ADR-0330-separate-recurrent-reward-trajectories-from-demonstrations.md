# ADR-0330: Separate recurrent reward trajectories from demonstrations

## Context

R4.2 requires a separately evaluated recurrent RL policy. Existing recurrent BC
and isolated fresh-task fits do not implement reward learning or useful navigation.
No reward update exists in the current learner. Using reward values as class
labels would silently change both objective and provenance.

## Decision

Add an exact core policy-trajectory type and explicit native reward method while
preserving the existing supervised interface/snapshot schema. Minimize the negative episodic
sum of discounted reward-to-go times log action probability, with the outer
per-step discount stated explicitly. Full BPTT/native SGD stages before publication.
Require the current complete learned/optimizer/gradient state to match the behavior
checkpoint and recomputed action probabilities; refuse stale reward replay.

A small app collector composes existing key/door dynamics and frozen zero-reset
policy sessions. Caller-owned PCG64 state advances permanently once per decision,
with before/after receipts and refusal of external sampler changes. Preserve
terminal/truncated/illegal endings and partial attempts honestly. Roles never
release final data. Native admission, not a generic opaque-state framework,
protects against changed native sources before SGD.

## Alternatives

Treating demonstrations as policy-generated rewards would misstate provenance.
A new stream framework would duplicate accepted scheduling. Changing the default
CE update or adding objective metadata to old snapshots would risk compatibility
without a current requirement. Caller-owned explicit RNG state avoids hidden
global randomness or another retention/authority mechanism.

## Consequences

Component checks can establish objective/gradient correctness, atomic continuation,
actual sampling and isolation, not learned success. Reward trajectories use a
separate API and are on-policy only. Collection and native failures retain actual
costs; zero-reward updates can still spend momentum. Existing final failures and
all negative evidence remain preserved. A separate existing-runner RL recipe and
frozen local protocol remain prerequisites for scientific learning evidence.

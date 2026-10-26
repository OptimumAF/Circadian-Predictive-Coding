# ADR-0342: Condition reward likelihood on observed legal support

## Context

Preserved train-only sampled reward episodes terminate illegally with zero
return. The existing environment provides visible-dynamics legal actions, while
the collector samples the full raw distribution. A collector-only mask would
make native unmasked policy-gradient likelihood disagree with behavior.

## Decision

Add an explicit collector option and a distinct owned trajectory wrapper. Record
raw probabilities, actual legal support and the conditional sampling law. Native
admission recomputes current raw/public conditioning, verifies masked-logit
agreement and differentiates the conditional log-softmax under the original
discounted sum objective. A new typed update and diagnostic distinguish this law.

Reject zero mass, invalid/excluded support, stale states and underflow disagreement
before live publication. No resampling or fallback. Preserve old schemas, snapshot
formats, default reports and unmasked behavior.

## Alternatives

Repairing an already sampled action obscures behavior probabilities. Updating the
old unmasked loss after sampling a mask is mathematically inconsistent. Replacing
old defaults would erase the negative control. A new runner is unnecessary.

## Consequences

Legal support is extra environment information and incurs normalization/storage
cost. Future controls need the same information before comparative claims. The
change addresses a train-role collection prerequisite; no decision is tuned to
final outcomes and no learning, retention, generalization or CPC benefit follows
from implementation correctness alone.

## Game composition — 2026-10-16

Use the existing reward recipe and live evaluator behind an explicit CLI/protocol.
Defer new-route final packets until all training commits; use greedy conditional
live decisions and preserve raw teacher CE as a separately labeled diagnostic.
Keep support/raw metadata in typed packets and retain committed receipts before
boundary refusal. Why this: collection-only conditioning would leave user-facing
live behavior on a different law. The first eight-episode native recipe remained
zero reward, so correctness does not close the useful-learning requirement.

## Paired warm start — 2026-10-16

Compose the same legal live information into the existing BC-only/warm-start pair;
full native warm-state equality precedes conditional reward work. Collect shared
final packets only after both arms finish; reuse two local completion guards and
at most two detached completed-final records to reject later callback mutation.
Why this: same support is necessary for interpreting a live-policy comparison,
and second-arm evaluation must not rewrite the first completed result. Additional
RL exposure/SGD and inherited momentum remain unequal; the executed zero-reward
pair showed no live gain, so reward benefit remains unestablished.

# ADR-0394: Opt-in weighted BP in the existing learnability recipe

## Context

R2.1 has accepted native weighted BP and the sealed sequential array adapter.
The MiniGrid learnability pilot already provides fixed TRAIN streams, raw fit,
isolated fresh-task probes, old-task readouts and return phases. Passing a weighted
TRAIN schedule directly into its generic probe fork would refuse probe arrays or
apply the wrong objective. Existing negative acquisition results remain valid.

## Decision

Add weighted_bp=False / --weighted-bp, admitted only with explicit bound BP
qualification and public/invariant history. Compose the exact encoded TRAIN
schedule through TorchTrainWeightedLearner in the existing _train_arm runner.
Keep native family identity; name the distinct benchmark objective in the protocol
and validate every completed receipt. Capture permanent benchmark costs before
new receipt refusal. All defaults and ordinary recipes remain unchanged.

For each probe, build an ordinary bound BP zero-update source template and prove
full equality with the untouched original benchmark source. The original probe
application restores the chosen full checkpoint/optimizer on a disposable fork.
Original benchmark guards remain active. Probe/return updates, raw scoring and
prior-CE gradients retain their existing unweighted definitions.

## Alternatives

A new runner would duplicate release, budget and actor separation. Allowing the
weighted wrapper to accept arbitrary probe arrays would weaken its TRAIN binding.
Changing generic probe behavior for every learner would expand the affected scope.

## Consequences

One explicit weighted control is runnable with the existing diagnostics/probes.
Template construction, sealed schedule copies, validation and candidate work cost
time/memory; exposure equality is not compute equality. No weights, seeds, metrics
or baseline recipes are tuned. A separately frozen/reviewed small protocol is
required before scientific comparison. No new dependency, durable format or
runtime framework is introduced, and correctness alone establishes no benefit.

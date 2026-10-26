# ADR-0393: Compose weighted bound BP in the existing array adapter

## Context

R2.1 has accepted weighted native BP mathematics/state behavior, while the
sequential weighted adapter supported only recurrent learners. Existing delayed
streams consume arrays and must bind weights to released complete TRAIN episodes.

## Decision

Extend TorchTrainWeightedLearner with an exact bound-BP policy branch and reuse
its sealed schedule. Admit every public view before native resources. Match next
arrays exactly; derive existing weights from the current complete TRAIN episode.
Stage BP work privately, prove full candidate/source/input/identity/autograd/RNG/
mode state, then publish fresh detached components. Refuse foreign changes without
rollback. Reuse full native snapshots and fork for deepcopy. Preserve RNN paths.

## Alternatives

A second runner would duplicate existing label-release/budget/actor behavior.
Inferring missing episode metadata from arrays would lose complete evidence.
In-place BP rollback could erase an external change observed during candidate work.

## Consequences

Existing delayed runtime now executes bounded weighted BP schedules with named
receipts and owned checkpoints. The adapter retains at most512 episodes/current
expected state, not an unbounded success chain. Failed detached work still costs
computation. Snapshots alone do not authenticate past objective receipts. There
is no new dependency, CLI/default protocol, format or scientific-benefit claim.
The next opt-in research route needs a frozen matched protocol before comparison.

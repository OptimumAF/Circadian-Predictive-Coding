# ADR-0445: Continuous BC through the existing native learner port

## Context

R4.4 has a validated continuous demonstration boundary but no continuous learner.
The existing native port already supports opaque feature/target/state types.
Simulator availability and actual task quality remain separate prerequisites.

## Decision

Implement a configurable NumPy tanh MLP baseline with bounded physical output,
explicit half-width-normalized half-MSE over active axes and analytic derivatives.
Separate pure math in core from owned SGD/momentum/state in adapters. Use local
initialization RNG, full detached parameter/velocity snapshots and atomic candidate
publication. Reuse the existing port and role validation instead of a new runner.

## Alternatives

Using categorical IDs for continuous controls would lose units and bounds.
Raw physical MSE silently mixes differently scaled units; the chosen named loss
declares its normalization. Clipping would hide controller mismatches. A second
backend/dependency is unnecessary for this small CPU component; a future stronger
baseline can implement the same native contract under its own explicit objective.

## Consequences

Constant axes are fixed and excluded from the denominator. Restoring only weights
is insufficient for continuation; momentum/counters/full policy are required.
Raw arrays require caller release authority, while train_episode additionally
refuses evaluation or mismatched spec/roles. No disk serialization, simulator,
task reward, replay/promotion mechanism or scientific/robot success is implied.
Defaults need a separately frozen task protocol before any comparative claim.

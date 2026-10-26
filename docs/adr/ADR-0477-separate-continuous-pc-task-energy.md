# ADR-0477: Separate continuous PC task energy

## Context

The current robot BC reference predicts bounded multi-axis tanh actions with a
half mean active-axis normalized squared-error loss. The existing coupled PC
energy requires scalar binary targets and a Bernoulli logit likelihood. Reusing
that evaluator for robot actions would change the task or violate its contract.
R2.1 diagnostics and repeated-update infrastructure already have accepted
evidence; repeating them does not resolve this learner dependency.

## Decision

Add a pure continuous energy module using the existing `CoupledParameters`, row
biases and BC geometry/array validators. Keep the binary evaluator and both
existing BC likelihoods unchanged. Match the default BC task loss and output
head exactly for finite feedforward one-hidden fixtures. Add batch-mean,
neuron-summed hidden residual penalties with explicit bounded precision.

Expose owned target-free initialization/prediction and a free-state energy with
exact partial gradients. Independently check every coordinate at one, two and
four hidden layers using a handwritten scalar energy and CPU float64 autodiff.
Check overflow before `tanh`; an infinity must not become an accepted saturation.
No optimizer, adapter, runner, dependency or comparative experiment is added.

## Alternatives

- Coerce continuous labels into the binary evaluator: loses the actual task.
- Change the existing evaluator's default likelihood: risks historical evidence
  and obscures definition IDs.
- Add a full continuous learner and native comparison together: mixes the energy
  correctness prerequisite with update, ownership and experimental questions.

## Consequences

The new primitive is useful for continuous PC implementation, but alone does not
learn. Feedforward hidden partials are not BP total gradients. Mixed binary
gripper loss, bounded updates, learner state, neutral CPC parity, returned-context
recovery, costs and scientific/release claims remain separate open requirements.
The next increment must compose this primitive with existing stream interfaces,
not create a second orchestration framework.

## Follow-up: share the bounded inference loop

Continuous inference now composes the accepted energy with the existing guarded
binary loop, extracted into one private typed helper. The public binary API,
validation order, guard math and work counters are preserved. The continuous
wrapper owns its problem/policy and returns the last accepted free state on
unsafe proposals, with failed work counted. This small shared function removes
real algorithm duplication; no new runner or public callback framework is added.
The [continuous inference guide](../continuous-pc-inference.md) separates labeled
free-state outputs from unchanged target-free predictions. Parameter learning,
adapter ownership and matched neutral CPC remain open follow-up requirements.

# ADR-0326: Separate recurrent activation from frozen learned state

## Context

The actual local key/door game erases its cue before later decisions. Focused
tests prove observation aliasing and the accepted native feedforward case fails
all320 live episodes. Existing native learners have no episode history. A useful
memory-capable control must learn recurrence without giving it the hidden cue
again, changing old objectives, or weakening frozen evaluation guards.

## Decision

Add a small CPU tanh Elman recurrent BC adapter through the existing NativeLearner
array/snapshot/fork interface. One complete ordered episode per call starts at
zero state; full BPTT minimizes mean action cross entropy with native SGD. An
existing DemonstrationEpisode convenience entry point rejects final-test roles.
No role authority is inferred from raw arrays.

Learned state includes parameters, gradients, SGD/momentum, policy and counters;
atomic staged publication and strict owned restore follow the existing adapter
pattern. Initialization has a private generator and no RNG is used afterward.
The new component always returns full categorical distributions, including K2;
old binary output contracts stay intact.

A fresh bounded live session owns its learner copy and separate hidden
activation. It compares actual complete learned state against its frozen birth
state before and after prediction, stages activation before publication, and
never trains or renews runtime authority. Common evaluation identity explicitly
adds recurrent_bc. Use the existing stream/shared packet replay/live evaluator.

## Alternatives

- Reintroducing the cue or adding a remembered target to features would bypass
  the demonstrated memory requirement.
- Broadly ignoring learned-state mutations during recurrent evaluation would
  weaken existing isolation.
- A new orchestration framework or recurrent RL now would enlarge this component
  change before its state and mathematical boundaries are verified.

## Consequences

There is an explicit trainable memory control and a tested integration boundary.
Architecture, memory and unrolled computation differ from feedforward controls;
future comparisons must report that cost, not imply matched native objectives.
Manual history fixtures and BPTT parity establish behavior, not acquired cue use.
No comparative run or scientific advantage follows from component acceptance.
Useful key/door acquisition/retention/recovery/generalization and separately
evaluated recurrent RL remain required under original R4.2.

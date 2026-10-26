# ADR-0392: Explicit weighted CE for encoding-bound feature BP

## Context

The completed canonical-PC TRAIN/F diagnostic improves CE but predicts only the
majority action after32 updates and never acquires exact traces. This supports a
class-imbalance hypothesis, not a causal finding. Existing complete-episode
weight derivation and weighted recurrent CE already exist; fixed-feature BP has
no explicit weighted update port. Previous neural/weighted negatives stay valid.

## Decision

Add a separate adapter function on the exact bound native BP owner. Use existing
TRAIN role/schema/public encoding/weight validation, existing head/SGD state and
sum(target_weight*CE)/sum(observed_target_weight). Return a distinct objective
identity. Build/validate a detached complete candidate and publish atomically
only after donor/input/RNG/mode proof. Do not change existing train_batch behavior,
policy/state formats or PC/CPC laws. Keep weight derivation/budgets at the caller.

Why this: an explicit control isolates a testable objective without changing the
preserved unweighted baselines or adding a runner. The existing fixed formula is
reused; no weights are selected from probe/final performance. Mathematical and
state correctness must precede a matched comparison; success is not assumed.

## Alternatives

- Silently weight the original BP baseline: rejected because it changes the
  experimental condition and obscures existing negative evidence.
- Add weighted PC/CPC concurrently: requires a separately derived latent energy/
  update definition; outside this small BP prerequisite.
- Create another stream framework: existing weighted schedule/stream ports are
  the next composition boundary.

## Consequences

Only complete TRAIN296/K7 episodes enter the new function; no_grad/inference mode
is refused. Next views validate complete evidence but never become CE inputs.
Failed candidates do not publish; detected external changes are not overwritten.
Candidate computation/validation/copies and failed SGD steps need caller budget
charges. Snapshot counters alone do not account for every failed attempt or prove
historical objective identity. Native snapshots/forks remain ordinary owned state.

119 distinct scoped cases qualify by118 unchanged passes plus one corrected local
oracle test, not one119-test green invocation. Scoped statics and independent
review qualify implementation only. Sequential array-port integration, fresh-task
comparison, useful neural skill, R4.2/release and the full goal remain open.

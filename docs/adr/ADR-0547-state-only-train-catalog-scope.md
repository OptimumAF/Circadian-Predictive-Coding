# ADR-0547: Explicit state-only TRAIN catalog scope

## Context

The accepted disposable state-only probe showed partial old-task retention
improvement and a worse new-task return suffix result. It did not update the
benchmark learner. A continuing TRAIN owner is required to investigate whether
this conventional constraint can help across a real A/B/return sequence.

## Decision

Add `CLMStateOnlyCatalogLearner` and exact `CLMStateOnlyCatalogSnapshot` in a
separate adapter. Reuse the existing catalog TRAIN admission and transactional
native engine. Only the state projection enters AdamW and gradient clipping.
Action projection and temperature retain their numeric contribution but have
no loss gradient; their owned values remain bound to one fixed birth pin.

Keep the existing all-head learner, default runner and snapshot schemas intact.
The new snapshot type identifies both catalog objective and optimization scope.
Restore validates fixed values against the receiving owner's birth before
publication; it cannot redefine those values or renew physical work allowance.
The temporal script selects the new learner with `--train-state-only` and uses
the existing twelve-update stream. Probe scopes remain independently selected
on separate owners. No probe weights or labels are installed into TRAIN.

## Alternatives

Changing `requires_grad` on loaded heads would silently change declarations and
make ownership checks ambiguous. A global policy field would migrate existing
snapshot formats. Importing the private probe learner would couple TRAIN to a
different data role. A new stream controller would duplicate working orchestration.

## Consequences

The short loss/invariant implementation duplicates the accepted private probe's
semantics deliberately; direct parity tests cover that boundary. This avoids
changing its accepted objective or role while exposing a different public
lifecycle. Copy work includes a bounded fixed birth pin. Optimizer and backward
work differ from all-head learning even at equal update/row counts.

Typed frozen binding currently admits the original exact catalog source/snapshot
family. The new CLI combination refuses before loading; extending binding requires
its own scope-aware admission tests. No serving, generalization, circadian advantage
or native task benefit follows from this software increment. A comparative native
sequence requires a separately frozen and reviewed small protocol.

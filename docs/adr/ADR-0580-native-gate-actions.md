# ADR-0580: Apply finite plasticity gates to one staged native update

## Context

R7.1 declares gate settings but the existing action runtime refuses execution.
Wake/replay/rate already use a fixed catalog and original TRAIN/shared admission.
Changing a learner's config to toggle its gate would conflict with complete
snapshot validation and could reset or silently alter other adaptive mechanisms.

## Decision

Add an optional native gate port and exact finite-string validator. Forward that
scalar through the existing resource wrapper, actor runtime, inbox and native
step. Refuse incompatible owners/combinations before resources or payload copy.
Keep unchanged on the original public training method. Gated CPC active uses its
bound plasticity rule; neutral uses ones for the existing plasticity multiplication
only. Neutral CPC accepts its own neutral setting; BP/PC reject nondefault gates.

The head accepts the scalar on one call and keeps chemistry, reward, importance,
traffic, age and sleep history native. The adapter stages a complete state and
publishes only after validation. No flag is persisted, config edited, model family
converted, snapshot criterion relaxed or new optimizer introduced. The counted
ticket learner forwards nondefault gate calls through its original work counters.

Why this: an update-scoped effect satisfies the finite action requirement without
changing immutable model identity or hiding the cost of adaptive transitions.

## Alternatives

- Temporarily mutate config or monkeypatch plasticity: rejected because mutable
  hidden state and exception paths make exact restoration harder to establish.
- Convert gated CPC to the neutral model family: rejected; reward/chemistry and
  complete history would acquire different meanings.
- New controller/runner or durable action ledger: unnecessary; original owners
  already enforce causality, resources, receipts and budgets.

## Consequences

The controller can select this finite learner effect, with unchanged defaults,
native atomicity and explicit requested-setting polls. It cannot use the effect
to tune against final evaluation or change permissions. Sleep/renewal and mixed
effects remain refused. Scoped correctness and the three-update engineering
example do not establish comparative or task benefit. Source/procedure/results
were self-reviewed, not independently accepted. Evidence: issue64 and
`artifacts/runs/bounded-gate-action-20261023/RESULTS.md`.

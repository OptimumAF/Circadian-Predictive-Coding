# Applying an existing fitted chooser to an ordinary TRAIN window

`AttributionStream.run_contextual_window(controller)` applies an exact,
already-fitted `CounterfactualController` through the same finite loop used by
fixed references. It uses the existing `choose_contextual` wait/wake1 rule and
internal oldest-ready TRAIN diagnostics. It adds no fitting algorithm or budget.

Why: the conventional MiniGrid path now learns and retains known tasks, but the
ordinary stream previously admitted only fixed `ReferencePolicy` objects. The
existing four-trial chooser needs complete-source binding before it can select
updates on that path. A bare `FittedController`, arbitrary chooser callback or
ticket fit with a different native payload is refused.

The ordinary Python composition is:

```python
from src.app.counterfactual_controller import CounterfactualController
from src.app.coupled_attribution import AttributionStream
from src.core.actor_ports import VersionedState

controller = CounterfactualController(
    lambda: VersionedState(actor_version, source.snapshot_state()),
    original_state_digest, prospective_objective, original_fit_clock,
)
# Four causal controller_train trials, one wait/wake1 return per clean/mistakes
# context. The existing owner supplies independent copies of its original base.
for observation in four_controller_train_observations:
    controller.run(observation, execute_disposable_trial)
controller.freeze()

stream = AttributionStream(
    source, training, (), training_only=True,
    checkpoint_updates=tuple(range(len(training) + 1)),
    label_delay=1, budget=original_application_budget,
)
result = stream.run_contextual_window(controller)
```

The caller supplies and budgets real trial execution, complete measures and a
stable digest port. `actor_version` must match the stream's original actor
(default `a0`, or the explicitly supplied actor's version). Source payloads must
match completely, including native optimizer/history/RNG state where present.
Fit and application logical epochs remain distinct; a fitted-at value is not
silently relabeled as an application tick. No fifth trial/freeze occurs here.

The fit owner retains at most four original returns and one frozen evidence pin.
`application_snapshot()` refuses unavailable, unfitted, stale, stopped, future or
changed fit evidence, including equal-valued replacement of original records.
It returns detached original/current source payloads and fit data. The stream
checks both complete payloads against its actual source and actor version;
a constant digest cannot authorize a different payload. The existing candidate
lease and original clock protect each callback, with native state checked against
the last owned checkpoint and after the callback. Source/actor/budget/role and
callback checks remain active. These are trusted local snapshot ports, not a
hostile-Python execution boundary.

The stream must be pristine, ordinary TRAIN only, with no heldout/probe data,
all0..n checkpoints and finite original credit. The chooser receives causal
scalar `ReferenceObservation` and frozen policy only. Predictions use a separate
candidate fork; labels enter mistake counts only after their original arrival.
Choices go through the original FIFO `train_ready(max_updates=1)` path. A correct
oldest packet can remain queued; no acknowledgement, skip, implicit drain,
renewal, promotion, final access or tuning is added.

`result["controller_application"]` reports binding-read attempts/completions and
unknown returns, separately from application diagnostics/updates/pending work.
Its `fit` retains four original request/return/measure records, completed and
discarded trial counts. `ControllerMeasures` provides execution-time, sampled
RSS and violation proxies; it does not supply unrecorded native update/FLOP
counts. Keep the external trial work ledger. Do not sum sampled RSS values as an
aggregate peak. Fit costs are never application budget credit.

`window_complete` still differs from `training_complete`. Partial or failed
windows preserve receipts, attempted/unknown work and existing charges, and seal
against reentry/evaluation. `run_reference_window` keeps its exact fixed-policy
entry and reports; counterfactual `snapshot()` keeps its original fields.

[Software qualification](reports/attribution-contextual-window-20261026.md)
uses real owners/stream with literal learners and scalar trial fixtures. No
native learned-controller experiment, public CLI option, utility advantage,
generalization or release qualification follows. Freeze any such protocol first.

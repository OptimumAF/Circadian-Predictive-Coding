# Prospective controller utility and chronology

The existing native BP serving demo has an opt-in engineering study:

```console
python -m scripts.run_bp_ticket_serving --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --outer-study --output outer.json
```

The source is the exact qualified96-update/576-row checkpoint. The output must
be fresh; `--outer-study` and `--reference` are mutually exclusive. This is a
fixed periodic reference, with period2, rather than a trained controller.
It runs at most eight new updates/48rows across two independent learner owners,
then evaluates detached frozen readouts with zero sealed training or promotion.
Use the already installed native dependencies described in the
[serving demo](native-ticket-serving.md). Existing default/reference modes remain
available. The saved-ticket HTML inspector does not yet accept this report schema.

## Fixed utility

`core/controller_objective.py` defines immutable scalar records and the pure
calculation. Default weights are all1, declared before model work:

```text
gain   = (new_after - new_before) / new_task_count
drop   = max(0, (old_before - old_after) / old_task_count)
cost   = execution_seconds / 120
memory = sampled_peak_process_RSS_bytes / (512 * 1024**2)
risk   = applied_unauthorized_or_stale_effects / 1
utility = gain - drop - cost - memory - risk
```

Negative gains and costs above reference denominators are preserved. Incomplete,
unknown, missing or nonfinite work/cost observations have no score. Counts must
be bounded integers. No missing-cost imputation or cost clipping occurs.
Correct effect-time refusals are not violations. Legal wrong-task actions reduce
task success but are not automatically safety violations.

Execution wall time covers the complete trial callback, including source/model
construction, TRAIN monitoring, updates, guards and task readouts. Sealed source
construction and readouts are inside its timer and RSS sampler. These are partial
cost proxies: sampling may miss peaks, process RSS includes framework residency,
wall time is not FLOPs, and report serialization/outer bookkeeping lies outside
the individual callback. Full native constructors, forwards, snapshots, typed
requests, dispatches, TRAIN rows and outer elapsed costs are recorded separately
by the bounded engineering procedure. No efficiency comparison follows from
these instrumented timings.

## Three causal stages

`app/controller_chronology.py` holds at most three detached scalar rows and one
freeze identity. It uses the original LogicalClock. The application supplies
trial callbacks; the boundary does not run models or own label/data stores.

| Stage | Actual source and owner | Outer labels / return |
| --- | --- | --- |
| controller_train | New-context TRAIN arrays; fresh qualified96 owner |1 /2|
| controller_validation | Old-return TRAIN arrays; separate qualified96 owner |3 /4|
| sealed_evaluation | New episode identities; readonly original96/development candidate |5 /6|

IDs and source groups are distinct and fixed in the protocol. Actual development
and validation arrays differ. Their underlying learner feedback keeps its
ordinary TRAIN role and original arrival clock. The outer event clock is a
separate, explicit trial-outcome clock, not a native update count.

Training feedback is exposed only after the first stage and before validation;
validation feedback only after its stage and before freeze. Final rows are never
returned by those interfaces. Freeze at4 is identity-bound to the original
objective, fixed policy, streams and clock. The sealed source callback cannot
open until that receipt passes admission. Role/group/policy metadata and causal
returns are checked before metrics; duplicate/reentrant/stale/forged releases and
mutated plans fail. Failed paid attempts retain their prefix, stop the study and
cannot be retried through the same owner. This is a trusted local Python contract,
not hostile-code isolation or a durable checkpoint/permission mechanism.

Within each native development/validation phase, the existing stream, finite
action admission, monitoring, budget, publication guard and rollback remain
authoritative. Nine opportunities admit four periodic updates and explicitly
leave four batches unapplied. `AttributionStream.finish` retains its original
full-completion semantics. Candidate task readouts remain separate from deployed
actor outcomes. Sealed scoring uses the development candidate even if publication
was refused; it cannot promote it or return feedback to a chooser.

## Scope and extension

The public grammar, encoder and historical pretraining are shared. New episode
identities and distinct source groups do not establish feature novelty,
statistical independence, held-out task generalization or meta-generalization.
This study freezes one reference before all stages; it does not fit/select a
policy using the training or validation rows. The broad scientific chronology
and learned-controller requirements therefore remain open.

Why this: make the prospective objective and role barriers executable through
the existing runner before adding controller optimization. A learned controller
will additionally need immutable trained-policy/version binding, counterfactual
learner snapshots, stale-outcome rejection and accounting for discarded work.
Do not rescore past final/reference outcomes to choose a policy or objective.
Preserve all prior PC/CPC failures and negative heuristic outcomes.

Focused checks (with existing native dependencies):

```console
python -m pytest -q tests/test_controller_objective.py tests/test_controller_outer_study.py tests/test_reference_controller.py tests/test_torch_serving.py
python -m ruff check src/core/controller_objective.py src/app/controller_chronology.py scripts/run_bp_ticket_serving.py tests/test_controller_objective.py tests/test_controller_outer_study.py
python -m ruff format --check src/core/controller_objective.py src/app/controller_chronology.py scripts/run_bp_ticket_serving.py tests/test_controller_objective.py tests/test_controller_outer_study.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent src/core/controller_objective.py src/app/controller_chronology.py scripts/run_bp_ticket_serving.py tests/test_controller_objective.py tests/test_controller_outer_study.py
```

Exact frozen limits, commands, failed attempts, source pins, raw reports and
costs are local in `artifacts/runs/controller-outer-chronology-20261023/`.
Live acceptance/handoff: [issue52](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/52).

The sole engineering study completed with four development and four validation
updates/48total rows. Both phases and sealed readouts stayed at new1/2, old2/2;
both publication guards refused. Sealed readout made24typed calls with zero
training/promotion. Utilities were -0.644462, -0.627600 and -0.621811, including
the fixed cost penalties. No acquisition, forgetting recovery or controller
advantage was demonstrated. Outer execution16.528s and8.747s metered work are
instrumented observations, not a latency benchmark. These results are preserved
without a numerical retry or objective/policy change.

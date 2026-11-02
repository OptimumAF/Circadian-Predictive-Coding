# Staged CE rehearsal updates for disposable probes

The post-repair original-clock qualification remains **FAILED/SPENT**: fresh16
retained4old/learned2fresh known tasks, but source30s expired during return.
At return4 both fresh classification accuracies fell to2/3; after-return task
outcomes remain unknown. Validation-cost improvements are not full runtime
qualification or scientific benefit. See [scoped evidence and costs](reports/ce-postrepair-qualification-20261030.md).

`src.adapters.torch_ce_probe_update.fit_ce_probe_update` computes one detached
output update from the current batch and retained TRAIN anchors. It returns a
distinct `CEProbeUpdate`; it never updates the supplied learner. This port is
wrapped by `apply_ce_probe_update` for the existing `run_fresh_task_probe`
loop (see #342 below). Ordinary ticket script/CLI composition is available as explicit ce_rehearsal
(see #343 below); native chronological qualification remains separate.

```python
update = fit_ce_probe_update(
    private_learner,
    current_probe_train_batch,
    retained_train_batches,
    phase="fresh",
    check_boundary=original_probe_boundary,
)
```

Inputs are exact `AttributionBatch` records. Fresh current role is `probe_train`;
return current role is `train` with `phase="return"`. Retained roles are always
`train`. Held-out/controller roles and role subclasses are refused before owner
or optimizer work. All rows use hard class targets, the same feature/class
geometry and at most128 combined rows. Retained batches have unique row IDs;
fresh IDs cannot collide with them. Return TRAIN IDs may overlap only when
context, features and targets agree, with that duplicate exposure explicitly paid.
The port validates batch contracts; the caller still owns arrival/provenance and
release authority. A role string is not proof of an original stream receipt.

The existing fixed float32 CE optimizer fits detached output parameters with
frozen hidden features. No probe-label replay bank, optimizer history or input
list is stored in a successful proposal. The result contains owned before/after
native snapshots, coefficients, bounded losses and scalar work.

`copy_ce_probe_update` revalidates internal state, phase, row counts and work.
The expected after-state changes only output parameters, preserving policy,
hidden parameters, gradients, momentum and native SGD counters. A receiving
caller must compare the proposal's before-state with its still-current private
owner and apply it only after its original source/budget/evidence guards pass.
The proposal grants no source restore, actor promotion, benchmark update or
replay permission. Existing TRAIN fit consumers reject this distinct result.

Work separates `probe_gradient_rows`, `current_train_gradient_rows` and
`retained_train_gradient_rows`; `native_sgd_updates` is0. Counts use actual
backward attempts, including failures. Forward/read work is totaled separately;
callers retain original phase/row metadata for its role allocation. `CEProbeUpdateFailed.work` preserves paid
fit work when no proposal returns. Foreign source/input mutations are refused
and remain visible. Callers must budget closure/rehearsal work and elapsed/RSS;
the original optimizer's hard64-closure limit does not replace those budgets.

[Tiny executed fixtures and costs](reports/ce-rehearsal-probe-20261030.md) establish
component behavior. They do not establish native fresh-task acquisition,
retention/recovery, held-out benefit, matched compute or CPC superiority. The
existing TRAIN-only fitter and ordinary SGD/default probe remain unchanged in
their data-role semantics. Opt-in app runner integration is available below;
ordinary ticket script/CLI integration is available in #343 below.


## Existing disposable runner integration (#342)

Pass `update_callback=apply_ce_probe_update` to `run_fresh_task_probe`, alongside
explicit `training_readouts` containing the released retained TRAIN batches.
Import the callback from `src.adapters.torch_ce_probe_update`. The ordinary
runner still creates/restores the private fork, scores its fixed checkpoints,
and uses that same private learner for an explicitly requested return phase.
Omitting the callback retains the original SGD/report/delegate behavior.

With the callback, TRAIN monitors are explicitly also rehearsal anchors;
fresh current remains probe_train and return current remains TRAIN. Only these
records reach fitting. Held-out probe_test targets remain application metrics.
The callback refuses stream rehearsal, extra F exposure and short affine mode;
it requires the existing native BP policy and fixed role/geometry bounds.
The ordinary CLI selects this callback with its existing --probe-update option
as described in #343 below. There is no probe-label replay bank.

Top-level `attempted_updates` / `completed_updates`, checkpoints and the original
ToyBudgetSession count callback application slots in this mode, not native SGD.
Native snapshot counters, gradients, momentum and hidden parameters remain
unchanged. `diagnostics` stays native-only (empty for these callbacks).
`auxiliary_updates.events` records phase, current rows, owned scalar receipt
and actual CE/application costs, with at most16 fresh and16 return events.

A typed failure retains observed fit/backward counts. A verified application
followed by a failed boundary remains charged as a completed slot, while the
event/phase fails and admits no later work. An interrupted restore can have an
attempt/return without verified application; it is never retried or rolled back.
Untyped failures keep work_known=false rather than claiming zero paid work.
Only the actual native fields attest to application, with before/after owner and
RNG checks. Original caller/source/clock/input/preceding-result guards still apply.

`retained_data_array_bytes` names the extra input-integrity copy of all admitted
cohorts (including metrics-only arrays). It is transient guard storage, additional
to reserved/readout/return arrays and native/report/temp memory. The callback
receives no held-out targets and persists no bank of earlier fresh labels.

This integration passed tiny fitting/state-isolation and scripted fresh/return
checks. It does not establish the ticket workflow's sequential acquisition,
retention, recovery, generalization, matched computation or CPC benefit.
See [execution report](reports/ce-probe-integration-20261030.md).


## Ordinary ticket probe composition (#343)

Select `probe_update="ce_rehearsal"` with ordinary `method="bp"`,
`context_return=True`, `reserved_probe=True`, `probe_initialization="ce"`.
The existing CLI option accepts `--probe-update ce_rehearsal` with
`--context-return --method bp --reserved-probe --probe-initialization ce`.
Omitting this selection preserves ordinary SGD; linear minimum_change is
unchanged. Native SGD diagnostics/output-only and other alternate modes are
refused before source work. The ordinary output path must be absent.

This uses the original source12/72 checkpoint and CE initialization, fixed
fresh16/return16 cohorts, private fork and six typed task readouts per phase.
Original source/actor/initialization/receipt/clocks and fixed session limits
remain authoritative. CE applications require32slots/144currentrows and
zero private native SGD calls. The unchanged default requires its original
32nativeSGD/144rows/32returns. No inference of equal work follows from32slots.

`scripts/ticket_ce_updates.py` seals a detached view of scalar auxiliary work
while preserving actual receipts privately. Replaced/corrupted public evidence
refuses further work without becoming a new baseline. Verified application
before a late failure stays paid. No current/future/held-out labels are stored
in this adapter; the existing app passes only current batch + retained TRAIN.

`reserved_probe.preserved_ce_rehearsal_work` contains attempts, current rows,
returns, applied slots and bounded32 scalar receipts. Root `probe_updates`
continues to count native private updates (zero here); `probe_auxiliary_updates`
reports the separate CE work. `reserved_probe_summary.reported_costs.ce_rehearsal`
aggregates known fit/rehearsal/role/application counts; missing work stays unknown.
The ordinary writer's failure fallback exports this paid work separately.

Scripted ordinary stream/probe/return/finish checks verify composition. Actual
CE math/private application is reused from #341/#342. No full native ticket
chronological outcome is claimed. Freeze a small local protocol with exact
source/fresh/old/return readout, rehearsal/CE/hidden/forward/gradient/storage/
wall/RSS costs before launching a new native study.
See [commands, failures and qualifications](reports/ordinary-ce-rehearsal-20261030.md).


## Native observation (#344)

One frozen ordinary CE rehearsal case produced a valid fresh16 prefix: old
closed-loop tasks4/4 retained, fresh tasks0/2 ->2/2. The full case FAILED/SPENT
when the original continuous30s source clock expired in the second return fit;
only1/16return updates completed, end-of-return outcomes unknown. Source/actor/
bothRNG preservation flags were true. No retry or clock reset.

F prefix cost2520gradient-row attempts including source/init; entire failed
attempt3186, native32.164932432s/347885568B. Saved initialized SGD controls pass
all available common-field checks before reuse; bundle and unequal-work
qualifications remain. No generalization, CPC/recovery/full-case/release claim.
See [full evidence and costs](reports/ticket-ce-rehearsal-native-20261030.md).
Next is bounded cost diagnosis preserving the original clock, not another sweep.

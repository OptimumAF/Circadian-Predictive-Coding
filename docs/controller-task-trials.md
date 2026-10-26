# Closed-loop controller-development trials

```powershell
$env:PYTHONPATH = (Get-Location).Path
.venv\Scripts\python.exe -m scripts.run_bp_ticket_serving --source artifacts/runs/context-head-control-20261023/actual.json --context-controller-task-trials --output local-controller-task-trials.json
```

Output must not exist. Budget the invocation before running models. This explicit
option uses the existing disposable controller trials and typed workflow runner.
It freezes a four-row scalar fit from complete closed-loop development outcomes
and performs no live application. The original `--context-controller-learning`
classifier-based fit/application and its saved results remain unchanged.

The first scientific invocation failed saving its report because the new mode
inherited the CLI's1MiB limit. The limit is now3MiB and metadata-only boundary
checks pass, but the original run was not repeated. Its measured work is retained;
task outcomes and fitted policy were lost and remain unverified. No successful
new fit, validation result or benefit is claimed.

If a future task-trial report exceeds3MiB, the CLI saves a FAILED summary before
raising the same error. It retains producer-reported training/readout counts,
elapsed time, sampled RSS, original budget/clock scalars and bounded error
prefixes for at most four trials. The summary is at most32KiB, marks missing or
invalid scalars null (never zero), and explicitly marks the detailed report,
task outcomes, fitted policy and trial receipts unavailable. It cannot validate
or reconstruct those details, admit more work, or renew an allowance. Accepted
output bytes, exclusive output creation and other modes' limits remain unchanged.
Fourteen metadata-only checks per platform qualified this failure path with
zero model work; this does not recover the original failed fit. See
`artifacts/runs/task-report-failure-20261023/RESULTS.md`.

## Procedure

Qualified complete source state is copied independently for four controller-TRAIN
trials: mistakes/wait, mistakes/wake1, clean/wait, clean/wake1. Mistake context uses
three v1legacy TRAIN rows; clean uses six v1invoice/password rows. Original causal
delay, version/digest binding, one-update budgets and deterministic tie rule stay
intact. All discarded candidates, monitoring and readouts are paid.

Each candidate is measured before and after on six ordinary tasks in fixed order:
v1invoice, v1password, v2invoice, v2parcel, v1legacy, v2legacy. Old denominator4/new2
are declared for this development metric. Private actors generation0 with explicit
controller-before/after versions execute the existing `_episodes` and typed
predictor. No actor is published or deployed. `controller_train` is admitted
before prediction; final/probe/guard metric sources do not supply task truth.

The task summary requires both exact phase buckets, all six original tasks per
phase, complete episodes/reads and expected version/native counters. Unknown or
changed roles, extra buckets, source state, work or outcomes stop fit. Paid
prefixes remain. Previous paid records are checked before another trial. No
incomplete task trial has utility. The existing objective retains its unit
weights, 120-second compute reference, 512MiB memory reference and violation
reference1. Violations are applied unauthorized/stale effects counted by the
existing helper. All four estimates must complete before policy freeze.

## Interpretation

These are controller-development tasks on known teacher templates. Their
ordinary-task metric is distinct from earlier classifiernew6/old12 and from the
live17–22 window containing intentional denial22. It does not rewrite either
result. Private shadow task success is not guarded deployed-actor success: a
single-step trial does not model later repair, publication refusal or long-term
effects. A positive fitted utility alone cannot establish controller benefit.

Full trial elapsed time and sampled process RSS include task execution and
ownership/copy/accounting overhead. Serial order and retained reports make costs
unpaired descriptive proxies; outer monitoring/fit costs remain separate. No
amortized efficiency, CPC, generalization, latency or release claim follows.
Final/probe data remain unavailable; validation needs a separately frozen protocol
and cannot refit from its results. Existing fixed controls and negative evidence
must be reused when their assumptions remain valid.

Evidence/protocol: `artifacts/runs/controller-task-trials-20261023/` and
[issue87](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/87).
Extend through the existing trial/task ports, preserving causal roles, original
objective and full work accounting rather than adding another runtime.

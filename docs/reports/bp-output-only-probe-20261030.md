# Explicit output-only BP probe updates - #336 / R2.1-BP.OUTPUT

Previous335 was progress: a saved-data calculation matched native first-step
losses and attributed the reference explosion to the hidden-parameter change.
Output-only avoided that jump but did not fix READY classification. This
increment implements that explicit update rule and verifies correctness; it
does not repeat323 or claim improved native adaptation/retention/recovery.

## Working behavior

`TorchFeatureLearner.train_batch_output_only` uses the existing staged BP path:
one full forward, mean CE, backward and SGD call. Hidden gradients are temporarily
None only during SGD, so hidden parameters and existing momentum buffers are
skipped. Computed hidden gradients are restored afterward for complete native
snapshots/diagnostics. Output parameters use the original SGD, including its
original output momentum, learning rate and group settings. No momentum reset,
rescaling, weight rollback or new optimizer. This saves no backward work.

Cold momentum snapshots lack all buffers. Missing hidden buffers are explicitly
initialized to zeros so the original strict four-buffer validation remains
unchanged after the first update. No existing buffer is changed; momentum0
retains an empty optimizer-state dictionary. Warm hidden parameters/buffers stay
bitwise equal in value. Staged post-step failure never publishes a partial
payload; source/live state remains original. Wrong families refuse before data
or model work. The ordinary full-update port retains its original SGD branch.

The existing ordinary/API option is `probe_output_only=True` / CLI
`--probe-output-only`. It requires BP, context-return and reserved-probe mode,
and composes with optional CE initialization and update diagnostics. Source
training, seed/actor and other families stay on their original paths. Existing
LS `--probe-update` modes are untouched; the new switch is separate.

The private case and optional preserved diagnostic table record
`parameter_update_scope="output_only"`. Existing paid metadata guards refuse
scope corruption before another update. Canonical failure-prefix scope is also
used in the bounded-writer fallback, preserving changed foreign values in the
raw incomplete report. Default calls omit the new report fields and keyword
forwarding. The norm residual uses learning rate times stored post-update momentum. For
frozen hidden parameters this is the preserved old buffer, not a reconstructed
counterfactual full-SGD direction. The residual is descriptive, may be nonzero,
and is identified by the explicit update scope.

API example for an already owned native BP learner:

```python
diagnostic = learner.train_batch_output_only(train_features, train_targets)
```

Ordinary CLI configuration:

```text
--context-return --method bp --reserved-probe --probe-initialization ce --probe-output-only
```

These specify code behavior, not authority to repeat a spent experiment. All
final/probe-test labels remain excluded from learning. The existing private
TRAIN/F-return machinery owns role checks, clocks, budgets and source guards.
Neither disposable output parameters nor diagnostics are promoted to source.

## Executed checks and costs

New test/report/stage paths were checked absent before creation. Five exact
preimages and scoped.diff retain this increment separately from inherited
dirty work. Legacy torch_bp_update_diagnostics.py was not changed;334's
restoration persists. Files changed: existing torch_feature_learner.py,
ticket_native_probe.py, ticket_ce_probe.py, recurring_ticket_workflow_example.py,
existing test_probe_update_diagnostics.py; new test_torch_feature_output_only.py.

Both platforms:55PASS/0failed/0skipped, one cohort each. Warm paired updates
verify bitwise hidden parameters/momentum unchanged, output parameters/momentum
and all computed gradients equal the coupled step at the SAME initial state,
loss equality, counters and RNG preservation. Cold .9/0momentum snapshots pass
the unchanged validator. An optimizer step that mutates staged output then
throws leaves the original payload identity and full state intact.

Existing26norm/CLI/failure checks include one additional interrupted tiny
output-only private prefix: actual dispatch/freezing, diagnostic composition,
canonical scope and corruption refusal. New API/CLI/CE/fallback checks cover
early admission and default omission. Source12/cohort/task/memory admission
seams in that tiny integration fixture are SCRIPTED. No genuine source12,
CEfit,16-update F/return or closed-loop task qualification occurred here.

Commands from physical `D:/Dev/Repositories/Circadian`, PowerShell TEMP/TMP
`D:/Dev/_Temp`, WSL TEMP/TMP/TMPDIR `/mnt/d/Dev/_Temp`:

```text
.venv/Scripts/python.exe -B artifacts/runs/bp-output-only-probe-20261030/check.py windows
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/bp-output-only-probe-20261030/check.py linux
```

Windows4.476922699977877s/329,994,240B; Linux13.79534533s/346,345,472B. Linux
session80742 was polled to terminalEXIT0, never restarted. Each recorded
46native components/92seeded draws460values,16SGD attempts and actual steps,
16forwards/backwards48rows,11summary attempts. Across both:32SGD96rows,
92components184draws920values,32forward/backward/SGD calls96rows. All per-run
64component/20SGD60row/90s512MiB and aggregate180s limits held, including the
intentional post-step failure. There is no new scientific model allowance.

Zero auxiliary CE-readout fitting/LBFGS/LS/LP/PC-latent/typed tasks/robotphysics/
final/GPU/remote work. Native mean CE and full backward ARE computed.
Full backward remains paid for output-only updates; no equal-compute or
efficiency advantage is claimed. State validation/copies and norm work add
overhead represented by counts, elapsed/RSS and preserved report work, not
independent tensor-copy-byte attribution.

Final Ruff/format7files and default+Linux-target mypy6sourcefiles PASS; expanded
argv/output/times in statics-*.json. Linux-target mypy ran on Windows. Initial
six test Optional-field type errors were fixed before native fixtures. One
failed formatted patch anchor made no edits, then smaller exact patches
succeeded. No native fixture failure/retry/budget extension. Sys capture and
distinct admitted336temporary directories preserve prior failed334evidence.

The stage is artifacts/runs/bp-output-only-probe-20261030/ with before-files,
scoped.diff, windows-work.json/linux-work.json and exact static receipts.
Cache D:/Dev/_Temp/circadian336-mypy; no stage cache. No dependencies, installs,
deletes, global settings changes, commit/push or unrelated reset.
Broad suite/unchanged family math/legacy135-row cohort/source12/F-return/
new wheel/clean clone/browser/GPU/remoteCI/release/sweeps skipped with scoped
rationale. Default core branch review and focused actual BP controls cover this
change; they do not establish repository-wide or other-family qualification.

## Acceptance and next action

Lead self-reviewed staged ownership, strict state shape/momentum membership,
cold initialization, hidden-gradient restoration, source/private mode routing,
default branch, canonical failure metadata, fixture outcomes and costs.
Independent final source/results/preservation review ACCEPT: five read-only
commands, all EXIT0, 0.9075042 seconds within six-command/30-second allowance.
Reviewer wrote only stage/REVIEW.md; no numerical execution, project imports,
tests, models, network or child agents. Finite exception ended.
Self-review is separate; no independent numerical replication is requested.

Completed software does not establish fresh-task acquisition or native task
retention/recovery. Original323 negative and first FAILED/SPENT capture,331
FAILED/SPENT,335 reference limits,333 capacity and328 simpler LS findings/caveats
remain unchanged. Full R2/R4/R5/R7/matched science/release/repository goal open.

Next precise action: freeze and review ONE small native ordinary output-only
variant using the accepted source/CE initialization and original F16/return16
tasks, seed, metrics and comparisons. Reuse323 full-update outcomes as the
historical control only after verifying the common initial state, and explicitly
pay new live source preparation/CE/diagnostic work. The update rule is genuinely
different; do not relabel it as a successful rerun of the spent default case.
Measure acquisition/retention/return honestly, preserving negative outcomes
and the fact that known templates do not prove generalization or CPC benefit.

## Tracking receipts

Scoped #336 closed after acceptance; canonical commands/results handoff:
https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/336#issuecomment-6535364051 .
Parents remain open: #18 comment6535367962, #21 comment6535368334,
#23 comment6535368587. Both plans and development log retain archival evidence.

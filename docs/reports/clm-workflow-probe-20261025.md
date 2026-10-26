# Workflow32 reserved-probe integration — R2.1-CLM-W.P

[Issue150](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/150)
adds a guarded path from the spent32-update catalog workflow source to the
existing isolated six-update probe. **The final combined tree passes37 focused
checks on both Windows and Linux.** This is software qualification; no pretrained
fresh-task experiment or new scientific benefit is established.

## Delivered behavior and rationale

`run_workflow_reserved_probe` in the existing checkpoint adapter calls the
existing `run_reserved_probe` and `CLMCatalogProbe`. It requires a source with
exactly32updates/384training examples/generation32, original policy32/12rows/no
restores, complete matching catalog work and no uncertain work. It preserves
full source state/owners/modes/gradients/optimizer/scheduler/private RNG/work and
caller Torch RNG across every injected boundary and at exit. The probe owns separate
weights, optimizer, clock, six-update budget and permanent work. No benchmark
update, export, installation, restore, promotion or budget renewal is introduced.

The explicit `workflow_training_readout` supplies the original canonical twelve
TRAIN rows and five candidates. The workflow old-task packet must alias those
original tensors, with role `train`. Reserved markers/labels/suffix templates
remain fixed; their three action features must equal the matching subset of the
original five. No original TRAIN packet becomes probe training or replay.

Why the handoff changed: the old four temporal queries were not this workflow
source's TRAIN observations. Calling their result retention would answer a
different question. The opt-in path instead reads the twelve known rows directly,
without spending duplicate optional arrived-TRAIN reads. Default temporal behavior
retains its original0/4/8/12 checkpoints, roles, schemas, four old-task rows and
optional three-row arrived-TRAIN readout. Saved-before versus current reports are
equal in both original modes. Existing paired/rehearsal/state-only mathematics
are reused under unchanged assumptions; their complete suites were not repeated.

Every workflow probe has six delayed commits at ticks1,3,...,11:12marker query
rows,18candidate rows and36logits. Before and after, its three groups have shapes
`probe_train:2x3`, `probe_test:2x3`, `old_task:12x5`: six reads,32query rows and22
candidate rows in total. Candidate accounting uses the actual declared widths;
the shared probability validator still defaults to three and validates an
explicit integer width. No relaxed shape, mask, replay or extra fit is admitted.
Old-task TRAIN is a known-row diagnostic, not held-out generalization or task
success. Repeated sealed/staged rows are not independent examples.

New mode rejects rehearsal, marker duplication, separate arrived-TRAIN reads,
state-only and arbitrary source-point combinations. It returns unscored scalar
facts, explicit original TRAIN metadata and actual causal receipts. The old
whole-temporal cached scorer is not a workflow32 scorer. Score only after a
separately frozen case has finished; do not use readouts for early stopping,
tuning, promotion or final/probe selection. Private Python callers remain trusted.

## Sources and review

```text
src/app/clm_reserved_probe.py          narrow admission, original inbox, work/prefix guards
src/app/clm_temporal_sequence.py       explicit probability width, default three unchanged
src/adapters/clm_checkpoint_probes.py  guarded workflow entry and existing native probe factory
tests/test_clm_workflow_reserved_probe.py  literal integration/default parity/refusals/failures
artifacts/runs/clm-workflow-probe-20261025/ before files, budgets, test receipts and rollup
```

No new backend, dependency, environment variable, CLI or orchestration framework.
App/core remain backend-free and depend inward; the outer adapter owns native
Torch checks. The accepted fitter and native update mathematics are unchanged.
HEAD09ff811f plus unrelated dirty/untracked work is preserved.

Lead self-review covered actual before/after source diffs, invocation procedures,
tests, failed and repeated costs and final results. It is **self-review, not
independent acceptance**. Seven scoped final source pins held. No frozen older
protocol/result was rewritten or model cohort repeated.

## Failures, correction and prospective limits

Windows attempt1 passed33 cases and failed the legacy extra-readout parity case:
the test factory incorrectly supplied six reads where the original protocol needs
eight. The failure occurred in the saved-before app after six paid probe updates.
Only that fixture's read allowance changed; the workflow probe still has six.
Attempt2 reran the failed case and the one unexecuted width check, both passed.
Original failure, source, logs, costs and XML remain. The factory-fix format check
also failed once, then the normal formatter corrected it. No static type/lint
failure and no pretrained/scientific attempt occurred.

Self-review then found that ordinary dictionary equality accepts `False == 0` in
source work fields. The new boundary now uses the existing typed `_require_work`
check and rejects noninteger source work at admission. Two additional checks
exercise admission and a corrupted observed work view after one real probe
update. The underlying native source counters stay unchanged; nothing is reset
or refunded. Source-weight corruption also refuses before a second update.

Before final full reruns, issue150 prospectively amended only cumulative Tiny
update160→272 and candidate-row800→1100 limits. Measured prior work was142updates/
554candidate rows on Windows and124/500 on Linux; the expected full final corpus
added125/503. Original120s body/1GiB RSS/48distinct cases/1920query/9600logit/
1024heads/1024forwards/8192projected rows/48schedulerinitial/32M returned-element
limits stayed binding. No limit had been exceeded when amended; no failed cost
was refunded and no scientific budget renewed. Original and amended harnesses
and prospective issue receipts are retained. The complete final37-case corpus
then passed on each platform.

The guards refuse changed canonical labels, packet roles/catalog/subset, original
feature/metadata owners, source progress/policy/work, budgets and paid prefixes.
A paid save failure preserves its receipt and denies reuse of the spent update
authority. Public-prefix corruption retains one last trusted JSON prefix, not a
growing chain. A lost native read retains its paid work with no returned public
probabilities; source/caller state remains preserved and no updates follow.

## Commands and measured work

Recorded commands (spent outputs refuse overwrite):

```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-probe-20261025/prepare.py
.venv/Scripts/python.exe -m ruff format src/app/clm_reserved_probe.py src/app/clm_temporal_sequence.py src/adapters/clm_checkpoint_probes.py tests/test_clm_workflow_reserved_probe.py
.venv/Scripts/python.exe -m ruff check src/app/clm_reserved_probe.py src/app/clm_temporal_sequence.py src/adapters/clm_checkpoint_probes.py tests/test_clm_workflow_reserved_probe.py
.venv/Scripts/python.exe -m mypy --cache-dir .mypy_cache/clm-workflow-probe-20261025-win --explicit-package-bases --follow-imports=silent src/app/clm_reserved_probe.py src/app/clm_temporal_sequence.py src/adapters/clm_checkpoint_probes.py tests/test_clm_workflow_reserved_probe.py
.venv/Scripts/python.exe -m mypy --platform linux --cache-dir .mypy_cache/clm-workflow-probe-20261025-linux --explicit-package-bases --follow-imports=silent src/app/clm_reserved_probe.py src/app/clm_temporal_sequence.py src/adapters/clm_checkpoint_probes.py tests/test_clm_workflow_reserved_probe.py
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-probe-20261025/launch.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-probe-20261025/launch.py windows-2 'tests/test_clm_workflow_reserved_probe.py::test_should_keep_original_temporal_default_reports_and_source_behavior[True]' 'tests/test_clm_workflow_reserved_probe.py::test_should_require_the_declared_probability_catalog_width'
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-probe-20261025/launch.py linux-1
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-probe-20261025/launch.py windows-3
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-probe-20261025/launch.py linux-2
```

Final Windows3/Linux2 each37PASS/0fail/error/skip on the combined final source:
125Tiny TRAIN/backwards,914query503candidate4278logits,404heads362forwards1825
projected rows,17constructor initial steps;615captures3130125elements +2755520
returned caller RNG elements per platform. Neither fixture claims accuracy or
native usefulness. All parameters/features are literal2D, no random initializer.

All failed and repeated work remains permanently counted:

| Cumulative software observation | Windows | Linux |
| --- | ---: | ---: |
| Attempts/executions/distinct cases |3/73/37 |2/72/37 |
| Historical failed/error/skipped executions |1/0/0 |0/0/0 |
| Body seconds |29.791766800 |31.780873858 |
| Parent seconds |31.552132000 |41.479338900 |
| Maximum sampled/boundary RSS bytes, after Torch import |331874304 |345550848 |
| TRAIN/backwards |267 |249 |
| Query/candidate/logit values |1874/1057/8694 |1826/1003/8550 |
| Head objects/forwards/projected rows |852/764/3753 |798/716/3618 |
| Constructor scheduler initial steps |35 |33 |
| Returned source captures/tensor elements |1251/6367125 |1200/6107640 |
| Returned caller RNG elements |5612160 |5384640 |

Original and prospectively amended bounds held. Returned elements plus the4MiB
other-copy reserve are not all allocations or FLOPs; import RSS is not sampled.
Existing copied-optimizer/scheduler warnings remain (10 per final platform run).
Encoder/backbone/tokenization/Transformers/assets/realphysics/serving/final/
replay/pretrained work is zero. Scoped default and Linux-target mypy, lint and
format checks pass; static receipts total61.8747033s before final documentation
checks, including the one failed format check. Ordinary caches95,711,690B are
outside the8MiB evidence stage under128MiB. Actual Linux statics, full unchanged
suites, clean-clone/install/GPU/CI/release and independent result review are
skipped. Two guessed optional test filenames were missing during orientation;
actual files were located directly, with no model work or new tooling layer.

## Next action and open scientific requirements

Freeze one small native workflow32→isolatedF6 experiment using this entry, the
original marker/suffix declarations and original twelve-row TRAIN old-task
readout. Reuse existing metric arithmetic after all updates. Keep its fresh
source construction/32fits, six probe fits, readouts, raw-feature extraction,
memory/copies and elapsed time prospectively budgeted and separate. Source
TRAIN must complete successfully before probe admission; no source/model export
or reconstruction of the spent149instance. Preserve every negative endpoint.

Existing adapter/probe machinery is sufficient; do not start another checkpoint,
capture, audit framework, horizon search or nearly identical known-task rollout.
New-task fit, suffix transfer and damage to known old rows remain unmeasured
natively here. Context-return relearning is still a separate open requirement.
Prior14912/12known-TRAIN,147eight-pass10/12 and its two wrong ready targets,
older task outcomes, matchedBP/PC/neutral/active evidence and all failed/unknown
results stay at their original scopes. Full goal/R2/R5.1/R5.3/G5/R7/release remain
open. Live status is issue150/tracker18; historical plans/log remain frozen.

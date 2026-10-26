# Workflow32 probe: explicit state-only integration

[Issue152](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/152),
R2.1-CLM-W.S. The guarded workflow probe now exposes the existing state-only
parameter scope. Final59focused cases pass per Windows/Linux; scoped statics
pass after one missing test annotation was corrected. This is software
qualification, with source/procedure/results **self-review only**. No pretrained
experiment, generalization, retention improvement, CPC advantage or release claim.

## Behavior and rationale

The preceding native all-parameter probe learned1/2fixed markers and lost two
old ready decisions (legacy/parcel defer→route, old TRAIN12/12→10/12), while its
source stayed unchanged. [Every endpoint and cost remains preserved](clm-workflow-fresh-native-20261025.md).
Those negative results justify a parameter-scope control; they do not identify
which parameter change caused either result. The native/temporal state-only
learner already exists, so no new optimizer, loss, framework or dependency is
needed. The workflow entry's explicit refusal was the concrete missing link.

`run_workflow_reserved_probe(..., parameter_scope="state_only")` validates exact
supported scope before source capture, selects the existing catalog-probe factory
and forwards mode to the app. The app removes only the workflow/state-only
cross-product refusal; other admission and paid-prefix checks remain. The all
default retains its original factory-call shape and report. CLI and temporal
defaults remain unchanged. State-only already excludes action/scale from the
optimizer, keeps their values and gradients fixed, verifies optimizer ownership
and reports optimized/fixed/total parameter counts. No native update math changed.

The source must remain the exact spent32/384/32workflow catalog owner. The probe
alone owns six delayed commits, ticks1..11, twelve F query/eighteen candidate/
thirty-six logit exposures. Original F labels/features/roles and old_task TRAIN
tensor owners stay fixed. Before/after readouts retain shapes2x3/2x3/12x5, six
reads/32query/22candidate rows. The source's full weights, owners, mode, gradients,
optimizer/scheduler/RNG/work and caller Torch RNG remain protected. Original
training-only consent, independent six-update authority and bounded preceding
receipt evidence are unchanged. No replay, final-data access, source update,
promotion, serving, export or return-A learning is added.

## Files and focused verification

```text
src/adapters/clm_checkpoint_probes.py       workflow keyword, validation/factory
src/app/clm_reserved_probe.py              allow the existing state-only mode
tests/test_clm_workflow_reserved_probe.py   combined integration and failures
docs/clm-reserved-probes.md                API, counts and scientific limits
artifacts/runs/clm-workflow-probe-state-only-20261025/
  *_before.py                             original scoped source/test snapshots
  PROTOCOL.md, prepare.py                  prospective software limits/harness reuse
  focused.py, launch.py                    existing measured fixture tools
  windows-*/linux-* receipts/XML/logs       both attempts retained
  statics-*.json                          passing and failed static checks
```

The final59cases include both all/state-only six-update workflow paths. State-only
changes the state head, and after every update verifies action/scale values,
absent fixed gradients and state-only optimizer parameter owners. Source/caller
preservation, all readout shapes, native work, budget and exact receipt clocks
pass. All default/old workflow reports match the saved pre-edit app exactly;
original temporal default and arrived-TRAIN options also retain report equality.

Refusals cover invalid/noncanonical scope before capture; opposite factory scope
before any read/update; non-training-only consent; wrong source progress/policy/
work types; invalid original data/labels/roles/owners; unsupported mode products.
Both modes retain trusted preceding receipts on changed budget/features/public
prefix/metadata owner or save failure, and permanent read charges on lost native
return. State-only scope metadata/owner or fixed action/scale corruption after
one update stops before a second and retains the original trusted prefix. Source
weight corruption and equal-valued Boolean/integer work corruption remain covered.
No fake-model acquisition accuracy is asserted. Unchanged underlying scalar
gradient/native fixed-geometry checks and broader suites are reused, not rerun.

## All observed engineering work

Separate software-only limits were frozen in the issue and local protocol before
fixtures: per platform320Tinyupdates/5000query2500candidate30000logits,
1400head/identity objects+identityforwards/1200headforwards6000projectedrows,
64schedulerinitial,120sbody/160sparent/1GiBsampledRSS afterTorchimport/CPU1.
Captures+callerRNG48Melements plus4MiBother-copyreserve/literal65536; not total
allocations/FLOPs. Zero physical random/encoder/backbone/token/pretrained/
Transformers/assets/physics/final/replay work; stage8MiB/log512KiB and ordinary
caches128MiB outside stage. No engineering cap revision or scientific renewal.

| Run | Cases | Body seconds | Parent seconds | Sampled peak RSS bytes |
| --- | ---: | ---: | ---: | ---: |
| windows-1 |56PASS |15.050249200 |15.610975900 |333430784 |
| windows-2 final |59PASS |12.809157500 |13.425622100 |333299712 |
| linux-1 |56PASS |18.062246263 |23.780050100 |349745152 |
| linux-2 final |59PASS |18.575550392 |23.629654700 |349851648 |

Zero fixture failures/errors/skips. Each run consumes152TinyTRAIN/backward/
optimizer/scheduler attempts,968query584candidate4440logits,580heads508forwards/
2370projectedrows32constructor scheduler initial steps. Thus **all paid work per
platform** is304updates1936query1168candidate8880logits,1160heads1016forwards/
4740projectedrows64initialsteps;2236captures11,379,396tensor+10,031,104callerRNG
elements, below48M. Initial-step allowance is fully consumed; no more fixture
process may construct a scheduler under this allowance. Cumulative Windows
body27.859406700s/parent29.036598000s; Linux36.637796655s/47.409704800s. Each run's
21existing copied-optimizer/scheduler warnings remain in logs (42/platform).

Initial format/lint pass; Windows-target and Linux-target mypy both report the
same missing `report` annotation in the new wrong-factory test. The failed
`statics-2.json` is retained; annotation corrected, both type targets pass. During
self-review three explicit state-only consent-refusal cases were added, then the
full final59checks/platform passed within the original cumulative allowance.
No production source change followed the first56case qualification. Final
Ruff/format/type and documentation checks are recorded alongside the failures.
Measured Ruff/mypy/diff checks total13.7201921s including the failures, plus
subsecond zero-model AST/docs/rollup checks, within120s. Reused ordinary mypy/Ruff
caches occupy96,092,605B outside the stage, below128MiB. Three scoped current
source hashes and all four XML results are in `rollup.json`. A closeout
documentation patch initially failed context matching without writing files;
the exact-context retry changed documentation only, with no model work.
Final stage32files/221,564B and combined stdout logs6,892B remain below8MiB/
512KiB. Three scoped source pins remain unchanged after final qualification.
Actual Linux Ruff/mypy unavailable and uninstalled; Linux-target typing runs on
Windows. No full unchanged suite, clean clone/install, GPU, CI, release or
independent result review. No old scientific JSON parse, unsaved source/head
reconstruction, refit, or native experiment; all #151/#149/etc permanent charges
and negative/failed/unknown results remain frozen. Dirty unrelated work retained.

Recorded commands (all run from repository root; statics include measured passes
and the failed first mypy invocation):

```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-probe-state-only-20261025/prepare.py
.venv/Scripts/python.exe -m ruff format src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py tests/test_clm_workflow_reserved_probe.py artifacts/runs/clm-workflow-probe-state-only-20261025/prepare.py artifacts/runs/clm-workflow-probe-state-only-20261025/focused.py artifacts/runs/clm-workflow-probe-state-only-20261025/launch.py
.venv/Scripts/python.exe -m ruff check src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py tests/test_clm_workflow_reserved_probe.py artifacts/runs/clm-workflow-probe-state-only-20261025/prepare.py artifacts/runs/clm-workflow-probe-state-only-20261025/focused.py artifacts/runs/clm-workflow-probe-state-only-20261025/launch.py
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-probe-state-only-20261025/launch.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-probe-state-only-20261025/launch.py linux-1
.venv/Scripts/python.exe -m mypy --cache-dir .mypy_cache/clm-workflow-probe-20261025-win --explicit-package-bases --follow-imports=silent src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py tests/test_clm_workflow_reserved_probe.py
.venv/Scripts/python.exe -m mypy --platform linux --cache-dir .mypy_cache/clm-workflow-probe-20261025-linux --explicit-package-bases --follow-imports=silent src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py tests/test_clm_workflow_reserved_probe.py
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-probe-state-only-20261025/launch.py windows-2
.venv/Scripts/python.exe -B artifacts/runs/clm-workflow-probe-state-only-20261025/launch.py linux-2
git diff --check -- src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py tests/test_clm_workflow_reserved_probe.py README.md docs/clm-reserved-probes.md
```

## Scope and next action

This closes the missing conventional control's software boundary. It does not
establish that state-only adaptation acquires both F markers or prevents the
old-task losses. Full R2.1/R5.1/R5.3/G5/R7, reliable acquisition/joint retention/
context-return recovery and release requirements remain open. Historical plans/
log remain frozen under AGENTS; live evidence/handoff is in issue152/tracker18.

**Exact next action:** freeze a separate, small native state-only workflow32→F6
question through this existing entry, reusing #151's detached accepted all-mode
endpoints as the fixed comparison. Preserve its seeds/data/dose/metrics and count
the required fresh source32 plus probe6, parameter-dependent work, encoder/readout/
copy/memory/time/storage costs. Compare pre-probe source/readouts under unchanged
assumptions without reparsing spent raw inputs or reconstructing unsaved weights.
Review the actual protocol/source prerequisites before any one-attempt launch;
report all negative endpoints. Equal exposure is not equal compute. No extra F
steps/seeds, checkpoint selection, native budget renewal or CPC claim.

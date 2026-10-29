# R2.1-CTX.R — Matched rehearsal retains the old tasks in one fixed case

Issue267; parent18 and broad21/23/24/26/33 remain OPEN, full goal ACTIVE.
Single-agent source/procedure/result self-review, not independent acceptance.
Historical plans and development log remain frozen under current AGENTS and
docs/github-issue-tracking.md; the issue contains the current session handoff.

## Delivered behavior and existing evidence

`scripts/context_head_rehearsal.py` certifies original F3 and old TRAIN6 batches
as complete teacher episodes, builds four original3-step TRAIN tickets, and
composes existing `run_fresh_task_probe`, old return and `StreamRehearsal`.
Source roles/values/contexts/classes and complete transition evidence are checked;
queue episodes use the existing next-ticket convention, terminal only at end.
Terminal zero observation is codec evidence, never an extra training target.
No core algorithm/runtime change. `probe_ticket_policy_learning._observe_tasks`
adds a strict default-off matched exposure mode, restricted to original transfer
head tasks; all old defaults remain. Native endpoint counters remain exact.
CLI and source-free preview are documented in `docs/context-head-rehearsal.md`.

Accepted266 already demonstrated new-task learning, temporary old-task forgetting
and recovery; it was not rerun. Original repeated-update/capture and matched
BP/PC/neutral/active evidence is reused, not reopened.264's worker-slot declaration
and output-RSS qualification gaps,262's lost output, and all earlier spent failures
remain unchanged. No new independent acceptance or broad scientific/release claim.

## Sole prospective native pair

Protocol/artifacts: `artifacts/runs/context-head-rehearsal-20261028/`:
`PROTOCOL.md`, `source-manifest.json`, `native.py`, `native-attempt.json`,
`native-body-work.json`, `native.json`, `native-final-work.json`, `native-summary.json`.
Command from physical `D:/Dev/Repositories/Circadian`, TEMP/TMP `D:/Dev/_Temp`:

```text
.venv/Scripts/python.exe -B artifacts/runs/context-head-rehearsal-20261028/native.py
```

EXIT0; shell7.9624281s, metered whole7.7613823s, sampled peak61,603,840B,
metadata0.1086605s, output1,236,180B. All original native caps held.
Case now COMPLETE/SPENT: no repeat, raw-result reread, restore or rescore.
The wrapper already performed its one raw metadata read; subsequent reporting
uses emitted summary/scalar cost receipts. Body costs flushed before encoding;
same sampler covered encoding/write/read. Final small receipt writes follow
final sampling. Sampled RSS does not establish a hard memory bound.

Qualified source SHA `addf37403d32682b680b74ea7a1bfc16ced2895cffe09b8cc5fdf701179be943`,
277216B, prior12 passes/72 rows/16 mistakes. Warm source only; no fresh baseline
or warmup refit. Fixed tasks/order/schema/model, no randomized search or tuning.
Two public D7/K6 heads,96 coefficients/768 numeric bytes per model.

| Control | Before F: new / old | After F: new / old | After return: new / old |
| --- | --- | --- | --- |
| extra F | 0/2 / 4/4 | 2/2 / 2/4 | 2/2 / 4/4 |
| old TRAIN replay | 0/2 / 4/4 | 2/2 / 4/4 | 2/2 / 4/4 |

All18 ordinary typed slots per arm completed using all6 available actions and
original6-step maximum. New pair: v1parcel/v2password; concepts and two stage
keys/task overlap original TRAIN, although source optimizer READY pairings differ.
Extra F loses v1password and v2parcel, then recovers both on the declared return.
Replay retains both. Primary rehearsal-minus-extra-F old-success count afterF=+2.
Both final6/6. This is a useful conventional rehearsal result on known templates,
not statistical reliability, held-out generalization or a circadian advantage.

F accuracy curves: both controls2/3 per new task at0, then1 at all three paid
checkpoints. Old TRAIN queue accuracy: extra F1->5/6->5/6->5/6 per head; replay1
throughout. All points retained; no endpoint selection or early stopping.
Native F checkpoints0/12/24/48 correspond to logical replay0/4/8/16. Row exposure
0/36/72/144 is balanced per head. Actual endpoint clocks match across controls:
aggregate12/72 ->60/216 ->76/312; heads8/48,4/24 ->32/120,28/96 ->40/168,36/144.

Extra F pays48 passes/144 F rows. Replay pays16 F passes/48 rows plus32 replay
passes/96 TRAIN rows,8 complete bank events. Both old returns pay16 passes/96 rows.
Per arm64/240; full pair128/480. Replay96 exposures permanently charged;32
attempted/returned/committed native replay passes. Complete fixed bank5216
numeric bytes, cap8192B/12 transitions. F, return and replay sessions remain
distinct original30s/512MiB sessions. Source/benchmark/deployed/promotion/final
updates0. Source owners/bytes, data/bank, global RNG and paid prefix preserved.

Whole observed work:14 learner constructors/28 zero heads,288 matrix prediction
rows,108 typed calls,150 sandbox dispatches,2 teacher builds/6 certified traces;
3847 context/8146 head copy visits/760320 conservative coefficient slots. Source
1773 reads,491503968 returned bytes/929564397 attempted-byte ceiling. Logical
copy visits include validation/double counting; not allocations/FLOPs. Per-arm
session clock spans include guards and preceding-report validation; do not infer
an isolated replay compute ratio. Native updates, rows, head exposure and capacity
match; all-in compute does not. Input diagnostics per arm3 bound calls/120 rows/
1080 coordinates/9600 attempted+returned numeric bytes, with no extra fitting.

## Software validation and preserved failures

16 distinct behavior cases/platform. Runtime/data cases test complete certification,
foreign roles/values/context/class/order refusal, full128/480 private pair and
per-head checkpoint matching, source/RNG preservation, counter/limit/task tamper,
lost native work, refused second arm, restored preceding paid evidence, failed
replay's permanent3-row charge and original metadata-free defaults. Three CLI
cases test source/model-free preview and existing/source output refusal.

Exact executed command prefixes, all stage paths under the directory above:

- Windows `python -B .../check.py win-first tests/test_context_head_rehearsal.py`:
  EXIT1 after8passes, first native-failure assertion fails because top completion
  error masked underlying reason. Production reason handling fixed; original
  partial probe/budget/cost evidence retained.129 native passes/483 rows.
- `.../check.py win-final tests/test_context_head_rehearsal.py`: first11 cases
  pass, then replay fixture fails before fitting: actor/shadow RSS guard requires
  an attached sampler.193 passes/723 rows. No production RSS guard bypass.
- `.../check.py win-replay <replay-failure-test> <source-refusal-test>`: EXIT1,
  diagnostic confirms missing fixture sampler,0 passes/0 rows. Attach sampler.
- `.../check.py win-replay-final <same2tests>`: pytest2PASS, wrapperEXIT1:
  cumulative packet constructions1511 exceed inherited1500 engineering cap by11.
  Embedded exit_code0 records pytest only, NOT wrapper success. Old window stays
  FAILED/SPENT; no model-fixture repeat or retroactive cap increase. Aggregate
 325 passes/1215 rows/17.0316782s, peak85,565,440B; original384/1440 and all
  native/typed/time/RSS/source limits held.
- WSL Ubuntu installed Python `-B .../check.py linux-final tests/test_context_head_rehearsal.py`:
  EXIT0,13PASS,196 passes/732 rows,14.392909949s/RSS75,223,040B,760 packets.
- Prospectively separate CLI-only allowance0 passes/0 rows/128 packets/15s:
  `.../cli-check.py win-cli <preview-test> <existing-output-test>` EXIT0,3PASS,
 1.2602015s/RSS72,560,640B; Linux same3 EXIT0,7.217782639s/RSS67,792,896B.
  Each55 packets,2 teacher builds/6 traces/42 dispatches,0 models/fits/rows/
  predictions/scientific reads. This new usability work did not repair/replay
  the spent original packet window. Scientific protocol unchanged.

Windows known body behavior therefore passes all16 distinct cases with disclosed
original engineering-window failure. Linux passes all16 including full original
cohort. Final Ruff check/format, mypy default and `--platform=linux`, all using
`--explicit-package-bases --follow-imports=silent`, PASS3 touched files. Initial
unused imports/fixture-name lint and Optional/list/variable type errors retained
in session output then fixed. Existing correctly passing model tests reused after
CLI-only additions; no repeated baseline or whole-suite work.

Skipped: full suite, clean clone/wheel/install, Linux scientific run, neural/
robot/game sweeps, live worker/fault table (not in this probe protocol), final
evaluation, independent review (current single-agent policy), remote CI/release.
No new dependency/install/network model/GPU/commit/push/delete/reset. Inherited
dirty/untracked/ignored work preserved; modified old file includes inherited266
work, not wholly attributable to267. Source manifest binds combined final tree.

## Critical path and one next action

Original R2.1 diagnostics/probes and matched BP/PC/neutral/active implementations
already exist; do not reimplement them or add another dose/capture sequel. This
increment answers the demonstrated local forgetting question using conventional
rehearsal. Broad R2/R4/R5/R7/generalization/reliability/release remain open.

Next useful delivery gap: this paired-head probe CLI still requires historical
qualified JSON. Existing `recurring_ticket_workflow_example.run_example` already
trains the original paired public-context source from0 through AttributionStream
and owns its native checkpoint12 in memory; it has no private adaptation handoff.
The affine first-party private workflow exists separately and should not be
duplicated. Add a default-off in-memory paired-head private adaptation mode there,
passing the actually qualified terminal source to these existing probe/rehearsal
ports only after its ordinary TRAIN task readouts pass. Preserve normal behavior,
private forks and original source/return/replay costs. First add focused ownership,
failed-source refusal and shared-budget cost tests; freeze a small product
qualification only after gates. No automatic new comparative experiment, model
family/seed/dose change or archive loader is implied.

# Native fixed controls with an arrived TRAIN bank

Issue [#200 / R7.2-MG.R.C](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/200).
One frozen three-control invocation completed; source, procedure and scalar
results passed **self-review**, not independent acceptance. No production learner,
controller, stream or evaluator changed in this observation.

## Observed behavior

Each cell is **old A3 / new A4 goals solved**, out of two per context.
Fixed logical observations were at ticks 2, 4 and 6 after the same six delayed
A3-blue/red → A4-blue/red → A3-blue/red TRAIN packets and seven opportunities.

| Fixed control | Tick 2 | Tick 4 | Tick 6 | Actual fits | Current / prior / total fit rows | Pending packets |
| --- | --- | --- | --- | ---: | --- | ---: |
| No learning | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 / 0 / 0 | 6 |
| Periodic, period 2 | 1 / 0 | 2 / 0 | 2 / 1 | 3 | 27 / 9 / 36 | 3 |
| Budget aware, reserve 4 | 0 / 0 | 0 / 0 | 2 / 0 | 2 | 18 / 0 / 18 | 4 |

All three observation windows completed; **none completed all queued training**.
Periodic reaches only the blue new-layout goal at return, not full new-task
acquisition during the A4 phase. Budget aware acquires the old task late; this
is not recovery. No control first loses an acquired old cue and then regains it,
so conditional recovery is null/untested for all three. New-task gain from tick
2 to 4 is zero throughout. Old retention differences are 0, +1, 0; returned-old
differences are 0, 0, +2. Incomplete initial acquisition remains explicit.

Successes terminated after 9 steps with return .49375; failures truncated at
16 steps with zero return. All native K7 predictions used raw first argmax,
seed 17, max 16, original public-history features and TRAIN roles.

| Control | Diagnostic calls / rows | Unique games / successes | Game steps | Unique return |
| --- | --- | --- | ---: | ---: |
| No learning | 6 / 54 | 4 / 0 | 64 | 0 |
| Periodic | 6 / 72 | 12 / 6 | 150 | 2.9625 |
| Budget aware | 6 / 54 | 8 / 2 | 114 | .9875 |

There are 36 logical game slots but 24 unique executed games. No-learning's
ticks 4/6 reuse its complete tick-2 state/readout; budget-aware tick 4 reuses
tick 2. Periodic runs three distinct readouts. Reuse required identical complete
state and preceding successful evidence, including container identity, and
remains labeled by the original tick. It adds no independent observation.
Episode identity across controls is `(control, episode_id)`; there is no
cross-control reuse or pooled unequal-readout winner.

## What this answers and leaves open

The controls received the same freshly produced TRAIN data, arrival clocks,
opportunities and enabled ready-diagnostic access. Their selected FIFO prefixes,
actual row exposure and diagnostic work differ. Correct-oldest packets remain
queued; nothing skips, acknowledges, drains or forces later updates.

The accepted [ready-drift observation](affine-ready-replay-native-20261026.md)
previously reached 2/0 → 2/2 → 2/2 with four fits/54 rows/18 prior rows and two
pending packets. The [fixed-six joint observation](a4-adaptation-20261025.md)
retained both with six fits/90 rows/36 prior rows. These are historical descriptive
references: identical source bytes and readout work across studies were not
established. They were neither refitted nor reconstructed. This study establishes
no paired efficiency, generalization, reliability, CPC or release advantage.
Earlier failed, spent, negative and inconclusive studies remain unchanged.

All windows finished before any game. No final, heldout, probe, consolidation,
promotion or tuning data entered training or controller decisions. Original
source, actors, full checkpoint states, caller Python/NumPy RNG after optional
imports, budget bindings/limits, prior charges/quiescent clocks and paid evidence
were checked before more work. Separate cold TRAIN owners were not renewals.

## Costs and validation

The only native command was:

```powershell
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/affine-bank-controls-native-20261026/execute.py
```

Supervisor terminal exit 0; body `complete`, error null. Whole body
106.2320114 s, parent 106.7582243 s; original body 150 s / hard 165 s / parent
180 s limits held. Sampled process RSS 89,526,272 B ≤ 512 MiB; this is not a
continuous or process-tree peak. Fixed arm order and shared preparation/evidence
checks prevent per-arm wall-time or net-utility attribution.

Aggregate native work: 5 LSE fits/54 rows; 5 LAPACK solves/54 rows; 9 prior-row
exposures; 33 empty affine constructors; 5,610 snapshots; 346 predictions/508
rows, including 18 diagnostic calls/180 rows; 26 environments/closes; 346 physics
steps = 328 game + 18 fresh TRAIN teacher steps. All prospective ceilings held.
Snapshot calls and the 80-million coefficient-slot reserve do not measure all
allocations, copies or FLOPs. One 94,705 B packaged TRAIN read supplied A3;
packaged collection is sunk work, not remeasured here. Each arm separately
prepared 213,840 input-array bytes, an 18-row planned bank and 90 potential fit
exposures; actual committed rows are in the first table.

Before launch, unchanged #199 evidence (71 case IDs/platform, scoped statics)
and #198's exact `read_checkpoint` / `summarize_case` helper bodies (8 pure cases/
platform) were reused. Relevant hashes matched. New case/supervisor commands,
each exit 0 on its first cohort, took 9.4384333 s cumulatively. A conservative
cohort-or-later timestamp bracket charges 110,899,200 B of touched mypy cache
(the entire changed database) plus 252 B of Ruff cache, below 160 MiB; the full
inherited caches remain separate (597,720,517 B / 390,519 B respectively).

Commands:

```powershell
.venv/Scripts/python.exe -m ruff format artifacts/runs/affine-bank-controls-native-20261026/case.py artifacts/runs/affine-bank-controls-native-20261026/execute.py
.venv/Scripts/python.exe -m ruff check artifacts/runs/affine-bank-controls-native-20261026/case.py artifacts/runs/affine-bank-controls-native-20261026/execute.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent artifacts/runs/affine-bank-controls-native-20261026/case.py artifacts/runs/affine-bank-controls-native-20261026/execute.py
.venv/Scripts/python.exe -m mypy --platform linux --explicit-package-bases --follow-imports=silent artifacts/runs/affine-bank-controls-native-20261026/case.py artifacts/runs/affine-bank-controls-native-20261026/execute.py
```

Post-run scalar review (`.venv/Scripts/python.exe -B -`, stdin assertions) passed
in .012287 s: original 12 source/protocol hashes, terminal success/caps, original
budget chronology/limits, receipt and current/prior/total-row conservation,
attempt/return equality, source/actor preservation, complete closed native
episodes, fixed seed/roles, exact-state reuse labels, outcomes versus emitted
episodes, prediction/diagnostic/physics/environment totals. `result-review.json`
records derived metrics, not new scoring. `terminal-summary.json` detaches the
same-process scalar line from `native.txt`. The 2,284,898 B scientific
`actual.json` was **never reopened**; only its size was inspected. Stage was
2,455,314 B before this closeout, within 12 MiB; log 47,382 B < 256 KiB.

No native/test/static failures occurred in this scope. A later terminal-handle
poll was unavailable after context continuation; the existing authoritative
terminal parent receipt and emitted scalar result resolved it, without another
launch. Oversized orientation output was truncated and replaced with bounded
reads. Two closeout patch anchors failed with zero edits; an exact smaller
patch corrected the documentation. A local-link check then caught an extra
parent traversal in the stage RESULTS link; it was corrected before the focused
rerun. These closeout failures ran no model work. Skipped: unchanged numerical/platform suites, Linux native rerun,
whole-repository/clean-clone/wheel/release checks, optional GPU/Torch/pretrained
models, remote CI, browser, sweeps and all new comparative tuning. No installs,
dependencies, Git mutations, agents or concurrent model workloads.

## Next useful implementation

R2.1 diagnostics/disposable fresh probes and repeated-update runtime are already
accepted; no stale reconstruction is needed. Full R2/R4/R5/R7/P7/P9 and scientific
release requirements remain open. Actual checkout inspection identifies a product
gap: `run_fresh_adaptation()` always chooses the fixed-six joint recipe, while
`run_context_linear_stream` explicitly rejects A4 plus reference policy. Thus
the observed controller-selected A4 behavior still needs a bespoke research
driver despite the existing first-party source/stream/game components.

Next action: open a bounded child to add explicit reference-policy A4 composition
to those existing paths, preserving fixed-six defaults, original clocks/limits,
source admission, partial-prefix readouts and pending-work accounting. Begin
with literal partial-window/CLI isolation tests before any separately frozen
native invocation. This makes existing learning useful from the ordinary fresh
workflow; it adds no controller algorithm, tuning knobs, checkpoint/audit
framework or automatic fourth/fifth-update sequel. Historical plans/log remain
frozen under AGENTS; issue #200 and tracker #18 carry the live handoff.

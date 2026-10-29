# Native reference learning through the fresh A4 command

[#202 / R7.2-MG.F.X](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/202)
completed its **first and only** public-command engineering trace. Source,
procedure and scalar results passed self-review, not independent acceptance.
No production code or algorithm changed in this observation.

## Useful observed behavior

The actual existing command entry point ran with:

```powershell
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --context-linear-reference drift --ready-feedback --output actual.json
```

It collected fresh blue/red A4 TRAIN episodes, learned from the original delayed
A3/A4/A3 stream, handed off actual partial checkpoints and played target-free
known TRAIN games. Original defaults, seed17/max16/K7 firstargmax stayed frozen.

| Logical tick | Actual updates | Old A3 solved /2 | New A4 solved /2 | Readout |
| ---: | ---: | ---: | ---: | --- |
| 2 | 2 | 2 | 0 | Four new games |
| 4 | 4 | 2 | 2 | Four new games |
| 6 | 4 | 2 | 2 | Reuses unchanged complete tick4 state/readout |

The known new layout was acquired while the old layout was retained. No old cue
was lost, so recovery remains null/untested. Tick6 is no independent return-game
sample. The seven-opportunity window completed; training did not drain, with
two return packets still pending. Correct-oldest FIFO waiting was preserved,
without acknowledgement, skip, forced updates or renewed credit.

Four LSE fits consumed54 rows =36 current +18 prior TRAIN reuse, versus the
prepared plan's90 potential rows. Six ready diagnostics consumed90 rows. Eight
unique games (twelve logical slots) produced six successes and two failures:
successes terminated at9 steps/return.49375, failures truncated at16/return0.
Unique total return2.9625,86 game steps plus18 fresh teacher steps. This is known
TRAIN acquisition/retention and a working ordinary entry point, not reliable
generalization, controller/CPC advantage, matched computation or release readiness.
Earlier198/200 outcomes remain historical descriptive references, not refitted
controls or evidence of paired source/compute equivalence.

## Execution, costs and isolation

The actual metered native launcher command was:

```powershell
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/a4-reference-cli-native-20261026/execute.py
```

Its child invoked the same `cli.main()` with the flags above, exclusive
`artifacts/runs/a4-reference-cli-native-20261026/actual.json` and normal exit
semantics. Duplicate full stdout was discarded; a wrapper retained the returned
report in memory and emitted only its bounded scalar view after CLI completion.
The scientific output **was never reopened or rescored** (size/stat only).

Parent terminal exit0, CLI exit0, body/workflow `complete`, no failure/capture
error. Body9.6356218s, parent10.3851928s, sampled process RSS90,382,336B. Original
150s body/165s hard/180s parent/512MiB caps held. Body includes optional imports,
preparation, learning, games, encoding/output and instrumentation; RSS is sampled,
not continuous/process-tree peak. No per-component efficiency attribution.

| Native work | Observed | Frozen ceiling |
| --- | ---: | ---: |
| LSE fits / rows | 4 /54 | 6 /90 |
| LAPACK solves / rows | 4 /54 | 6 /90 |
| Prior TRAIN rows within fits | 18 | 36 |
| Empty constructors / snapshots | 13 /1535 | 128 /16000 |
| Predictions / rows, diagnostics included | 92 /176 | 198 /300 |
| Ready diagnostics / rows | 6 /90 | 6 /108 |
| Environments / closes | 10 /10 | 14 /14 |
| Physics steps, teacher included | 104 | 224 |
| Packaged TRAIN source reads / bytes | 1 /94705 | 1 /94705 |

The packaged producer decodes54 raw rows and selects18 A3 rows; its historical
collection cost is sunk, not remeasured. Fresh A4 retains TRAIN provenance from
birth: two complete closed9-step episodes,18 history rows, no teacher retry.
Planned213840 input-array bytes/18 bank rows/90 potential exposures remain
separate from committed54 rows. Snapshot calls/conservative80M coefficient-slot
reserve do not count all allocations, copies or FLOPs.

All training finished before games. Original source/actor/full-state/paid-prefix
identity/value checks stayed active. Shared original0-update/0-replay150s owner
started before preparation; original6-update90s TRAIN owner remained separate.
Bindings/typed limits were pinned; TRAIN4 charges/zero consolidation replay and
quiescent clock were sealed after wrapper return and preserved through readouts.
Caller Python/NumPy RNG after optional imports stayed unchanged. No final,
heldout, probe, tuning, consolidation or promotion access occurred.

## Validation and limitations

Current #201's74 case IDs/platform and scoped statics were reused after all six
source pins matched. Unchanged #199 evaluator geometry/native loop and native
learner/composer/controller/producer contracts were reused. No new numerical
fixtures or control cohort. FrozenPROTOCOL/qualification14source pins and
source/procedure self-review GO preceded the sole launch. An early GO comment
said17 pins; the authoritative metadata and observed count are14.

New case/supervisor final Ruff format--check/check and default/Linux-target
mypy all passed. Initial two local-list annotations failed both mypy targets;
explicit annotations fixed them without ignores. All static attempts cost
18.5624384s <90s; exact commands/outcomes in `statics-*.json`.
Conservative cohort-or-later touched caches110924028B <160MiB; inherited whole
mypy/Ruff caches597745093B/391055B remain separate. Final stage3,522,137B <12MiB,
metadata67,981B <1MiB and scalar log23,479B <256KiB (before receipt closeout).

Post-run `.venv/Scripts/python.exe -B -` scalar-only assertions passed in.0196651s:
14 source hashes, original budgets/clocks/charges, receipt/current/prior/native
row conservation, complete native episodes, source/actor flags, fixed roles and
settings, exact reuse labels, goals/rates/recovery, diagnostic/prediction/physics/
environment totals and all caps. Same-process `native.txt` was detached into
`terminal-summary.json`; `result-review.json` contains the checks/derived metrics.
`actual.json` is3,430,677B <8MiB and remains local/unopened. No native/test failure,
timeout, retry or allowance amendment occurred. One orientation search named a
nonexistent inspector file; its separate guide/source reads continued. No extra
scientific reader was created to compensate.

Skipped unchanged suites, new Linux/native repeats, whole-repository/cleanclone/
installed-wheel/release checks, GPU/Torch/pretrained/browser/CI/sweeps. No install,
dependencies, agents, Git mutations or unrelated change removal. Old failed,
spent, negative and inconclusive evidence stays unchanged. Historical plans/log
remain frozen; issue202/tracker18 carry live status, not a release claim.

## Critical path and exact next action

The public conventional learner now demonstrates known-task acquisition and
retention; repeated-update runtime and R2.1 isolated diagnostics/probes already
exist. Repeating those or another readout/capture increment would add little.
The existing stress-validation study #103 already reported negative deployed
utility (no-learning best), with shared semantic templates; it is not an unseen
semantic generalization result and must not be retuned or rerun as a successor.

A concrete remaining dependency blocks applying the **existing** learned chooser
to this functioning learner: `AttributionStream.run_reference_window` admits only
exact `ReferencePolicy`, while `core/contextual_controller.py` and
`app/counterfactual_controller.py` already implement finite fitted wait/wake
selection and full-source/version-bound disposable trials. Their ticket fits
cannot be relabeled as MiniGrid fits or have their state binding ignored.

Next implementation: add explicit fitted-controller admission to the existing
finite TRAIN-window path, using the existing source/version/digest ports and
original action/budget/feedback boundaries. Start with literal tests that reject
stale/different complete source, changed policy, changed prior fit evidence and
final/probe inputs, and preserve original fixed-reference behavior. Reuse the
existing chooser/owner; no new algorithm or framework. Any later counterfactual
training/application/withheld-stream experiment needs its own frozen costs and
chronology, including discarded trials; current results select no new tuning.
Full R7 external utility/generalization and broader repository/release parents
remain open, with negative results valid.

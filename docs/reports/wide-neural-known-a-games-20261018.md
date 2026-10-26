# H32 original-TRAIN checkpoints in known A games — 2026-10-18

## Question and protocol

The existing saved-source fit CLI now executes actual known-A games from a genuine
H32 checkpoint. This is one fixed exploratory four-method case, not a sweep or an
independent replication. Original six TRAIN sources/54rows,128passes6912rows each,
D296/H32/K7/seed17/LR.03/momentum.9/PC S2/eta.04 and native objectives unchanged.
Training is charged again because the accepted prior controls did not persist
native weights. Every0/128 fixed TRAIN readout, outcome, gap, receipt and protocol
(excluding the explicit live extension) exactly reproduces its earlier control.

Each checkpoint then plays both fixed A blue/red TRAIN games in that order with
owned frozen readers, raw seven-action first argmax and at most16 steps per cue.
No masks, action repair, teacher/target access or live learning. Source/actor/
candidate/optimizer/adaptive/RNG/input/budget guards and original parent clock/RSS/
permanent charges remain active. Separate20s readonly child budget has0training/
0replay and its binding survives publication. No E/F/probe/final data, newteacher,
selection, promotion or baseline tuning enters this study.

## Results

The last three values in each game cell are success, return and actual steps.

| Method | Correct TRAIN rows | Exact TRAIN traces | TRAIN CE | A blue | A red |
|---|---:|---:|---:|---|---|
| bp | 51/54 | 3/6 | 0.1862326696 | 0, -1.00000, 7 | 1, 0.49375, 9 |
| pc | 30/54 | 0/6 | 1.1213744914 | 0, 0.00000, 16 | 0, 0.00000, 16 |
| neutral_cpc | 30/54 | 0/6 | 1.1213744914 | 0, 0.00000, 16 | 0, 0.00000, 16 |
| gated_cpc | 30/54 | 0/6 | 1.1922821535 | 0, 0.00000, 16 | 0, 0.00000, 16 |

All four runtimes complete and all full-fit witnesses remain false. BP chooses
the same prefix for both cues: forward,forward,right,pickup,left,forward,toggle.
Blue terminates with wrong-key penalty−1 at7steps; red continues forward,forward
and reaches the goal at9steps for+.49375. BP mean success=.5 and mean return=
−.253125. All six PC-family games choose forward16times, truncate, success0 and
return0. Every chosen index is a valid native action; validity is not progress.

Active-minus-PC and active-minus-neutral game success/return/steps are all0;
active TRAIN CE is+.0709076621 with equal accuracy/exact. PC-minus-BP success
is−.5 but return is+.253125 and mean steps+8. Thus neither metric supplies an
unconditional winner. Neutral-minus-PC is0; their raw TRAIN and live predictions
are exactly equal. No CPC benefit, statistical generality or equal-compute claim.
The one known-TRAIN cue success is not heldout acquisition, retention or recovery.

## Costs and verification

Exactly512native updates27648trainingrows, BP128derivedSGD/backward, PC384derived
parameter updates41472nominal latent-row steps. Fixed P48calls432rows plus112live
calls/rows gives160predictioncalls544rows;8actualresets112actualsteps. Live
projection and history each return112*296*8=265216array-equivalentB; these are
distinct charged transformations/storage, not disjoint peak-memory measurements.
One148value initial history per active episode=1184numericB. All live training/
replay0. Copies, validation, native adaptive/optimizer/RNG state and Python/backend
costs add to9735parameter elements38940array-equivalentB; not total memory.

| Method | Actual process seconds | Enclosing fit body seconds | Included live seconds | External peak RSS B | Internal peak RSS B |
|---|---:|---:|---:|---:|---:|
| bp | 9.1463834 | 6.4683023 | 1.7052940 | 368922624 | 366927872 |
| pc | 8.0718170 | 5.5382807 | 2.2326265 | 287338496 | 284545024 |
| neutral_cpc | 11.3073359 | 8.8135727 | 4.3425217 | 287301632 | 284160000 |
| gated_cpc | 11.2432486 | 8.7476539 | 4.3008696 | 287444992 | 284831744 |

Times are one serial run per method with thread1/GPU0; not speed comparisons.
Science parent-command41.4201329000/480seconds CLOSED; engineering root900+
review300+sixchecks4.4017423000=1204.4017423000/1400 CLOSED.
All prior grants CLOSED. CPU1/GPU0/threadenv1/interop1/RSS768MiB/stage32MiB.

All11commands exit0/no resource problem or cancellation. New two-file Ruffformat/
lint and four ordinary CLI previews pass. Unchanged production/tests reuse308
qualified scoped cases and their accepted Ruff/Win/Linux mypy/independent review;
no claim of a new all-green full-suite run. Root saved arithmetic validates all
432fixed+112live rows,512receipts, histories/actions/transitions/returns and costs.
Independent SOURCE/PROTOCOL GO precedes native work; ACTUAL review separately
checks the actual saved evidence. No experiment failures or retries this stage.
One guessed source filename and one Windows shell wildcard lookup failed during
orientation; explicit discovered paths resolved them without additional tooling.
Reviewer had one similarly missing optional guessed source path; no model effect.

## Critical path and limits

Completed R4.2 original-H32 known-A-game scientific child, not full R4.2. Existing
bounded R2.1/three-update acceptance and all prior negatives remain. Full useful
adaptation, acquired retention/recovery, heldout layouts, recurrent RL, R2.7,
capture/prune, other roadmap integrations and release/full goal remain OPEN.
Skipped unchanged full suites/statics, cleanclone/release/archive audits, new
probes/final/replay/teachers, sweeps and all old baseline reruns.

Why next: unlike the preceding H8 joint control, this BP checkpoint can complete
one A cue. The existing isolated F16→A16 diagnostic already separates new-task
fit, old-A monitoring and return learning with costs, but its history factory and
shared counter currently hardcodeH8. Extend that existing path explicitly; do not
select only red, change seeds/doses, tune PC or build another runner. Preserve
blue/unacquired outcomes and all four native controls. No probe experiment is
authorized by this completed study's spent budget.

**Exact next implementation:** Extend the existing joint-catalogue minigrid_learnability API/CLI with an explicit fixed H32 four-method recipe for the existing isolated F16 then A16 probe at genuine0/128 checkpoints. Update its hardcoded H8 factory/policy/counter admission without broadening defaults, preserve both A cue monitors and original source/actor/RNG/budget isolation, and account for all probe and return costs. Run focused final-tree tests and independent source review before freezing any new comparative probe experiment.

[Frozen protocol](../../artifacts/runs/wide-neural-live-study-20261018/PROTOCOL.md),
[commands/artifacts](../../artifacts/runs/wide-neural-live-study-20261018/RESULTS.md),
[independent review](../../artifacts/runs/wide-neural-live-study-20261018/REVIEW.md),
[API/CLI and source structure](../wide-neural-train-rollout.md).

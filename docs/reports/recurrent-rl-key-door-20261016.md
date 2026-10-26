# Separate recurrent key/door RL case — 2026-10-16

One prospectively frozen exploratory local seed17 case. This new CLI route uses
the existing ActorShadowRuntime/ExperienceInbox/LearnerStep, native reward update
and key/door collector/evaluator. No accepted BP/PC/neutral CPC/active CPC or BC
experiment was rerun. Full R4.2 and the repository goal remain open.

## Frozen training and final evaluation

One recurrent_rl arm, no replay. D8/H8/K5, SGD rate.03/momentum.9, gamma1,
max native16steps, sampled episode limit12, private PCG64seed29. Eight actual
A+/A-/B+/B-/C+/C-/A+/A- training episodes. Each restores the complete current
candidate checkpoint on a disposable collection fork; only the collected train
record and a matching opaque ID release enter the existing inbox. Native updates
retain the original on-policy fullstate/probability checks. No BC warmup, teacher
training target, shaping, masking, action repair, retry, baseline or entropy term.

All five native checkpoints0/2/4/6/8 are retained. Only after all eight commits do
40 cached classifications and40 fresh greedy live episodes evaluate A/B/C and
final-only D with both cues. Final teacher packets never reach updates, tuning,
promotion or probes. Initial parameters and eight final packets match accepted
BC evidence exactly; objective/variable training exposures differ, so this is not
an equal-work or causal comparative experiment.

## Every sampled training outcome

| Index | Context | Cue | Actions | Decisions | Applied legal steps | Return | Ending |
|---:|---|---:|---|---:|---:|---:|---|
| 0 | A | +1 | [0, 2, 2] | 3 | 2 | 0 | illegal_action |
| 1 | A | -1 | [1] | 1 | 0 | 0 | illegal_action |
| 2 | B | +1 | [0] | 1 | 0 | 0 | illegal_action |
| 3 | B | -1 | [0] | 1 | 0 | 0 | illegal_action |
| 4 | C | +1 | [1] | 1 | 0 | 0 | illegal_action |
| 5 | C | -1 | [1] | 1 | 0 | 0 | illegal_action |
| 6 | A | +1 | [0, 1] | 2 | 1 | 0 | illegal_action |
| 7 | A | -1 | [3] | 1 | 0 | 0 | illegal_action |

**Negative:** all eight episodes produce zero reward. Initial optimizer momentum
is zero; the explicit reward objective supplies zero gradients, and parameters
stay identical across all five checkpoints. The eight native SGD commits still
consume eight permanent update charges and11training rows. This run did not learn.
It is not evidence that recurrent reward learning cannot learn in other protocols.

## Every final result

All40final live episodes end at an illegal first action:0success/0return/0applied
steps. Every A/B/C/D± episode at all five checkpoints is preserved in pilot.json.
All context CE/accuracy curves remain flat; B gain, C gain, A retention loss and
A return gain are exactly0. Flat retention of an untrained failed policy is not
useful retention or recovery. All old400failed baseline episodes remain intact;
this case adds40honest failures, no acquisition/generalization/circadian benefit.

## Exact observed work and resources

- 8native SGD calls/11training rows,8sampled episodes/11prediction decisions.
- 40cached classification calls/320rows and40live forwards/episodes.
- 91native predictions total;382unrolled recurrent timesteps =11collection +
  11training +320classification +40live.48fresh sessions,56environment constructions.
- 67physical steps =64final teacher +3sampled train +0live. Illegal actions are
  unapplied and do not fake terminal reward or physical progress.
- Recipe8.3504064s/worker8.8596765s/parent9.4085024s; sampled process RSS313483264B,
  818samples at requested5ms. JSON234626B/stderr0,one sampler joined,violations0.
- All fixed8SGD/96trainrows/616predict/992unroll/640steps/90body120parent/768MiB/
  8MiB limits pass. CPU/BLAS/Torch/interop1/GPU0. No budget renewal or extra update.
- Native tensor payload:181parameters/724B, gradients/momentum724B after updates,
  activation32B. Observation records exclude owned checkpoint/session/record/copy/
  graph/Python payloads; sampled RSS covers the full process separately. Storage
  bounded by8planned train records/8inbox entries/5checkpoints and one overwritten
  fixed8record comparison copy. No unlimited success-record chain.

Saved-data analysis verifies all eight receipts, exact native/example/budget
progression, eleven PCG64 draws and sampled actions from recorded probabilities,
final sampler state, source/serving actor isolation and all22revised frozen texts.
No learner or physics rerun was used for this aggregation. First descriptor/freeze
remains preserved: independent review corrected only inherited mean-CE architecture
metadata before native GO. Data, parameters, metrics and budgets were unchanged.

## Critical path reassessment

The separate RL route now works; do not start another receipt/capture/audit or
nearly identical episode-count increment. The direct **training-only** evidence is
lack of any nonzero reward, so this update rule had no learning signal in the case.
Original R4.2 calls for demonstration-first development followed by recurrent RL.
The next learner-facing feature is an explicit demonstration warm-start phase
before sampled reward updates, through the same typed runtime boundary.

**Exact next implementation:** add an explicit train-role demonstration packet
and a fixed demonstration→reward phase schedule to the existing recipe, preserving
this accepted pure-RL mode. Keep objective IDs, native optimizer/gradient state,
per-phase exposures/charges and on-policy admission explicit. Verify phase routing
and continuation before a separate small train-only chosen protocol. A future
comparison needs a same-architecture, same-demonstration warm-start BC-only control
and must account for carried momentum (zero-reward steps can still move a
warm-started policy). Do not retune
old baselines, increase episodes until success, select seeds, alter metrics or use
any previously final-scored weights/data for warm start. No new experiment runs
inside this closed allowance. Useful learning/recall/recovery/generalization,
fullR2.7/R2.8/scientific/release criteria remain unfinished.

[Raw report](../../artifacts/runs/r42-recurrent-rl-pilot-20261016/pilot.json),
[accounting and RNG arithmetic](../../artifacts/runs/r42-recurrent-rl-pilot-20261016/analysis.json),
[commands/failures](../../artifacts/runs/r42-recurrent-rl-pilot-20261016/RESULTS.md),
[independent review](../../artifacts/runs/r42-recurrent-rl-pilot-20261016/REVIEW.md).

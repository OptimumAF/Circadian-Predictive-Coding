# H32 original-data fresh-task fit, old-A monitoring and return — 2026-10-18

## Protocol and scope

One frozen exploratory case each BP/PC/neutral CPC/active CPC, in that order:
original54TRAINrows/128jointpasses and independent source0/128 disposable F16
then A16. D296/H32/K7/seed17/LR.03/momentum.9/PC S2eta.04, native objectives and
active/neutral settings unchanged. All5TRAINreadouts and bothcue0/4/8/16phase
curves retained. No best endpoint, seed/dose/metric/cue selection, baseline
tuning, retries, replay, final D/E, promotion or learner games.

Each publiccase collects original6TRAIN+4Fteacher episodes; sources/geometry/
roles/causal support match the accepted H8joint recipe exactly. Every0/128raw
TRAIN score and128benchmark receipts exactly reproduce acceptedH32controls.
Retraining creates genuine in-process native checkpoints because earlierreports
didnotpersistweights; it is charged again and is not independent replication.
Probe_test never trains. Benchmarks/deployedactor/originalsource remain unchanged;
F/A updates stay on two separate disposable cohorts, each with independent
original16-update budgets/0replay, inside original180s parent/RSS allowance.
Fullnativeoptimizer/adaptive/gradient/RNG/input/precedingreport/state guards pass.

## All fixed endpoints

Cells show CE / rowaccuracy / exacttracefraction. A sequence is before F → after
F16 → after A16; Freturn is F after A16. All quantities are teacher-forced fit.

| Method | Source updates | F0 → F16 | A before → after F → after return | Freturn16 |
|---|---:|---|---|---|
| bp | 0 | 2.003617 / 0.055556 / 0.0 → 1.044148 / 0.555556 / 0.0 | 2.008331 / 0.111111 / 0.0 → 1.250240 / 0.555556 / 0.0 → 0.697719 / 0.777778 / 0.0 | 0.750371 / 0.777778 / 0.0 |
| bp | 128 | 1.910845 / 0.666667 / 0.0 → 0.404399 / 0.944444 / 0.5 | 0.175711 / 0.944444 / 0.5 → 0.700203 / 0.833333 / 0.0 → 0.324890 / 0.888889 / 0.0 | 0.574178 / 0.888889 / 0.0 |
| pc | 0 | 2.003617 / 0.055556 / 0.0 → 1.390003 / 0.555556 / 0.0 | 2.008331 / 0.111111 / 0.0 → 1.420462 / 0.555556 / 0.0 → 1.343442 / 0.555556 / 0.0 | 1.319533 / 0.555556 / 0.0 |
| pc | 128 | 1.175813 / 0.555556 / 0.0 → 1.124428 / 0.555556 / 0.0 | 1.120504 / 0.555556 / 0.0 → 1.099171 / 0.555556 / 0.0 → 1.059562 / 0.555556 / 0.0 | 1.104839 / 0.555556 / 0.0 |
| neutral_cpc | 0 | 2.003617 / 0.055556 / 0.0 → 1.390003 / 0.555556 / 0.0 | 2.008331 / 0.111111 / 0.0 → 1.420462 / 0.555556 / 0.0 → 1.343442 / 0.555556 / 0.0 | 1.319533 / 0.555556 / 0.0 |
| neutral_cpc | 128 | 1.175813 / 0.555556 / 0.0 → 1.124428 / 0.555556 / 0.0 | 1.120504 / 0.555556 / 0.0 → 1.099171 / 0.555556 / 0.0 → 1.059562 / 0.555556 / 0.0 | 1.104839 / 0.555556 / 0.0 |
| gated_cpc | 0 | 2.003617 / 0.055556 / 0.0 → 1.400992 / 0.555556 / 0.0 | 2.008331 / 0.111111 / 0.0 → 1.430376 / 0.555556 / 0.0 → 1.355237 / 0.555556 / 0.0 | 1.331087 / 0.555556 / 0.0 |
| gated_cpc | 128 | 1.225591 / 0.555556 / 0.0 → 1.196036 / 0.555556 / 0.0 | 1.190465 / 0.555556 / 0.0 → 1.176185 / 0.555556 / 0.0 → 1.154627 / 0.555556 / 0.0 | 1.182763 / 0.555556 / 0.0 |

All runtimes complete/qualified; all fulljointfit witnesses remain false. From
source128 BP F accuracy improves12/18→17/18, one red trace exact. Mean A drops
17/18→15/18 and returns16/18; F returns16/18 after A updates. Thus new F fit
improves while the previously acquired A-red fit is lost and not fully recovered.
Actual game success after F/A was not measured; the earlier A-red game success
is a separate unchanged reference. PC/neutral/active all remain5/9/zeroexact at
source128 in every F/A phase, so stable accuracy is not evidence of retaining a
previously acquired skill. Source0 cohorts start unacquired and stay nonexact.

### Preserve both old-A cues

Source128; each arrow is beforeF / afterF16 / afterA16. Acquired means an exact
original TRAIN trace at the initial point; it does not relabel blue as acquired.

| Method | A cue | Correct rows and CE before → after F → after return | Initially exact |
|---|---|---|---|
| bp | blue | 8/9; CE0.168798 → 8/9; CE0.643094 → 8/9; CE0.309324 | False |
| bp | red | 9/9; CE0.182624 → 7/9; CE0.757313 → 8/9; CE0.340456 | True |
| pc | blue | 5/9; CE1.110022 → 5/9; CE1.088523 → 5/9; CE1.047739 | False |
| pc | red | 5/9; CE1.130986 → 5/9; CE1.109820 → 5/9; CE1.071384 | False |
| neutral_cpc | blue | 5/9; CE1.110022 → 5/9; CE1.088523 → 5/9; CE1.047739 | False |
| neutral_cpc | red | 5/9; CE1.130986 → 5/9; CE1.109820 → 5/9; CE1.071384 | False |
| gated_cpc | blue | 5/9; CE1.179138 → 5/9; CE1.163677 → 5/9; CE1.140647 | False |
| gated_cpc | red | 5/9; CE1.201792 → 5/9; CE1.188694 → 5/9; CE1.168608 | False |

BP red9→7→8correct, blue8→8→8. Red's loss is observed per cue; averaging cannot
turn the unacquired blue or PC-family cues into retention evidence. All per-cue
F/A intermediatecurves, source0curves and signedchanges remain in analysis.json.

## Frozen comparisons

Active-minus-PC (active-minus-neutral is identical here). Positive fresh/return
CEgain favors active; negative forgetting/netchange/Fcost favors active.

| Source | F CEgain difference | A forgetting difference | A returngain difference | Net A CEchange difference | Freturn CEcost difference |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0109888212 | +0.0099137975 | -0.0018803922 | +0.0117941897 | +0.0005649430 |
| 128 | -0.0218295494 | +0.0070522593 | -0.0180516330 | +0.0251038924 | +0.0063157414 |

All corresponding active-minus-PC/neutral accuracy/exact differences are0.
Active's source128 freshCEgain is.029555 versusPC.051385, and Areturngain.021558
versus.039610. PC and neutral full TRAIN probabilities match exactly/maxdiff0;
their reported curves/percue endpoints match exactly. No CPC benefit established.
Both secondary PC-minus-BP/neutral-minus-PC contrasts and all15signedmetric
differences at bothsourcepoints are retained with absolute levels in analysis.
Smaller trained-PC gains have different initialloss/headroom and do not establish
plasticityloss or a causal interference mechanism. BP source128-minus-source0
FCEgain is+.546977; PC/neutral−.562229 andactive−.573070, with allstarts/ends kept.

### Accepted H8 references (descriptive, no reruns)

Source128; native computation/parameters differ and these are fixed one-seed
records. This table is not a circadian causal or matched-compute comparison.

| Method | H8 F0CE → F16CE | H32 F0CE → F16CE | H8 Areturn16CE | H32 Areturn16CE |
|---|---|---|---:|---:|
| bp | 1.300716 → 0.576781 | 1.910845 → 0.404399 | 0.672060 | 0.324890 |
| pc | 1.294417 → 1.258822 | 1.175813 → 1.124428 | 1.213163 | 1.059562 |
| neutral_cpc | 1.294417 → 1.258822 | 1.175813 → 1.124428 | 1.213163 | 1.059562 |
| gated_cpc | 1.337343 → 1.323230 | 1.225591 → 1.196036 | 1.287635 | 1.154627 |

## Costs, execution and qualification

Exactly768native calls/29952trainingrows; benchmark512/27648, F128/1152,
return128/1152. P376calls3384rows:1080fullbenchmarkprobrows+2304recordeddecision-
lossrows.40actualteacherresets360steps, all10teacherspercase succeed/close;
0learnergames/replay/final. BP192derivedSGD/backwards; PCfamilies576parameter
updates44928nominal latent-rowsteps. Model9735float32elements38940equivalentB
is not totalmemory. Complete nativeadaptive/optimizer/gradient/RNG/copies,
history/projection/support/validation/report/Python/backend/time/RSS add costs.

| Method | Actual process seconds | Enclosing body seconds | External peak RSS B | Internal peak RSS B |
|---|---:|---:|---:|---:|
| bp | 28.1718520 | 25.3800411 | 434970624 | 432558080 |
| pc | 25.9117944 | 23.2985597 | 354267136 | 350846976 |
| neutral_cpc | 35.1207292 | 32.1593642 | 359768064 | 352063488 |
| gated_cpc | 34.7341390 | 31.9675084 | 354668544 | 352423936 |

SerialCPU1/GPU0/threadenv1/interop1/RSS768MiB/stage32MiB. Times are single measured
processes, not speedup/equalcompute evidence. Scienceparent125.7613474000/960s
CLOSED; engineeringroot1000+review350+sixchecks5.2252868999=
1355.2252868999/1600s CLOSED; priorgrantsCLOSED. All11commands0, noresource
problem/cancellation/nativefailure/retry. No production/test/dependency/Git/index/host/
network change;46trackeddirtypaths/unrelatedchanges preserved.

New two-file stageRuffformat/lint and4CLIpreviews pass. Unchanged production reuses
354scoped tests/WinLinuxmypy/Ruff and accepted native math/isolation criteria.
Source/protocol reviewed before native work. Rootstdlibsavedscores and independent
ACTUAL review check1080probabilityrows,2304decisionlossrows,512receipts,256phase
blocks, all120changes/8contrasts/16source-Acue matches, sourceequality/charges/
continuity/costs. Probe records do not persist fullprobabilities; their recorded
loss/decision reductions are verified without reconstructing missing values.
Skipped fullrepo/fullsuite/cleanclone/release/archive audits, final/heldout/RL,
replay/teacherchanges, baseline reruns, sweeps and newcapacity/dose experiments.

## Critical path

Completed R2.1/R4.2 H32 scientificchild; original boundedR2.1 and three-update
acceptance stand. Useful fullacquisition/retention/recovery, fullR4.2/R2.7,
heldout/recurrentRL/capture-prune/otherlanes/release/fullgoal remainOPEN.

Why next: rowfit now shows one newF exacttrace and loss of the previously exact
A-red trace. R4.2 requires actual complete episodes. The existing inline
ProbeRollouts already reads A/F games at paid F16/A16 endpoints, but accepts only
count snapshots and its9*allupdates law. H32 joint checkpoints have mixed
54-row benchmark plus9-row probe/return exposures and require their exact law;
the H32 TRAIN evaluator separately accepts only0/128(0/6912rows). Extend the
existing observer/evaluator explicitly; do not broaden those old contracts or
add deferredcapture/provenance/another runner, tune another dose or select red.

**Exact next implementation:** Extend the existing inline minigrid_probe_rollout observer and evaluator to explicit H32 after-F16/after-A16 disposable endpoints from source0/128. Bind updates=source+16/32 and training rows=54*source+9*relative; read both A and F blue/red target-free games on owned frozen forks with zero-update budgets inside the original parent clock. Preserve paid failures, benchmark/probe/return curves, state/RNG/budgets and count/H8/default routes. Run focused tests and independent review before any original-source game experiment.

[Frozen protocol](../../artifacts/runs/wide-neural-probe-study-20261018/PROTOCOL.md),
[commands/artifacts](../../artifacts/runs/wide-neural-probe-study-20261018/RESULTS.md),
[independent review](../../artifacts/runs/wide-neural-probe-study-20261018/REVIEW.md).

Document closure failures retained: first close_docs.py invocation raised AssertionError before any writes because it expected the heading ACTUAL — GO while the accepted review uses ACTUAL GO — saved evidence accepted. One attempted patch then failed context verification without changing files. After these two tooling failures, a direct literal edit bound the actual reviewed heading and this saved/doc closure was rerun. No learner, native case, score, budget or scientific artifact changed. The eleven supervised commands all passed; these were separate document-only tooling failures within the reserved root engineering grant.

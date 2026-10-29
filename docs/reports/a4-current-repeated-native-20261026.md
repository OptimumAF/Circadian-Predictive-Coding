# R4.2-A4.EC.X — native repeated-current exposure control

[Issue188](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/188).
Previous187 was verified software progress. First/only new native case completed
under its frozen protocol. One agent source/procedure/result SELF-REVIEW, not
independent acceptance. Full goal and broader parents remain open.

## Observed result

Extra current-packet exposure did **not** preserve the old task in this run.
The affine learner acquired A4, forgot A3, then relearned A3 on return while
forgetting A4. Both cues followed the same trajectory. All12actual known-TRAIN
game episodes completed and closed from the actual private causal checkpoints;
no new publication/deployment or final/probe evaluation occurred.

Each cell shows oldA3 goals / newA4 goals, denominator2fixed cues/task:

| Recipe | After A3 | After A4 | After A3 return | Fits / row exposures |
| --- | --- | --- | --- | ---: |
| Historical joint171, reused |2/2 /0/2 |2/2 /2/2 |2/2 /2/2 |6/90 |
| Historical current-only172, reused |2/2 /0/2 |0/2 /2/2 |2/2 /0/2 |6/54 |
| New repeated-current188 |2/2 /0/2 |0/2 /2/2 |2/2 /0/2 |6/90 |

The new control matches historical current-only goal/step endpoints despite
90rather than54fit-row exposures. At the same stated6fit/90row batch geometry,
repeated-current does not reproduce historical joint replay retention. This is
consistent with replayed old information mattering beyond simply repeating the
current data in this bounded affine replacement recipe. It is not a universal
causal result or evidence about every gradient/recurrent learner.

Frozen metrics: acquisition+1.0, old-retention delta−1.0, return gain+1.0,
recovery true for both old cues. Recovery here costs36return-fit row exposures:
18returning A3 arrival rows plus18current repeats, versus18in historicalcurrent54.
It is supervised relearning, not automatic recall. There were6successful games,
each9steps/return.49375/terminated, and6failures, each16steps/return0/truncated.
All outcomes/end flags/cleanup/source-role facts retained, including A4loss at
return. Failed goals are not incomplete evidence or hidden zeros.

## Reuse and comparison limits

Accepted171/172scalar results and protocols were reused, not rerun or restored
from old weight matrices. Same specified source hashes, A3/A4/A3 ordering,
public-cue representation, fixed least-squares recipe, seed17/max16/K7firstargmax,
checkpoint and metric definitions. Historical producer was the Windowsvenv;
the new case used that path, not the WSL environment used for software tests.
New versions are Python3.14.7/NumPy2.4.6, MiniGrid3.1.0/Gymnasium1.2.3/
pygame-ce2.5.7/cloudpickle3.1.2/Farama-Notifications0.0.6.

Old exact package/source-byte receipts are incomplete, and the recorded09ffgit
revision does not establish every original dirty scientific file byte. Current
source comparison could not prove strict historical byte identity; this was
disclosed BEFORE the new run. Thus the table is an exploratory/descriptive
historical comparison, not exact paired replication, parameter equality or
independent confirmation. Fit-count/exposure geometry is matched; wall time,
hardware/runtime overhead, RSS sampling scope and FLOPs are not controlled.
No CPC benefit, broad statistical reliability, unseen-layout generalization,
neural benefit, efficient-controller or release claim follows.

## Data and new costs

ONE170TRAINpayload428191B/SHA
`bce498c34c23d9ef128f9344bf7164a70edc6afa156d42e7534bb708b2e3c26f`
and ONE bundledTRAIN94705B/SHA
`5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508`
were admitted as read after GO; no preparation reread. Decoder prepares54bundled
TRAIN rows/selects18A3; unselected36notfit/forward. A4 was TRAIN from birth.
36unique source rows,54arrival exposures including18old return arrivals,
36explicit in-wake current repeats and0prior-packet bank/replay. Six batches
9/9/18/18/18/18; native counters2/18→4/54→6/90. Allsixfits and16TRAIN diagnostics
144rows finish BEFORE any of12fixed read-only games. No outcome selects work.

| New-case measurement | Observed |
| --- | ---: |
| LSE calls / fitted rows |6 /90 |
| Prediction calls / rows, incl TRAIN diagnostics |166 /294 |
| Port projection rows |384 |
| Port snapshot copies / typed-scalar upper bound |6071 /18451459 |
| Metadata owners / native environments / closes |9 /12 /12 |
| Actual native physics steps, attempted and returned |150 /150 |
| Source / bundled payload reads |1 /1 |
| Body / parent seconds |8.512329600 /8.855836400 |
| Sampled process RSS bytes |86294528 |
| LogicalJSON / countingUTF8 bytes, incl16KiBterminalreserve |3921643 /3921643 |
| Actual exclusive output / log bytes |3686637 /5299 |

Sampler starts after project imports and includes optional runtime preflight;
not exhaustive import/process-tree/allocation peak. Port-copy/scalar counts and
othercopy256MiB/sourceRNG32MiB reserves do not measure all allocation/FLOPs.
Do not rank universal speed/memory against historical runs. Original6update90s/
512MiB stream owner and one0update0replay150s/512MiB readout owner start before
admission/training/allphases; native/readout limits unchanged and readout start
equals observedstart. All150swhole/165hard/180parent/work/storage caps pass.

Both Python/NumPy global states captured after optionalimports BEFOREsources/
models matched immediately after run_adaptation BEFORECLIserialization and
again after the command. This does not retrofit old missing witnesses. Source,
actor and each of3readout states unchanged. Original protocols/receipts/clock/
budget/paid-prefix guards held. No new teacher/collection/SGD/optimizer/backward/
Torch/randommodeltraining/probe/final/consolidation/promotion work. Gymseed17
local12environment resets are separately counted, not global model RNG draws.

Sunk shared170source collection18teachersteps/2env/2.5406233s and171hashprep
522896B/.8282145s remain separate provenance/setup costs, not new authority.
Pre-existing bundled generation cost was not remeasured. All original negative
results and spent171/172/177/89/182/184allowances remain unchanged.

## Procedure, checks and artifacts

Exactly once at root:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/a4-current-repeated-native-20261026/execute.py
```

Existing172finite driver adapted, no new framework/product changes. It verifies
187accepted37Windows/37Linux/static receipts and current three-file product
hashes, current helperstatic pass, reviewGO and exclusive absent outputs. Reused
187checks; no unchanged fixture/model cohort rerun. Twohelper Ruffformat/check/
format--check and default+Linuxmypy `--explicit-package-bases --follow-imports=silent`
PASS firstattempt,7.515297400s. No test/static/native failure or retry. One absent
guessed helper path, one guessed docs path and one unexecuted JS comment-string
syntax error were simplified without model work. Forcedtimeout/tree cleanup
source-reviewed only, not injected. Unknown timeout would spend full reservation.

Local `artifacts/runs/a4-current-repeated-native-20261026/` retains frozen
PROTOCOL/REVIEW, case/execute, static receipts, execution/native terminal summary,
work sidecar and raw actual full parameters/probabilities/public inputs/receipts/
end flags. Result review inspected emitted same-process summary, NOT rawactual
again or old model matrices. Nativeweights/data remain local; paths do not imply
committed/published bytes. No install/download/GPU/browser/CI/Gitmutation/delete/
reset/agents. Fullsuite/cleanclone/newwheel/independent/multiseed/statistical/
neural/CPC/release gates skipped. Historical plans/log frozen perAGENTS; issues
are live tracking. Full goal remains ACTIVE.

## Critical path / exact next action

This narrow exposure-control question is answered; do not add another dose,
seed, copy/capture/provenance/control or packaging sequel. Reconciliation found
R2.1 probes and existing affine held-out D binding126 already implemented (with
partial native outcome), CLM paired replay20 already implemented, and34native
comparison still FAILED-SPENT. Reuse these; do not restart their old handoffs.

The explicit R5.1 parent24 source/model/terms requirement remains unchecked.
Existing `docs/upstream/context-and-causal.md` already records distinct ContextLM
and causalQwen contracts, but its bounded inspection could not identify released
trained ContextLM RL checkpoint/tokenizer/weight terms. A harness/source pin is
not a released trained model. This is a real prerequisite before extending that
learner family and cannot be filled with random/unofficial substitute weights.

Next exact validation: bounded official ContextLM release/checkpoint metadata
lookup from the existing pinned upstream record, no weights/tokenizer vocabulary/
benchmark content/downloads/model calls. Record exact released revision/tokenizer/
terms if available, or preserve the external unresolved prerequisite explicitly;
update existing upstream records and #24 rather than creating another loader or
catalogue framework. Freeze metadata limits before network inspection. No final
outcomes or changed baselines are used to choose that work.

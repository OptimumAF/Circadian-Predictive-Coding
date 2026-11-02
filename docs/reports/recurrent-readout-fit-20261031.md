# Frozen recurrent readout feasibility — #376 / R2.1-RNN.FIT

[Canonical scoped completion and handoff](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/376#issuecomment-6548452501).
376closed; the full repository goal remains active.

The existing cold recurrent features permit a separate affine readout to
interpolate all nine TRAIN actions per cue (18/18 total). Both designs have
rank9 with9columns (eight hidden features plus intercept). This is known-row
interpolation, not native task success, generalization or a diagnosis of the
unavailable132trained representation. No optimizer/hidden-learning change is
justified by this result alone.

| Fixed TRAIN cue | Correct | Rank | Squared error | Coefficient L2 | Max absolute coefficient |
|---|---:|---:|---:|---:|---:|
| A-blue |9/9|9|6.450401606939506e-28|39.431122398328704|23.814587586870758|
| A-red |9/9|9|3.995580484944425e-28|32.887121935711164|13.64321466232821|

All18required-action margins are positive and approximately1. Blue target and
predicted sequence [2,2,0,3,1,2,5,2,2]; red [2,2,1,3,0,2,5,2,2].
All seven action logits, nine margins and full nine singular values per cue
remain in native-result.json. No unfavorable cue, rank or row was discarded,
and no cutoff/regularization/metric/seed/dose/representation was adjusted.
Nine independent rows and nine columns demonstrate interpolation capacity on
this small TRAIN set, not semantic representation, robustness or improvement
over an accepted learner.

## Fixed composition and isolation

Direct existing APIs suffice; no product module or new wrapper was needed.
One new PublicCueRecurrentLearner, original two raw148/H8/K7/seed17 heads,
lr.03/momentum.9/max16 policy; no SGD or native source update. The existing
zero-reset recurrent activation reader is called once per original A-blue/red
9rowTRAINepisode, on its corresponding cold head. It performs18single-row
native forwards, including18discarded original output projections. Its144hidden
float64 scalars occupy1152B total. Declared cue order in this offline TRAIN
check does not qualify future target-free native routing.

Two existing categorical one-hot minimum-norm LS fits use fixed appended
intercept/RCOND1e-12,9rows each. Two existing core logit readouts evaluate only
those same TRAIN rows. No native source prediction, backward/gradient, native
TRAIN, physics, teacher, replay, probe, final, promotion, deployment or controller.
The fitted readouts are disposable objects; their2fits18rowexposures are paid
learning work, distinct from the unchanged native source0updates/0rows.

A real StableActor serving fork was created but never predicted. Source and
actor exact policy/state owners, parameters/gradients/momentum/counters and
globalTorch/NumPy/Python RNG are checked around reader/fit/readout calls and
remain unchanged. Raw TRAIN/target/hidden arrays retain their copied values.
No hidden arrays, coefficients or native tensor states are exported. JSON
metadata is not a resumable policy and cannot be bootstrapped for games.

Input: current bundled TRAIN94705B,
SHA2565a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508,
read once through existing bounded resource/decoder(raw_frames=True) and
six-source typed admission. Only original A episodes enter fits; B/C labels
are not scored/selected/fit. No historical132model or old140/143/133partial
endpoint is read/restored. Original meanCE/weighted/default and matched
BP/PC/neutral/active/affine/robot controls stay unchanged.

## Prospective gates and actual costs

Freeze before models: issue376comment6548318412 and stage PROTOCOL.md.
One cold-source/readout question, no sweep or comparative benefit study.
Independent PRE GO2reads/.2395155s required the named compile gate. Final
stdlib compile, Rufflint and format --check returned0 before execution,
recorded in statics-final.json (three explicit exits0). Static preparation
executes no project imports/models; accepted unchanged activation/LS math reused.
No correctness-model cohort or broad unchanged mathematical rerun.

One hidden owned process72312 completed terminal0/no host expiry: body
9.430904700013343s/peak318914560B, host9.8240023s. All attempted measured calls
returned; all frozen physical caps held. No failed model/parser/static attempt
or budget amendment. SPENT/exclusive attempt markers retained; no retry.

- 2native constructors/6private randn/2608initialized scalars;
  2optimizer constructions,0SGD/backwards/autograd/nativepredict/native TRAIN.
- 2activation readers/18rows;18native single-row forwards/18rows.
- 2LS fits/18rows,2design solves/18rows,2LAPACK calls/18rows;
  2logit readouts/18rows. The2postfit products and2prediction products
  (18rowvisits each set) are derived from unchanged core source, not direct
  matmul instrumentation. These named costs are not allFLOPs/copies/heap/I/O.
- 54native snapshot validations/21Tensor.clone calls/10332cloned scalars.
- Result3977B under128KiB; Python3.14.7/NumPy2.4.6/Torch2.14.0+cpu.

The one diagnostic budget has2permanent LSfitattempt charges, original birth
230131.791072/last clock230132.746923, max2fits/60s/512MiB. Source SGD is a
separate zero counter; these fit charges are not erased or renamed as free
diagnostics. No source/serving budget or lifetime was renewed.
Whole90s/512MiB, host100s and stage4MiB/output+log128KiB apply separately.

Stage artifacts/runs/recurrent-readout-fit-20261031/ holds source/protocol/
static/counter/error/host/capture receipts and reviews. PhysicalD: HEADd9119e8,
inherited dirty checkout and all prior negative/SPENT evidence preserved.
Exact child command with TEMP/TMP D:\Dev\_Temp, physical repository cwd:

    .venv\Scripts\python.exe -B artifacts/runs/recurrent-readout-fit-20261031/native_observe.py

PowerShell Start-Process used WindowStyle Hidden, one retained process handle,
redirected stdout/stderr and100s deadline. No output-yield restart or another
model process. No install/download/push/delete/global settings.
Full suite/unchanged parity/cleanclone/wheel/Linuxnative/browser/GPU/CI/final/
controller/comparative cohorts and production types skipped: product behavior
and core math unchanged. Driver lint/format/compile pass; source/procedure/
result self-review is separate from independent interpretation. One attempted
combined report/comment tool call had a JavaScript quoting error before edits
or calls executed; it performed no model/IO/scientific work.
Independent POST ACCEPT of actual procedure/results/interpretation: five total
PRE/POST read-only calls/.6326807s, no truncation/tests/models/numerical imports
or historical JSON reads/replication. Final stage28124B, result3977B/stdout1310B/
empty stderr, all storage caps held before final reviewer notes. The named
compile receipt predates native.attempted. Finite exception ended; unused read
allowance does not carry forward. REVIEW.md/timing ledger records scope.

## Next useful implementation

375located wrong Forward choices on the teacher trajectory;374ruled out absent
initial updates in its different tiny trace. This result shows cold features
can interpolate fixed A. It does not explain132trained failures, establish this
control as preferable or prove that changing readout learning solves control.
373simpler jointcold8/8 and132finalC2/2/Aredsuccess remain valid within scope;
140A32negative and all lost/partial cases remain FAILED-SPENT. No baseline tune.

Next [377](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/377):
choose the smallest explicit fitted-readout recurrent policy through the
existing activation/core-fit and native policy-loop ports. Preserve ordinary
learner/default/weighted behavior and public initial-cue routing. Require
focused ownership/role/cost/failure/sequence/default checks, followed by one
prospectively frozen finite target-free native A-blue/red qualification.
Keep all seven actions; no teacher lookup/fallback or supplied action sequence
during play. Preserve float64 readout semantics rather than silently casting
coefficients into the native float32 head.

Do not claim known-row fit as success, reconstruct these disposed376objects,
or start another rank/gradient/capture/audit increment. No next implementation
fixture/native allowance here. Full goal/acquisition/retention/recovery/
recurrentRL/heldout/scientific/release requirements remain open.

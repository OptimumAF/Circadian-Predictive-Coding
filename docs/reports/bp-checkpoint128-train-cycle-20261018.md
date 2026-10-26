# Real-source BP checkpoint128 TRAIN cycle

## Result and scope

The single frozen seed17 case completed and qualified against the accepted
original BP report: identical TRAIN/F inputs and recipes, all270 benchmark
probability rows with maximum difference0, and unchanged reported F/old-A/A-return
curves. The three diagnostic updates affected an owned copy only, before F
collection. Source/actor/state/input/RNG/budget guards passed; final evaluation
remained not_run. Independent SOURCE/PROTOCOL and ACTUAL review accepted this
bounded descriptive result.

The fixed cycle used B-blue TRAIN wake, A-blue replay, then A-red replay.
Old-A mean CE decreased0.39374376833438873→0.39061352610588074,
change−0.0031302422285079956. All three aggregate old-A gradient dot actual
displacement and dot carried-momentum displacement were negative.

| TRAIN step | old-A CE before | old-A CE after | CE change | gradient · actual delta | gradient · carried momentum |
|---|---:|---:|---:|---:|---:|
| B-blue wake |0.3937437683|0.3929515183|−0.0007922500|−0.0011689031|−0.0009808483|
| A-blue replay |0.3929515183|0.3918360621|−0.0011154562|−0.0011663352|−0.0003866778|
| A-red replay |0.3918360621|0.3906135261|−0.0012225360|−0.0014342720|−0.0009577563|

Negative dot indicates local first-order descent, not a causal contribution to
finite-step CE. Per-parameter evidence is mixed for carried momentum: hidden bias
has positive dots on the first two steps, and feature→hidden weights have a
positive dot on A-blue replay. All blocks, signs, cosines and zero-norm reasons
are preserved in the raw report/analysis. Momentum-buffer formula errors are0;
maximum parameter-step formula error8.66129994392395e-8.

This cycle provides **no evidence of aggregate old-A ascent in this local TRAIN
trajectory**. It contains no momentum intervention and no F inputs, so it cannot
isolate momentum's causal effect or explain historical F/rehearsal interference.
The earlier warm BP rehearsal result still worsens both F gain and old-A loss.
All old-A and joint exact-trace acquisition failures remain unchanged. Original
bounded R2.1 diagnostics are accepted; useful neural acquisition, retention,
recovery, generalization and circadian benefit remain unestablished.

## Costs

Exactly195 native calls/7515rows:128benchmark/6912rows,3observer/27rows and
64isolated F/A calls/576rows. Observer includes18 separately charged replay rows.
Original classifier work94forwards/846rows. Diagnostic counters observe15CE
forwards/135rows and9backwards/81rows, plus54admission projection rows127872
array-equivalent bytes and319680returned diagnostic projection bytes.

From the unchanged native source recipe, native training additionally performs
195CE forwards/backwards/optimizer steps over7515rows. Thus combined forwards
are304/8496rows and backwards204/7596rows. These totals are **source-derived** in
this real run; the preceding synthetic engineering test instrumented operations.
Narrow diagnostic counters are observed directly. API validation1 excludes the
integration's separate checkpoint validation/copy and repeated parent guards.
Retained arrays, native states, Python objects, copies and serialization add costs.

Actual case28.5584225seconds, parent28.8262888seconds, external peakRSS435957760B.
Pilot27.2554826seconds includes nested observer0.3049988seconds; do not add them.
Ten teacher episodes/90physics steps, zero learner games/final episodes. Saved-only
readback parent0.4957774seconds, no model work. Science closes29.3220662/300seconds,
no retry, alternate seed or replacement. Engineering1003.1509964/1300seconds CLOSED;
all prior grants CLOSED. Stage remains below32MiB/RSS768MiB, CPU1/GPU0.

## Evidence and next implementation

[Frozen protocol](../../artifacts/runs/bp-checkpoint128-real-20261018/PROTOCOL.md),
[native JSON](../../artifacts/runs/bp-checkpoint128-real-20261018/native-bp.txt),
[saved analysis](../../artifacts/runs/bp-checkpoint128-real-20261018/analysis.json),
[exact commands/failures/skips](../../artifacts/runs/bp-checkpoint128-real-20261018/RESULTS.md),
[independent review](../../artifacts/runs/bp-checkpoint128-real-20261018/REVIEW.md).
Historical full F/return probability vectors are absent: saved outcome equality
does not independently reconstruct their probabilities. Geometry is saved-scalar
consistency plus reused accepted source/NumPy-oracle evidence, not new raw-vector
recomputation. The initial missing TRAIN collection-metadata review gate was
repaired before native admission; no native failure or retry occurred.

Reassessment: another momentum observer or seed/dose change is not the next
implementation. Existing TRAIN bounds have54distinct conflict-free inputs, yet
the one-hot least-squares control and longer H8 fit each miss two A labels.
Least-squares52/54 does not establish linear classification impossibility.
Implement a **separate bounded multiclass perceptron control** through existing
TRAIN-fit/learner ports, with a fixed mistake rule, owned snapshots, row/update/
prediction costs and a small separable synthetic acceptance case. Keep original
least-squares/BP/PC/neutral/active recipes frozen. A real-source fit requires its
own small reviewed protocol; negative/inconclusive outcomes remain valid. This
adds a simple trainable classification control without a new orchestration layer
or another checkpoint audit. Full research, adaptation and release goals remain open.

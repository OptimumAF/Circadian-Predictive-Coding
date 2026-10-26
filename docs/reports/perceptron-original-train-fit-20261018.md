# Original TRAIN catalogue: fixed perceptron fit

## Result

The sole frozen128-pass case completed and qualified, but its full-fit witness
is **false**. On the original six nine-step TRAIN traces, accuracy improved from
6/54 to52/54, exact traces from0/6 to4/6, and unweighted mean CE from
1.9459101490553137 to0.5659225130116193. All128 passes executed; no successful
intermediate endpoint was selected.

| TRAIN episode | Correct rows at128 | Exact trace | CE at128 |
|---|---:|---:|---:|
| A-blue |8/9|0|3.0701134573|
| A-red |9/9|1|0.0004442051|
| B-blue |8/9|0|0.2131944411|
| B-red |9/9|1|0.0705495668|
| C_train-blue |9/9|1|8.6351e-16|
| C_train-red |9/9|1|0.0412334077|

A-blue step2 predicts1 instead of0; B-blue step1 predicts0 instead of1. A-blue
CE **worsens** from1.9459101491 to3.0701134573 despite aggregate improvement.
The128th online pass still makes4mistakes; total row-mistake updates are1743.
Every pass diagnostic and both fixed readouts remain in the artifact.

The accepted bounds helper reports54distinct conflict-free canonical inputs,
row-accuracy ceiling1 and CE floor0. Endpoint accuracy gap is2/54 and CE gap
0.5659225130. A finite negative perceptron run does not prove linear
inseparability or general incapacity. No retention, recovery, generalization,
momentum causality or CPC advantage follows from these TRAIN scores.

The historical original BP128 result remains45/54 with CE0.3929002645; its
longer512 fit and the one-hot least-squares control each remain52/54. The new
linear classifier has higher accuracy than that BP128 result but worse CE.
It is a different objective, parameterization, dtype and computation; this is
not a matched-compute winner claim. All other BP/PC/neutral/active controls and
earlier negative/incomplete studies were reused without rerunning or changing them.

## What executed

The existing saved-source CLI read only original complete TRAIN148 records from
the accepted BP report, built raw296 initial/current histories, and composed
the existing invariant wrapper/NativeLearner/AttributionStream. All coefficients
started at zero. The supplied-order, unit-step, first-argmax mistake rule was
unchanged; no RNG, optimizer, shuffling, averaging or early stop was added.
All6912row exposures completed under the original bound30-second/128-update
allowance. Source and actor remain unchanged, and no final/probe data, new
teacher, game, physics, replay or control fit was used.

Independent SOURCE/PROTOCOL GO preceded execution. Saved-only readback and
independent ACTUAL review checked all108probability rows, first argmax, clipped
CE, all per-episode/context/gap reductions, all128receipts and the1743mistake sum.
Output JSON and stdout agree. Reused accepted implementation179scoped checks,
math oracle, source/state/isolation and static gates remain applicable because
production source did not change. Bound-key construction and native coefficients
were not independently reconstructed in this saved-only readback.

## Cost and limits

128native pass calls/6912training rows/6912online decisions;1743mistakes each
update two297-coordinate columns.12classifier calls/108score rows. These internal
primitive operations follow from the accepted source and native counters; they
were not separately instrumented here. Zero SGD/backwards/optimizer steps and
zero final/probe updates. Diagnostic, projection, validation and copying work
remain additional costs; there are no new fits in the readback.

2079coefficients correspond to16632float64-array-equivalent bytes, with tuple/
Python overhead additional. Existing128stream input copies total16422912logical
bytes, plus128304template and128304score-input bytes and6048prediction bytes.
At most two complete checkpoints are retained; these logical payload sums are
not a measured peak. Actual process5.0598444seconds, external peakRSS78409728B;
enclosing parent5.3259958seconds. Fit body4.3423061seconds/internalRSS78401536B.
Saved readback0.4902314parent seconds, no model calls. Science closes
5.8162272/120seconds; engineering751.8387362/1000seconds CLOSED, all earlier
grants CLOSED. No failed or replacement native case, seed or dose extension.

## Artifacts and next implementation

[Protocol](../../artifacts/runs/perceptron-real-fit-20261018/PROTOCOL.md),
[native JSON](../../artifacts/runs/perceptron-real-fit-20261018/native.json),
[analysis](../../artifacts/runs/perceptron-real-fit-20261018/analysis.json),
[commands/resources/skips](../../artifacts/runs/perceptron-real-fit-20261018/RESULTS.md),
[independent review](../../artifacts/runs/perceptron-real-fit-20261018/REVIEW.md).

Do not extend this linear dose or add another observer in response to52/54.
Next implement a separately named bounded degree-2 polynomial-kernel perceptron
through the existing native/TRAIN-fit ports. Fix the kernel as
`(1 + dot(x, y) / D) ** 2`, retain the unit mistake rule, and bound the owned
prototype dictionary (maximum128) with atomic capacity refusal and explicit
kernel/prototype/state costs. Verify nonlinear XOR learning and complete owned
fork/restore before any real-source fit. Why this: a nonlinear capacity control
can test richer classification without changing the frozen linear or neural
recipes; this result does not establish that nonlinearity is mathematically
necessary. Its real use requires a new small reviewed protocol. Full acquired
retention/recovery, broader research and release goals remain open.

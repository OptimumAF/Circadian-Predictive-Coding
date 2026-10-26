# Public-cue affine TRAIN control

The existing disposable linear-fit CLI can fit one model for each initial
visible cue using its original solver and objective:

```powershell
python -m scripts.run_minigrid_linear_fit_control --context-linear-fit --describe
```

After declaring a local fit budget:

```powershell
python -m scripts.run_minigrid_linear_fit_control --context-linear-fit --source artifacts/runs/joint-catalogue-study-20261018/bp.txt --output fresh-context-linear.json
```

The new flag is mutually exclusive with the other diagnostic modes. It refuses
fit-dose, time, encoding and rollout overrides. The original pooled mode remains
the default. Output files are exclusive and bounded to 512 KiB for this mode.

## Executed result

The single original-source case classified **54/54 TRAIN actions correctly**,
with all six traces exact. Each public-cue head classified 27/27 rows correctly.
The minimum target probability margin was 0.06617507953304086; every margin was
positive. Global CE was 1.1871627092232293.

Both design matrices had rank 20 and squared error approximately 2/3. This is a
complete action-classification fit, not exact one-hot interpolation or zero CE.
It is a constructive witness that this two-head affine capacity can represent
the original known labels. It does not establish online perceptron convergence.

All known TRAIN traces were available together. No game, chronological learning,
retention/recovery experiment, held-out layout, probe or final evaluation ran.
There is no deployed task-success or circadian-advantage claim. Earlier pooled,
perceptron, kernel and neural negatives remain unchanged. Capacity, objectives
and costs differ; this is not a matched-compute comparison.

## Composition and ownership

Complete causal TRAIN admission validates all current and next views before
either fit. Only the initial public image chooses a head; metadata names and
labels cannot route it. The existing per-head fitter uses canonical 296 current
coordinates, K7, a constant intercept and fixed equal-row one-hot least squares
with `rcond=1e-12`. Each model has297 x 7 float64 coefficients.

The composition retains separate coefficients, rank, singular values, residual,
probabilities, metrics and work for both heads. It checks the preceding paid
result before another call and after return, refuses altered or replaced
evidence and preserves failures. Aggregate metrics appear only after both heads
complete, in original source order. The full-fit witness requires every complete
source trace to be classified correctly; a complete negative remains valid.

Why this: the earlier input diagnosis ruled out pairing and exact aliasing, but
could not prove affine attainability. Composing the existing fitter answers
that question without another optimizer or game variant. These fit models have
their own objective and provenance; they are never injected into perceptron
snapshots as invented update history or runtime authority.

## Evidence and costs

25 focused checks qualify per platform (23 original passes and two affected
checks after a failure-report preservation repair), with scoped Ruff/format/mypy and
source/procedure/result self-review. Software fixtures used 17 real least-squares
fits over 36 rows and 15 prediction calls over 32 rows per platform; they are separate charges.
Linux's additional LAPACK-path counter was unavailable under its older NumPy
layout; the core fit-entry counter remained operative. The real-data case used
the pinned Windows producer and observed both LAPACK entries.

The real-data case cost two fits over 54 rows, two predictions over 54 rows, nine score
reductions over 162 rows and 228 nested projection entries. Parent elapsed was 11.5284477 s;
sampled peak RSS was 53,026,816 B. No optimizer, teacher, simulator, replay, probe or
final work. Source bytes and caller NumPy RNG remained unchanged.

`artifacts/runs/context-linear-train-fit-20261024/` retains PROTOCOL.md,
actual.json, all test/work/parent receipts, saved-result checks and RESULTS.md.
The launch is permanently spent; do not repeat it to select an outcome.
[Issue105](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/105)
contains scoped acceptance and handoff. Self-review is not independent review.
Full R4.2, scientific benefit and release readiness remain open. A future game
observation needs a separately frozen protocol and an ordinary readonly policy
facade over these already-fitted models, with no refit or old-budget renewal.

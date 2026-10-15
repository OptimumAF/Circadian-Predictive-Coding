# Fixed/adaptive coupled inference: allocated-time pilot, 2026-10-15

## Outcome

The one frozen six-case invocation is **incomplete**, CLI exit 2: adaptive seed 41
exceeded the prospectively fixed 11 ms maximum window. Its nine committed updates
and work records are retained, with both heldout score dictionaries null. No
window extension, retry, seed replacement or threshold tuning was performed.
All 29 adaptive updates reached the six-step cap; none stopped early or fell back.
This pilot demonstrates no stopping savings or general learning/speed benefit.

## Protocol and implementation

Exact prospective protocol:
`artifacts/runs/coupled-inference-comparison-20261015/PROTOCOL.md`.
Seeds 23/41/59, PC D2/hidden(8,8), scalar BCE, precision(1,1), parameter rate .03,
latent rate .1/cap6. Fixed inference is unchanged; adaptive uses the unchanged
default min1/gradient1e-6/relative-energy None. Both methods start from exactly
the same initialized tensors per seed, all saved in `pilot.json`.

Fixed train16 RNG7421 and heldout128 RNG9221 uniform[-1,1], target
`x0 + .25*x1 >= 0`. Available labels match; update/exposure counts can differ.
Order: 23fixed/adaptive, 41adaptive/fixed, 59fixed/adaptive. No warmup/calibration.
The equal **10 ms allocated** windows include native policy/check/candidate work,
fresh per-call buffers, clock/resource checks, records and bookkeeping, including
the terminal deadline branch. Initial setup/copies and deferred scoring are
outside windows, included in whole-case/global time. Calls are not preempted.
Successful observed windows must be [10,11] ms inclusive.

All training in each case precedes detached initial/final target-free heldout
prediction. Only common BCE clip1e-12/accuracy>=.5 metrics use final labels.
Scores do not feed back to subsequent fixed cases. Any failure suppresses all
that case's scores, including initial scores. The failed adaptive seed is not
imputed using its successful fixed counterpart.

## Every declared case

Window/call times below are milliseconds; all displayed values are rounded.

| Seed | Method | Status | Updates | Window | Whole calls | Initial BCE | Final BCE | Initial accuracy | Final accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 23 | fixed | complete | 10 | 10.1090 | 9.9298 | .812998 | .768877 | .2578125 | .375 |
| 23 | adaptive | complete | 11 | 10.7406 | 10.5358 | .812998 | .764787 | .2578125 | .390625 |
| 41 | adaptive | overrun, unscored | 9 | 11.4425 | 11.1133 | — | — | — | — |
| 41 | fixed | complete | 12 | 10.8053 | 10.6159 | .606214 | .580161 | .8984375 | .9375 |
| 59 | fixed | complete | 12 | 10.5238 | 10.3832 | .618301 | .608415 | .78125 | .7578125 |
| 59 | adaptive | complete | 9 | 10.1786 | 9.9667 | .618301 | .610770 | .78125 | .7578125 |

Failed seed41 overrun is 1.4425 ms; its last whole iteration including terminal
checks is 1.8542 ms. This retains the real nonpreemptible-call timing failure.

All 63 attempts committed. Fixed completed34/adaptive29 updates; nominal latent
proposals204/174, inference energy evaluations238/203, parameter-candidate energy
evaluations34/29, gradient norm reductions0/203, monotonicity comparisons0/174,
fallbacks0/0, training-row exposures544/464. Every adaptive reason is `max_steps`.
These are known call counts, not FLOPs. Threshold arithmetic is timed, not
individually enumerated. Fewer total adaptive latent proposals here arise from
fewer updates, not early stopping.

## Seed summaries and interpretation

`pilot.json` retains all three ordered seeds for every method/paired metric.
Nominal marginal 95% t intervals use n3/df2/critical4.303, with no simultaneous
coverage claim. Fixed vectors are complete; every adaptive/paired complete mean
and interval is null because seed41 failed. Observed-only descriptions are
explicitly separate and do not become a two-seed replacement experiment.

Paired adaptive-minus-fixed final BCE is
`[-.004089968196232752, null, +.0023553263359756826]`.
Paired committed updates `[+1, null, -3]`; paired actual window seconds
`[+.0006316000071819872, null, -.000345199994626455]`.
Neither qualified pair establishes a consistent benefit. Short windows, runtime/
cache/scheduler variability, the explicit 10% qualification and fixed data/n3
constrain interpretation. Intervals mix initialization and runtime variability;
this is not exactly equal observed wall time, CPU time or matched exposure.
No CPC, scaling-limit, general convergence or generalization claim follows.

## Resources and verification

Actual execution .08661519999441225 s after imports/before JSON encoding,
sampled absolute RSS39,337,984 B (start36,577,280 B,102 samples), output115,023 B,
stderr0 B; CPU1/GPU0/download0/replay0/environment work0. All global ceilings
passed. The case qualification failure is separate from global resource success.
Science63/480 training calls spent; no retry. Engineering4/24 real calls spent;
all unused allowances closed. Joint67 training calls within512.

58 distinct focused cases pass: worker39 (including four real native calls),
adapter18 with no training, and one terminal-boundary control. The terminal
change additionally passed38 existing synthetic cases; the unchanged native
learning control was not repeated. All scoped Ruff/format and Windows/Linux
mypy gates pass. Source/benchmark/state isolation independently reviewed before
execution. A test-only heterogeneous trace annotation/type cast and a new test's
fixture-unpack error were corrected; initial failures remain recorded, no
scientific retry or acceptance weakening. Whole-repository, clean-clone/release,
historical audits, prior pilots and large sweeps were skipped; prior accepted
native math/default evidence reused unchanged.

Artifacts: `artifacts/runs/coupled-inference-comparison-20261015/` contains protocol,
raw `pilot.json`, exit/stderr, focused JUnit/text and scoped static results.
Independent actual outcome/parent assessment is recorded in `RESULTS.md`.

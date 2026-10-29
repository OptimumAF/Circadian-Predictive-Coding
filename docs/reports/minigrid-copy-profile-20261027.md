# Affine coefficient validation cost — P7-MG.COPY / #238

Repeated validation was a measured bottleneck in the existing target-free
MiniGrid evaluator fixture. The coefficient validator now uses explicit loops
instead of nested generators. It still checks every row and scalar on every
invocation, preserves exact builtin type/shape/finite rules and error text, and
refuses corruption after earlier success. No snapshot, callback, source, actor,
budget, counter, algorithm, prediction, solver or evidence check was removed.
There is no cache or identity shortcut.

Forty-four focused cases pass on Windows and cached Linux. A fixed unprofiled
microbenchmark measured median1000-call time .15705269994s before versus
.07901689992s after, about50% less time for this validator. All three alternating
pairs agree in direction. This is a literal296/K7 record measurement, not a
native-game, whole-runtime, learning, retention, efficiency or CPC claim.

## Diagnosis and bounded scope

The previous turn was progress237: a native naming regression was repaired.
#236 remains FAILED/SPENT with an unadmitted warm baseline and no validation
endpoints. Its measured13,833 snapshots/314 predictions motivated this cost
investigation; it was not rerun, reconstructed or backfilled.

The existing timing/evaluator fixtures and historical exact-comparison repair
were inspected and reused. A tiny namespace state does not exercise the real
coefficient validation, so two fixed calls used the existing public affine
reference-window evaluator with a literal warm2/18 snapshot, real empty native
metadata owners/copy validators, scripted predictions and the existing fake
environment. Four A3/A4 TRAIN fake games/two fake steps each. First a thin
immutable-budget boundary; then actual `_Study.require` with its four synthetic
TRAIN sources and a literal warm owner. Original paid fitting/trial history,
real learned forward and physics were excluded. This does not apportion the
full236 run or prove native task success.

| Profiled fixture | Call seconds | Coefficient validation inclusive seconds |
| --- | ---: | ---: |
| Thin boundary | .55318029993 | .5104994 |
| Study boundary | 1.40245739999 | .7479060 |

The study fixture also paid .5789426s inclusive exact comparison. Inclusive
spans overlap and cannot be summed; cProfile amplifies generator-call overhead.
This diagnosis justified targeting the coefficient loop rather than weakening
owner/preservation checks. The measured implementation scope was amended on
GitHub before the source edit, with original evidence and budgets retained.

Both calls completed: body3.04583890014s/parent3.36585519998s, peak68,034,560B.
Attempted==returned: five empty native metadata owners/518 snapshots/1017 copies/
1308 fit-record validations/16 scripted predictions16 rows/eight fake environments/
16 fake steps/eight closes/245 boundary callbacks. Original source, global RNG,
budget identity/start/clock/sampler and zero charges were preserved. No genuine
fit, solver, learned prediction, teacher/native physics, payload/probe/final or
model random work occurred. Scalar aggregate profile only, no weights or raw
profile/state history output. Per-case work includes preparation and call;
post-call checks additionally appear in global totals. Preparation was
.00028849998s/.02632519999s; profiling/capture/import overhead is additional.

## Implementation, tests and benchmark

Production change: only the coefficient block of
`src/core/categorical_linear_fit.py::CategoricalLinearFit.__post_init__`.
The rest of the class checks and all fit/predict arithmetic remain unchanged.
New `tests/test_categorical_linear_fit_validation.py` supplies42 pure cases:
valid widths1/296 and classes2/7, malformed outer containers/lengths, bad rows
and scalars at first/middle/last positions, builtin type restrictions, NaN/Inf
and post-success corruption. Two existing evaluator timing cases cover ordinary
and prediction-failure execution with unchanged paid work/guard/error behavior.
Both platforms report44PASS, zero test failures/errors/skips.

Windows software .73564699991s body/1.70200070017s parent/RSS75,747,328B;
Linux4.226154007s/9.95005570003s/RSS72,253,440B. Each cohort pays eight scripted
owners/158 snapshots/87 attempted record validations/ten scripted prediction
attempts10 rows/six fake environments/eight fake steps/six closes. Native
constructors/train/predict and LAPACK are forbidden during these tests. Known
injected prediction failures remain test evidence, not hidden free work.

The preserved old core file is SHA-bound to the original profile qualification.
The unprofiled benchmark loads that module and the changed module, constructs
one identical literal record each, and runs exactly three alternating old/new
1000-call pairs, with no warmup, selection or tuning:

| Pair and order | Old seconds | New seconds |
| --- | ---: | ---: |
| 0 old/new | .15557769989 | .07901689992 |
| 1 new/old | .15705269994 | .07903619995 |
| 2 old/new | .15780639998 | .07885019993 |

Both validators then reject an infinite last coefficient after prior success.
Benchmark6000 validation calls/two manual constructors/two known refusals;
body .72229309985s/parent .96281290008s/RSS32,460,800B. No model fitting or learned
forward. The median ratio is1.98758 for this function only; native runtime gain
was not measured. No Linux benchmark duplication was needed for this local
Windows engineering observation.

## Reproduction and evidence

Stage `artifacts/runs/minigrid-copy-profile-20261027/` retains original protocol/
qualification/source pins, source preedit, fixed producer and scalar profile,
scope amendment, new source patch, test XML/logs/parents, all six benchmark
times, typing diagnostics and final self-review. Source/procedure/result review
is **SELF-REVIEW**, not independent acceptance. Relevant237 pins matched before
work; frozen236/237 evidence was preserved. Executed qualifier/benchmark helper
bytes are retained before later annotation-only fixes. Profile producer and its
protocol remain unchanged after execution.

Exact profile launch:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/minigrid-copy-profile-20261027/execute.py
```

Exact focused child:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/minigrid-copy-profile-20261027/qualify.py win-final
```

Linux uses the same child/selectors through cached WSL Python, with the exact
environment/path in `linux-final-parent.json`. Exact benchmark launch:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/minigrid-copy-profile-20261027/supervise.py benchmark .venv/Scripts/python.exe -B artifacts/runs/minigrid-copy-profile-20261027/bench.py
```

Initial Windows tests used the existing237 supervisor; its new receipt was
initially written there, then moved unchanged to this stage. A local40-second
supervisor was copied for subsequent calls. No frozen237 source/result changed.
One failed patch-context match was corrected before executing the benchmark.

Eight-file final Ruff format/check pass. Core and new tests pass ordinary mypy
for default and Linux targets. Six helpers pass scoped mypy with
`--explicit-package-bases --follow-imports=silent`; helper bodies are checked.
Initial local annotation/Stats declaration errors and later heterogeneous-dict
annotation errors were retained and repaired without ignores. Three inherited
`return_policy_evaluation.py:215` diagnostics reached through the legacy fixture
CLI import are explicitly outside the helper gate, unchanged and not claimed
fixed. Exact commands/outcomes are in all four `static-*.json` receipts.

All work remained within amended finite engineering limits:30s body/45s parent/
512MiB per platform, no scientific allowance. Original two-call profile costs,
software and benchmark are cumulative. No production/native profile rerun,
full math/model cohort, whole suite, clean clone/current wheel/install, independent
review, native task comparison, UI/GPU/remote CI or release qualification ran.
No dependencies, agents, Git mutation or final/probe access. Historical plans/log
remain frozen; live issue/tracker receipts are the handoff.

Final measured Windows body including the pure projection gate is4.66913659987s;
Linux4.226154007s. Windows cumulative13 empty/scripted owners/676 snapshots/
1017 native snapshot copies/7399 attempted record validations/26 scripted
predictions26 rows/14 fake environments/24 fake steps/14 closes/245 original
boundary callbacks; Linux software counts are stated above. Constructor/validator
calls include malformed-record refusals and the benchmark, not optimizer fits.
All-attempt statics30.51302890014s/60; mtime-selected cache113,131,520B/160MiB
(not growth). Final source review confirms all fit/predict arithmetic, class
fields and pre-coefficient admission checks unchanged. Current source/helper
pins are recorded in `review-final.json`.

## Full goal and next action

This measured performance repair is useful progress, not completion of the
learner or scientific goal. Original repeated-update/R2.1/matched BP/PC/neutral/
active evidence and useful conventional learning/deployment are reused.
All failed or inconclusive results, especially236, remain intact. Full goal and
R2/R4/R5/R7/P7/P9/science/release parents remain open.

Return to useful adaptation rather than another profile/capture increment.
Source inspection shows the ordinary fixed-reference API/CLI passes ready
feedback to `run_reference_window` but does not expose its already-implemented
`acknowledge_correct` option. Existing long-phase evidence234 shows FIFO learning
lagging into the returned context; proven-correct packets need not consume
optimizer work. Next precise implementation: add an explicit, default-off
correct-acknowledgement option through the existing fixed-reference API/CLI,
requiring ready feedback and preserving the actual fresh proof, FIFO receipts,
original clocks/budgets and all default baseline behavior. First reconcile any
existing equivalent route; add focused causal/cost/failure tests. No new policy,
runner, native comparison, baseline tuning or replay of236 is authorized by
this closeout, and the option itself cannot establish native adaptation benefit.

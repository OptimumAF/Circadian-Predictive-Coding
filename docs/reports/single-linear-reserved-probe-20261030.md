# #355 — ordinary single-head isolated fresh-task probe

R2.1-LIN.SHARED.F; parents #18/#21/#23/#24. Full goal remains active.
Physical checkout `D:\Dev\Repositories\Circadian`, HEAD
`d9119e80926a139a2f38460564d7eab1b6d2a2b0` plus preserved dirty local work.
Evidence stage: `artifacts/runs/single-linear-reserved-probe-20261030/`.

The ordinary command now admits an owned single-head LS checkpoint12 into its
existing isolated fresh-F16/old-A-return16 probe. It constructs a detached
minimum-change child, retains the fixed original cohorts and frozen readout,
and protects source authority and preceding paid evidence before further work.
The source TRAIN operation remains independently explicit. Default source cold
updates and the existing separate-head path remain supported. There is no new
optimizer, orchestration framework, label conversion, source budget renewal,
publication, or single-head replay+probe composition.

This is software delivery, not demonstrated fresh-task acquisition. No genuine
source-trained single-head probe study ran. #354's mixed replay outcome and its
unverified cache/review-time gates remain OPEN/PARTIAL-SPENT. #352's missing-cost
gate and #348/#344/all previous negative or spent results remain unchanged.

## Actual integration and scope correction

The handoff suggested extending `ticket_linear_probe.bind_source`, but inspection
showed that the ordinary CLI uses `ticket_native_probe.run_owned_probe` directly.
The composed binder and `run_arm` remain separate-head-only. Broadening those
would add unrelated behavior; the scope correction was posted before edits in
#355 comment 6539571362, preserving the intended ordinary isolated behavior.

- `scripts/recurring_ticket_workflow_example.py`: narrow shared-head reserved
  API/CLI admission, preserve the head choice in forwarding; import local module
  names separately from existing `Any` locals for scoped typing.
- `scripts/ticket_native_probe.py`: exact D9/K6 shared snapshot/source definition,
  no-replay admission and a detached minimum-change child type. Revalidate source
  mode/role headers at each private boundary; reuse original clocks, budgets,
  checkpoint identity, paid prefix guards and generic fresh/return runner.
- `scripts/ticket_linear_probe.py`: explicit ordinary single-head child type;
  source callbacks are not inherited. Existing composed two-head binder unchanged.
- `tests/test_single_linear_reserved_probe.py`: new admission, CLI, isolation,
  source mutation, prior evidence, failure charges, actual frozen readout and
  ordinary scripted runtime handoff checks. The old ordinary file loses one
  obsolete shared-head refusal and gains a NumPy variable annotation.
- Existing usage/README updated. No core numerical, runtime, summary or baseline
  implementation changed. Exact inherited preimages are retained in the stage.

## Executed checks and permanent failed attempt

All commands use physical D: cwd and process-local `TEMP`/`TMP`
`D:\Dev\_Temp`; WSL also uses `TMPDIR=/mnt/d/Dev/_Temp`. Existing Linux Python
is `/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python`.
No installation/download or global setting change. Bytecode disabled; pytest
plugin auto-discovery/cache disabled and basetemp directed to D:.

1. Initial six-file Ruff/format passed. Linux-target mypy found five errors:
   two existing module-import/local-name redefinitions, one old NumPy variable
   annotation, and two new test annotations/return expressions. Initial receipt
   `statics-initial.json` / log preserved (2.7344887 s). All repaired without
   changing numerical operations. Final scoped Ruff/format/mypy pass,
   0.8053889 s, `statics-final.json`.
2. Windows initial command:
   `.venv/Scripts/python.exe -B artifacts/runs/single-linear-reserved-probe-20261030/check.py windows D:\Dev\_Temp\circadian-single-probe-win-20261030`.
   Both test files ran. Their 87 then-existing nodes give **86 passes and one
   failed legacy full-app fixture**; the initial extra quiet output omits that
   pass total, so this count is reconciled against the later same-node collection
   after adding exactly two runtime cases. Full initial log/receipt retained.
3. That last legacy test invokes a real F16/return16 probe, exceeding the new
   tiny-fit software allowance. The observer charged its ninth LS admission,
   refused it before delegation at the shared cap8, and the fixture correctly
   reported incomplete. Cohort exit1; 2.570980600 s / 79,060,992 B sampled RSS.
   Before-call counters: nine solve admissions (one refused), 22 charged fit
   rows; eight delegated solves by wrapper control flow. Training products27 /
   75 rows include preparation of the refused solve. Existing initial receipt
   uses the older key `solves`; do not mislabel nine as nine delegated solves.
   Actual tiny shared and separate-head updates passed before that failure.
   **The eight-fit allowance is spent.** No extra fitting or retrospective cap
   increase. This failed cohort remains failed, not a clean full-suite pass.
4. Prospective selection correction recorded in #355 comment 6539614411:
   retain the failed full numerical test, reuse its earlier accepted separate-head
   evidence, and exclude that full fixture plus both successful tiny-fit
   parameters from remaining Linux checks. Failing-fit parameters remain.
   No mathematical/control refit or scientific protocol changed.
5. Two new Windows ordinary runtime cases (cold/minimum-change source), same
   wrapper with selector
   `tests/test_single_linear_reserved_probe.py::test_shared_runtime_reaches_private_fresh_return_without_spending_source_authority`,
   invocation `windows-runtime`, D: basetemp suffix `win-runtime-20261030`:
   **2 PASS**, 1.798523900 s / 72,912,896 B. Zero actual solves/affines; 88 scripted
   source/private callbacks / 432 rows. Source12/72 receipts and clocks conserved;
   private32/144 callbacks and source/actor/RNG flags preserved. This is transport,
   not learning evidence.
6. Linux command uses `wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env
   TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp
   PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
   OPENBLAS_NUM_THREADS=1 <python> -B
   artifacts/runs/single-linear-reserved-probe-20261030/check.py linux
   /mnt/d/Dev/_Temp/circadian-single-probe-linux-20261030
   tests/test_single_linear_reserved_probe.py tests/test_ordinary_linear_reserved_probe.py`
   followed by the three exact `--deselect` node IDs recorded below. Same live
   process handle polled to terminal exit0; **86 PASS, 3 deselected**, 20.812593285 s /
   73,846,784 B. New shared file39, old file47. Zero actual solves/training
   products; 72 actual fitted affine calls/rows (shared + two-head frozen
   readouts). Shared scripted counter120 calls576 rows includes two actual
   ordinary runtime handoffs; legacy scripted callbacks are separate below.

Exact Linux deselections:

```text
tests/test_ordinary_linear_reserved_probe.py::test_real_existing_fixed_probe_and_return_app_from_literal_source_fixture
tests/test_ordinary_linear_reserved_probe.py::test_one_real_tiny_private_update_or_failure_keeps_charges_and_source_detached[False]
tests/test_single_linear_reserved_probe.py::test_shared_tiny_fit_or_failure_preserves_source_and_permanent_attempt[False]
```

Windows qualifies 40 new shared cases through initial38 plus affected2, and
48 old cases, alongside the failed full fixture. Linux executes39 new +47 old;
its successful-fit numerical parameter is explicitly unexecuted. Do not claim
all 89 nodes pass freshly on both platforms or a successful full32-fit fixture.
No new production edits followed the passing Windows runtime/Linux checks.

## Costs, limits and qualifications

Prospective #355 start comment froze at most two focused invocations/platform
plus one <=20-node repair/platform, each60 s/256 MiB sampled RSS, shared8 tiny
fits24 rows/32 scheduled training products96 rows, scripted512callbacks3072rows,
typed256calls1024rows, scoped statics90 s, stage4 MiB excluding separate task
cache64 MiB. No renewal of these limits after the failed Windows cohort.

Totals from all three executed receipts: nine before-call solve admissions,
eight delegated /22 charged rows; 27 scheduled training products75 rows;
152 affine ports180 rows, including144 actual typed-logit calls. Initial other
eight affine calls/36 rows came from the generic scoring path; no total FLOP or
allocation accounting is inferred. RSS is sampled process memory, not a hard
process-tree limit. The refused ninth admission remains a failed budget gate.

New shared scripted counters observed32/144 initial,88/432 Windows runtime,
120/576 Linux. The older file did not expose a `SCRIPTED` meter: conservatively
debit128callbacks640rows for each of its two executions, plus one unmetered
shared failed callback/one row per full execution. Combined conservative debit
**498 callbacks/2434 rows**, within original512/3072; this is a reservation bound,
not a fabricated exact measurement. Clock/RSS includes data encoding and teacher
helpers, whose complete operation counts were not metered; they are not zero.
All attempts/failures remain charged. No neural/Torch import, SGD/latent work,
physics, final data, promotion, search, baseline cohort or new scientific probe.

Dedicated caches start0 at `stage/cache/`; final scoped statics measured
10,613,807 B, below64 MiB. No default shared-cache change was requested. Stage
source/log/report artifacts use a separate4-MiB allowance; final size is in
handoff. Previous #354 cache/timing gates remain unresolved, not repaired here.

Full repository suite, clean clone, installed wheel/build, unchanged mathematical
and matched baseline cohorts, native scientific adaptation, GPU/physics/browser,
remote CI and release checks skipped. Pre-existing full-app acceptance is reused
only for unchanged two-head assumptions, not as current-tree full numerical
qualification. No dependency/global/delete/Git publication operations.

## Review and next useful outcome

Lead self-review covers combined actual diffs, strict role/source admission,
before-operation protections, charged failure, actual check selection and costs.
Independent actual-source/focused-evidence review **ACCEPTS the scoped software**
(`stage/REVIEW.md`). Four read-only commands all exited0, total1.1485679 s;
all four staged diffs exactly matched the actual current files. No imports,
tests/models/network/children or numerical replication. Initial86-pass count
remains explicitly a lead reconciliation, the scripted498/2434 debit a conservative
reservation rather than an independent exact meter, and teacher counts unknown.
The failed initial numerical cohort stays FAILED/SPENT. This acceptance does not
qualify full shared-head adaptation, a clean full cohort, broad resource/scientific
or release claims. Lead self-review is separate; finite review exception ended.

The next substantial validation, after this software is accepted, is one small
prospectively frozen **ordinary single-head fresh-F/old-A-return** observation
through this existing public command. Preserve original fixed cohorts/checkpoints,
source12/72/30 s and independent F16/return16 authorities, actual teacher/typed/
fit ports, all losses/successes and diagnostic costs. Source and probe TRAIN
choices must be explicit; no replay composition or extra dose. Preserve #352/#354
as qualified historical observations, not compute-matched controls or retries.
This new fresh-task outcome differs from their old-context-only sequences.
Do not execute from the exhausted software fitting allowance: freeze/review the
small new protocol first, with dedicated D: caches and an exclusive attempt.
Full acquisition/retention/recovery, R2/R4/R5/R7/science/release/goal remain open.

Canonical #355 scoped software acceptance: https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/355#issuecomment-6539691223. Live readback CLOSED/four software criteria checked; failed numerical cohort retained. Parent receipts18/6539693573,21/6539693730,23/6539693877,24/6539694013. Planned #356 only, no new study. Full goal remains active.

# Native categorical frozen-readout software qualification

R2.1-TW.FR / [#317](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/317),
following accepted #316. Physical checkout `D:/Dev/Repositories/Circadian`,
HEAD `d9119e80926a139a2f38460564d7eab1b6d2a2b0` plus inherited dirty work.
All inherited source, negative/SPENT results and artifacts remain unchanged.

Delivered `src/adapters/torch_categorical_readout_fit.py` and its focused tests.
The actual-owned BP prior feature read and disposable categorical TRAIN fit now
run. Existing native head math, categorical solver, optimizer, snapshot schema,
runner flags and defaults are unchanged. The adapter admits all TRAIN roles
before labels, uses actual full owner/optimizer validation and an unexposed
baseline, and refuses changes without restoring them. See
[usage and interpretation](../torch-categorical-readout-fit.md).

This is software qualification on small D2/H3/K3 fixtures. A planted representation
supports all three known TRAIN labels at rank3 and squared residual below1e-24,
with12 affine coefficients. The existing native update continues identically
after the diagnostic. **The real D9/H3/K6 source12 feasibility study is unexecuted.**
No useful task, generalization, causal optimization/capacity, matched compute,
plasticity or CPC advantage claim follows. Parent #21/#18 and the full scientific
and release objective remain open. Existing #278 class balance stays negative,
closed/SPENT; no rerun, re-read or renewed scientific budget.

## Executed checks

Windows Python3.14.7/NumPy2.4.6 and existing Linux Python3.12.3/NumPy1.26.4;
both Torch2.14.0+cpu. Set `TEMP=TMP=D:/Dev/_Temp` in the PowerShell parent.
One process, CPU thread environment1. Let
`S=artifacts/runs/torch-categorical-readout-fit-20261030` and
`T=tests/test_torch_categorical_readout_fit.py` in the commands below.

```text
.venv/Scripts/python.exe -B S/check.py S/windows-first-work.json T --basetemp=D:/Dev/_Temp/circadian-fr317-win-first
.venv/Scripts/python.exe -B S/check.py S/windows-second-work.json T --basetemp=D:/Dev/_Temp/circadian-fr317-win-second
.venv/Scripts/python.exe -B S/check.py S/windows-role-work.json T -k all_roles --basetemp=D:/Dev/_Temp/circadian-fr317-win-role
.venv/Scripts/python.exe -B S/check.py S/windows-guard-work.json T -k "working_snapshot or changes_without or each_boundary" --basetemp=D:/Dev/_Temp/circadian-fr317-win-guard
```

Replace literal `S`/`T` with the paths defined above. Linux prefix:

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B
```

Same `S/check.py T`, destination `S/linux-first-work.json`, `linux-role-work.json`
or `linux-guard-work.json`; selections respectively all, `-k all_roles`, or
`-k "working_snapshot or changes_without or each_boundary"`. Each Linux command
uses `--capture=sys` and
`--basetemp=/mnt/d/Dev/_Temp/circadian-fr317-linux-{first,role,guard}`.
Every receipt stores exact argv. Linux yielded command observation was resumed
or checked from its terminal status0 receipt; no model process was restarted.

| Invocation | Outcome | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
| Windows first |0 pass /1 fail|4.326373300|327970816|
| Windows second |18 pass|4.035072200|327225344|
| Linux first |18 pass|10.472912022|344514560|
| Windows role |4 pass /0 models|2.476808800|242057216|
| Linux role |4 pass /0 models|9.247173104|262385664|
| Windows final guard |8 pass|3.802194600|326930432|
| Linux final guard |8 pass|9.986768911|342585344|

All60 successful test executions, one initial failure, zero skips. There are
**19 distinct qualified checks per platform**: initial18, then eight affected
guard checks containing seven repeats and one new simultaneous-mutation case.
The four role cases also repeated after the test-only type cast. No full-suite
post-guard rerun is claimed; the unchanged numerical/role/continuation evidence
is reused, while the added baseline guard is covered by final affected checks.

All attempts total44.347302937s, maximum344514560B. Paid shared work:

- 58 native component constructions;116 named seeded initialization draws,
  870 values. No diagnostic constructs a model or optimizer.
- 9 actual BP backward/SGD calls and27 training-row exposures.
- 37 observed hidden-helper calls /111 attempted rows; injected failures and
  mutations are included. No gradient or latent inference in a diagnostic.
- 16 actual NumPy LAPACK least-squares calls /48 fit-row exposures, plus two
  literal solver failures injected before LAPACK (one per initial full platform).
- 8 fitted affine readouts /24 rows;12 native head forwards /36 rows.
- Two intentional global NumPy random mutations per platform across initial
  and final guard tests are refused, never restored; each is one scalar draw,
  additional to the named seeded initialization count above.
- No autograd.grad, SVD calls, PC latent work, simulator interactions, replay,
  genuine benchmark/probe/final data, GPU or remote model execution.

Wall/RSS include validation, conversion, batch encoding and imports; individual
tensor/BLAS operation counts are not a full matched-compute ledger. The ordinary
local test observer is adapted from accepted315, wraps known work ports before
tests, checks shared prior receipts and samples RSS before Torch import. It is
local qualification tooling, not a new production runner or admission system.

## Failures, scope amendments and review

First Windows run failed because the reused tensor/dataclass equality helper
does not compare NumPy arrays. The input guard now checks exact array type,
dtype, shape and array equality. Receipt retained; cost is included above.

Initial statics: Ruff check/format passed; default/Linux mypy failed the
deliberately invalid-owner test's `object` argument. An explicit type cast fixes
that test only. `statics-first.txt` preserves the diagnostic; individual initial
static durations were not measured. The enclosing shell7.8506202s also read a
receipt and is not presented as pure static cost. Final pre-review static batch
four exit0 /7.1434111s. Final guarded batch four exit0 /7.5165949s:

```text
.venv/Scripts/python.exe -B -m ruff check src/adapters/torch_categorical_readout_fit.py tests/test_torch_categorical_readout_fit.py
.venv/Scripts/python.exe -B -m ruff format --check src/adapters/torch_categorical_readout_fit.py tests/test_torch_categorical_readout_fit.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent src/adapters/torch_categorical_readout_fit.py tests/test_torch_categorical_readout_fit.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux src/adapters/torch_categorical_readout_fit.py tests/test_torch_categorical_readout_fit.py
```

Independent review found the working snapshot was itself the comparison
baseline, allowing a faulty helper to change it and the owner identically.
A separate private baseline now checks both. The literal regression refuses
before fitting and retains the foreign mutation. Final eight affected checks
pass on each platform. Reviewer accepted the actual fixed source, tests and
receipts in exactly four read-only commands /0.8521270s, all exit0, no writes,
imports, models, tests, network or children; exception ended. Lead also reviewed
source, procedure and costs. Self-review is not independent review.

Before any model invocation, static fixture inspection showed32 component
constructions would not cover both platforms' isolated fault owners. Prospectively
amended to64 /128 named draws4096 values (issue comment6523983427); no spent
experiment was renewed. After the review blocker, prospectively amended invocation
allowance to4 Windows /3 Linux (6524053538), without increasing numerical caps.
All seven invocations satisfy120s/768MiB each, one CPU process; shared caps remain
16updates128rows/32solves256rows/96hidden768rows/64affine512rows. Local stage cap
4MiB and D temporary cap16MiB; preserve failures and evidence. One initial tool
orchestration syntax error occurred before any nested action, then was corrected.

Skipped full repository suites, clean clone, installed wheel, native source12,
other families, simulator, comparative cohorts and release checks: this bounded
BP diagnostic changes no deployed behavior, solver or default protocol. Software
acceptance does not confer a numerical experiment allowance.

Next: freeze and review ONE small Linux BP source12/72 plus one known-TRAIN12
frozen-readout solve protocol, using the existing chronological runner and an
actual retained owner. Account separately for reconstructing the live source
because saved scalar fixtures contain no owned hidden state. No historical JSON
parameter loader, baseline sweep, class-weight/seed/dose tuning, probe/final
labels or promotion. Successful fit means only frozen-feature known-TRAIN
feasibility; failure does not disprove other CE optima.


Closeout storage: stage12files19,954B; no matching pytest temporary roots or
retained files under the declared D temporary names (0logical bytes). Scoped
gitdiff whitespace and all four new-file whitespace checks passed, with
LF/CRLF advisories only. Two malformed documentation patch attempts were
rejected without changing files; direct append completed the handoff. No
install/download/delete/commit/push or global-cache change.

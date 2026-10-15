# Finite growth/refusal endurance

```powershell
python -m scripts.run_coupled_growth_endurance --describe
python -m scripts.run_coupled_growth_endurance
```

Description does no operational work. Actual execution runs the frozen local
engineering protocol once across PC, neutral CPC and active CPC. There are no
seed, count or threshold knobs. It reuses the existing coupled growth ceiling,
native learner, actor/shadow runtime, serving cache and original attempt budget.

## Fixed workload

Each source starts at width2 with two raw updates, seed23/mean-field/two hidden
layers/gamma1.5/precisions(1,1), fixed train rows and external rates .01/2/.05.
The candidate's original consolidation quota is128 and split width ceiling8.
The first two attempts grow2->4->8. The next126 attempts refuse proposed16 before
expanded allocation; each failed ID remains consumed. A duplicate-ID probe after
the first width refusal and an exhausted-ID probe at the end refuse callbacks.
That is130 requests and128 callbacks per family, with two committed receipts.

Each request includes64 cached actor reads. One occurs inside each admitted
callback, including refusals, and the rest follow the result. The actor remains
at width2 with its original complete cache/version/generation/configuration and
metadata. Serving uses fixed logical tick2 and TTL10. After the stream, one arrived
delayed-label update uses the candidate's original single training allowance,
then another cached read verifies actor availability. Total:24,966 serving calls,
24,963 cache hits, three cold native predictions, nine raw updates/36 energy
evaluations and six actual split kernels. No promotion or guard scoring occurs.

Full candidate raw state, precision/scaling configuration, CPC history and
structural tags are checked throughout refusals. Attempt metadata is checked
separately; it changes even when native state does not. Continued CPC wake count
is2->3; PC and neutral raw parameters remain exact at matching phases.

## Bounded retained evidence

Why stream summaries: retaining a full native payload for every failed request
would make the diagnostic history itself grow with the workload.

The report keeps scalar counters for every request/read, at most eight block
records and six scalar/ID request samples per family. Their aggregate canonical
JSON must fit16KiB. It separately retains complete initial/capped/final native
states and final original cursor, each cursor bounded by16KiB. There is no native
payload or full cursor per request and no list of every serving result.

Actual logical array bytes, cursor/summary sizes, cache entries and process RSS
are reported. RSS is sampled at boundaries under a512MiB ceiling; body wall budget
is240seconds and CLI output2MiB. These are observed bounds for a finite workload,
not continuous preemption, hours of uptime or complete long-run/recovery acceptance.
An incomplete run remains incomplete; thresholds or seeds are not changed to pass.

## Structure and validation

```text
src/adapters/coupled_growth_endurance.py  fixed native stream + bounded summaries
scripts/run_coupled_growth_endurance.py  description/actual bounded JSON CLI
tests/test_coupled_growth_endurance.py   independent pure controls + tiny prefix
```

No core/app policy, math, dependency or environment configuration change. The
private test engine accepts a small explicitly reported engineering control;
public execution always uses128attempts/64reads. Tests run only a4attempt/2read
prefix, and the full endurance execution is separately budgeted and retained.

```powershell
python -m pytest -q tests/test_coupled_growth_endurance.py
$enduranceFiles = @('src/adapters/coupled_growth_endurance.py', 'scripts/run_coupled_growth_endurance.py', 'tests/test_coupled_growth_endurance.py')
python -m ruff check @enduranceFiles
python -m ruff format --check @enduranceFiles
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @enduranceFiles
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform linux @enduranceFiles
```

See ADR0280 and `artifacts/runs/coupled-growth-endurance-20261015/` for the frozen
scope, commands, actual results and independent review. Extend only through a new
finite protocol with original quotas and bounded evidence; retain negative or
incomplete runs. Scientific benefit and full managed recovery remain separate.

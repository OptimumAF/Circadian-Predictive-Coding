# Bounded rejected consolidations

Run a fixed engineering stream through native PC, neutral CPC and active CPC:

```powershell
python -m scripts.run_coupled_rejected_consolidations --describe
python -m scripts.run_coupled_rejected_consolidations
```

Description performs no operational IO, model, process, thread or memory sampling
work. Actual execution uses seed23, D2/H(3,2), precisions(1,1), CPU1 and fixed
training rows/rates. Each model gets two warm updates; one arrived delayed-label
training example updates the private candidate once. Nine training calls/36 energy
evaluations total. Actor A retains its full warm state while candidate B retains
its one-update state, including complete CPC policy, chemistry, gates and count.

## Repeated failure and serving behavior

The runtime has an original lifetime consolidation limit of eight. Each unique
`reject-00` through `reject-07` operation mutates its detached parameter copy and
CPC chemistry where present, serves the already primed actor cache while holding
the candidate gate, then raises a fixed injected refusal. The candidate model is
never restored from the failed copy, and no consolidation receipt is committed.

After the first failure, a duplicate-ID probe must refuse before its callback.
After all eight unique failures, a ninth unique ID must refuse at the original
attempt quota before its callback. Ten explicit requests and eight callbacks per
family are finite; no automatic retry loop is involved. Every failed ID remains
consumed. Candidate attempt counts and cursor revision remain visible even though
the native model state is unchanged.

The actor's full serving snapshot, cache, configuration, version, generation,
metadata and logical cache tick stay unchanged. The cache has one entry under a
two-entry configuration, and serving at the same logical tick keeps cache hits
valid. Only the initial cold call per family invokes native prediction; calls
during refusals and admission probes use the existing cache.

Why this: rejected transformations must consume bounded attempt history without
blocking actor availability or accidentally publishing a partially changed copy.
This composes the existing runtime and cache contracts; no learning equation,
promotion guard, metric, seed selection or baseline changes are needed.

## Observations and limits

The report contains complete actor/candidate states before and after, every
request/refusal/cursor and callback count, detached mutation evidence, serving
results, actual cursor JSON bytes and native array bytes. Failed transforms keep
native array sizes fixed; the cursor must stay within16KiB and retain eight IDs
with zero receipts. The complete JSON report is capped at2MiB.

Existing process RSS sampling runs at explicit operation boundaries with a512MiB
ceiling and a60-second execution bound. Readings are observations of this process,
not allocation attribution to a model, continuous hard preemption or a long-run
memory guarantee. Unsupported or unavailable RSS observations fail explicitly.
The fixed stream does not establish the broader long-run/recovery gate.

## Structure and checks

```text
src/adapters/coupled_rejected_consolidations.py  native/runtime/cache composition
scripts/run_coupled_rejected_consolidations.py  bounded JSON CLI
tests/test_coupled_rejected_consolidations.py   independent behavior/integration
```

Core/app runtime, serving, budget, model and cursor modules remain unchanged.
No dependency or environment configuration added; CPU thread variables follow
the existing local defaults. Extend with explicit finite streams and original
quotas; never reset failed IDs or discard CPC state to admit more attempts.

```powershell
python -m pytest -q tests/test_coupled_rejected_consolidations.py
$refusalFiles = @('src/adapters/coupled_rejected_consolidations.py', 'scripts/run_coupled_rejected_consolidations.py', 'tests/test_coupled_rejected_consolidations.py')
python -m ruff check @refusalFiles
python -m ruff format --check @refusalFiles
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @refusalFiles
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform linux @refusalFiles
```

Actual model runs need a prospective local budget. Current evidence and commands
are in `artifacts/runs/coupled-rejected-consolidations-20261015/`; see scope,
results and independent reviews. Full managed recovery, actor promotion,
coordinator loss, long runs and scientific benefit remain separate requirements.

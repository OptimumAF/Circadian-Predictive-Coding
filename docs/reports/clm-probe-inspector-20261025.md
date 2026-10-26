# Detached CLM probe inspector: implemented, browser gate open

[Issue154](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/154),
R4.8-CLM.I, **OPEN: interactive UI criterion remains unchecked**. Implemented
bounded saved-review projection through the existing offline CLI/loader/table
renderer. Final64pure checks pass on Windows/Linux; lint/format/default and
Linux-target mypy and extracted JavaScript syntax pass. Both actual detached
reviews export and their embedded projections/rendered bytes match the final
combined code exactly. Browser opening of a local file URL was rejected by
policy; no UI interactions/screenshots or visual acceptance are claimed.

## Functionality and scope

See [API, file tree and usage](../clm-probe-inspector.md). Recorded F TRAIN/suffix
and old TRAIN cases/metrics/margins, source/probe work/clock records, parameter
scope, preparation/review costs and missing-native observations use the existing
source-cell tables/dialog. No metrics are recomputed, no labels enter a model,
and no model/encoder/scoring/training/physics/game/final/replay work runs.
Inputs are detached reviews, not raw native scientific artifacts; source claims
remain unauthenticated. Missing/null/false/zero stay distinct and negative cases
are retained. The original scientific conclusions and all budgets remain frozen.

The new app module has no IO; the existing infra file adapter owns bounded JSON
and exclusive output, and existing HTML owns table/source presentation. The
dispatcher recognizes an explicit review-field signature, then strictly validates
its typed fields and workflow32case/catalog/role bounds. Other routes retain their
dispatch and templates. Native loss, learners, update/metric math and CLI options
are unchanged. No new dependency/framework/renderer.

## Executed checks and failures

Initial Windows run:26PASS/1FAIL (27executions). The fresh dependency-import
assertion exposed pre-existing NumPy imports through the generic inspector:
EvaluationIdentity and legal-probability helpers imported numerical dependencies
before any report type was known. The first import move alone did not resolve
both paths. Both imports now occur in the same legacy functions immediately before
their existing uses; validation/math is unchanged. New CLM projection and CLI
import no NumPy/Torch/Transformers in fresh child processes. Five initial/test/
diagnostic processes loaded NumPy before the correction; this failed gate is not
retroactively accepted or refunded. No model work occurred. Initial peak RSS was
not sampled and remains unknown; initial tool elapsed2.0297119s/XML1.283s.

| Metadata run | Outcome | Measured body seconds | Sampled peak RSS bytes |
| --- | --- | ---: | ---: |
| windows-2 |64PASS/0fail/error/skip |1.976793200 |58,359,808 |
| windows-3 final |64PASS/0fail/error/skip |1.901178300 |60,174,336 |
| linux-1 |64PASS/0fail/error/skip |8.332040366 |41,684,992 |
| linux-2 final |64PASS/0fail/error/skip |8.397622570 |41,508,864 |

Windows has155executions/64distinct cases including the original failure;
Linux128executions/64distinct. Successful Windows body3.877971500s plus the
separately retained failed first run; Linux body16.729662936s. Final measured
runs remain below90s cumulative/1GiBRSS and120sparent allowances. First RSS is
unknown, not asserted compliant. Windows's three-run fixture allowance is fully
consumed; no more Windows test runs under this allowance. No cap revisions.
RSS samples cover the test parent; temporary CLI-child peaks were not sampled.
Final stage23files/360,519B and scoped caches96,470,317B outside the stage remain
below8MiB/128MiB. Console/XML failure output is retained; no large logs or models.

New27cases cover both review routes, every pointer and detached input preservation,
negative/unknown cases, role/catalog/probability/metric/counter/scope refusals,
inert hostile text, exclusive existing-CLI export and model-free imports. Affected
ticket pure fixtures and file-admission/export/CLI regressions also pass. The
one existing file test reading an original saved paired artifact is deselected;
other old scientific-file tests are not selected. No spent native actual/
execution/ledger is opened. Full unrelated model/game/math suites are reused,
not rerun; legacy numerical helper bodies were not changed.

Ruff/format pass. Both initial mypy targets found one already-validated fit-row
list lacking static narrowing; a `typing.cast` fixes it without changing values.
Final both five-file type targets pass. Measured statics4.9596306s including
failed checks, plus subsecond export/JavaScript review, within120s. Actual Linux
Ruff/mypy unavailable/noinstall; Linux-target checks run on Windows. Full clean
clone/install/wheel/GPU/CI/release/independent review unrun. Lead source/procedure/
results review is **self-review only**, with the UI limitation left open.

## Actual exports and browser limitation

The existing CLI reads each accepted detached input once and creates:

| File in artifacts/runs/clm-probe-inspector-20261025/ | Bytes | Recorded native status | Sections |
| --- | ---: | --- | ---: |
| workflow151.html |200,913 |true |16 |
| workflow153.html |33,070 |null/unknown |4 |

151retains its partial F acquisition and old-task interference (F0/2→1/2,
oldTRAIN12/12→10/12, source unchanged).153retains failed preflight/no native
result. Neither is a new scientific observation. A zero-model final review reads
the generated HTML's embedded data, verifies every projection and rendered byte
against the final source, and syntax-checks its extracted script with Node.
It never reopens either spent original scientific input nor rereads the original
detached export inputs. No source data were modified.

The Playwright CLI was not installed; npx exists but no package was installed or
downloaded. Available browser controls rejected:
`cua.createBrowserTab('iab', 'file:///.../workflow151.html', {visible:false})`.
The policy permits only HTTP(S) and explicitly forbids workaround/alternate
browser surfaces. No loopback server, alternate browser, raw browser commands
or other bypass was attempted. Zero browser DOM assertions/screenshots completed;
opening was one failed attempt, not interactive verification. Static Node syntax
checking does not satisfy the browser criterion and is not presented as doing so.

## Reproduction and pending action

```powershell
# Literal/affected metadata checks; the harness deselects one old saved-file test.
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH=(Get-Location).Path
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.venv/Scripts/python.exe -B artifacts/runs/clm-probe-inspector-20261025/checks.py windows-2
.venv/Scripts/python.exe -B artifacts/runs/clm-probe-inspector-20261025/checks.py windows-3
# Cached WSL runtime, same checks.py: linux-1 and linux-2; full command in parent receipts.
.venv/Scripts/python.exe -m ruff format --check src/app/clm_probe_inspector_projection.py src/app/key_door_inspector.py src/infra/key_door_inspector_files.py scripts/build_key_door_inspector.py tests/test_clm_probe_inspector.py artifacts/runs/clm-probe-inspector-20261025/checks.py
.venv/Scripts/python.exe -m ruff check src/app/clm_probe_inspector_projection.py src/app/key_door_inspector.py src/infra/key_door_inspector_files.py scripts/build_key_door_inspector.py tests/test_clm_probe_inspector.py artifacts/runs/clm-probe-inspector-20261025/checks.py
.venv/Scripts/python.exe -m mypy --cache-dir .mypy_cache/clm-workflow-probe-20261025-win --explicit-package-bases --follow-imports=silent src/app/clm_probe_inspector_projection.py src/app/key_door_inspector.py src/infra/key_door_inspector_files.py scripts/build_key_door_inspector.py tests/test_clm_probe_inspector.py
.venv/Scripts/python.exe -m mypy --platform linux --cache-dir .mypy_cache/clm-workflow-probe-20261025-linux --explicit-package-bases --follow-imports=silent src/app/clm_probe_inspector_projection.py src/app/key_door_inspector.py src/infra/key_door_inspector_files.py scripts/build_key_door_inspector.py tests/test_clm_probe_inspector.py
.venv/Scripts/python.exe -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/clm-workflow-fresh-native-20261025/saved-review.json --output artifacts/runs/clm-probe-inspector-20261025/workflow151.html
.venv/Scripts/python.exe -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/clm-workflow-state-only-native-20261025/saved-review.json --output artifacts/runs/clm-probe-inspector-20261025/workflow153.html
node --check artifacts/runs/clm-probe-inspector-20261025/inspector-script.js
```

These exclusive output/run names are already used: do not repeat them over their
receipts or reread spent scientific originals. All failures, snapshots, XML,
sampled work/static receipts and two HTML outputs remain local. Dirty unrelated
changes and historical plans/log are preserved; live status belongs to issues.

**One precise next action:** manually open the two generated HTML files and
verify marker/old-case tables, unknown native header and value→JSON-pointer source
dialog, then record the UI result on154. Browser criterion stays unchecked until
that evidence exists; the implemented projection is not repository-wide readiness.
Original153scientific gates remain open/failed-spent, not retried. Full reliable
learning/joint retention/context return/R2/R5.1/R5.3/G5/R7/release remain open.

# Shared-head probe inspection — 2026-10-31

R4.7-LIN.SHARED / [#361](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/361).
The existing offline inspector now accepts explicitly requested shared-head
minimum-change LS reports and displays their auxiliary fitting costs honestly.
It no longer labels zero direct private `train_batch` counters as a private total.
No learner, algorithm, source report, native experiment or scientific result changed.

Previous turn was PROGRESS: #360's one native comparison and independent acceptance
were reused. HEAD remains d9119e80926a139a2f38460564d7eab1b6d2a2b0 with inherited
dirty/untracked work preserved. Current goal remains ACTIVE. Current AGENTS and
AGENT_PROMPT supersede the missing historical downloaded coordination prompt.

Changed production files: `src/app/key_door_inspector.py` adds one shared-family
dispatch member; `src/app/reserved_probe_inspector_projection.py` explicitly
classifies both linear families, validates optional recipe/control/source/cost
metadata, and adds saved auxiliary counter/receipt/source-budget cells;
`src/infra/coherent_stream_inspector.html` only adds a receipt-table width/wrapping
rule after an observed visual defect. No new renderer, framework or dependency.
New `tests/test_shared_linear_probe_inspector.py` has 35 model-free cases.
[Usage](../ordinary-probe-inspector.md) describes scopes and limitations.

All three phases, six tasks, fixed curves and inactive methods remain visible.
Auxiliary rows distinguish attempts, returns, observed/unknown applications,
current input rows and additional charged exposure. Each event exposes actual
fit rows, current/retained TRAIN use, uncertain work and its original work object.
Recipe/source metadata, original auxiliary budget and overlapping clocks remain
separate. Every displayed numerical/flag cell resolves to a saved pointer; no
summed total, fake SGD credit, source authentication or rescoring is introduced.
Other unsupported costs are unavailable, not zero. Partial/missing/null/false/zero
and role/task/final-data/refusal boundaries remain covered.

## Executed checks

Physical cwd `D:\Dev\Repositories\Circadian`; every shell sets TEMP/TMP to
`D:\Dev\_Temp`, Linux also TMPDIR to `/mnt/d/Dev/_Temp`. Stage
`artifacts/runs/shared-linear-probe-inspector-20261031/` holds receipts/helpers;
browser scripts, snapshots and screenshots are under
`output/playwright/shared-linear-probe-20261031/`.

The one-off checker reuses the established checker, blocks numerical imports and
plugin autoload, disables pytest cache, uses D: test temporary directories and
samples RSS. It ran the new cases plus existing reserved-probe, contextual-linear,
ticket and coherent-stream suites. Final 177 = 35 new + 142 affected regressions.
All five local #356/#358/#360 captures were present; zero skips. Original captures
are byte-preserved. Local-capture cases explicitly skip when captures are absent;
that does not qualify a clean clone.

```powershell
$env:TEMP='D:\Dev\_Temp'; $env:TMP='D:\Dev\_Temp'
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/check.py artifacts/runs/shared-linear-probe-inspector-20261031/windows-work.json
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp PYTHONDONTWRITEBYTECODE=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/shared-linear-probe-inspector-20261031/check.py artifacts/runs/shared-linear-probe-inspector-20261031/linux-work.json
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/check.py artifacts/runs/shared-linear-probe-inspector-20261031/windows-final-work.json
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/check.py artifacts/runs/shared-linear-probe-inspector-20261031/windows-corrected-work.json
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp PYTHONDONTWRITEBYTECODE=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/shared-linear-probe-inspector-20261031/check.py artifacts/runs/shared-linear-probe-inspector-20261031/linux-final-work.json
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/statics.py statics-initial.json
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/statics.py statics-final.json
```

These exclusive receipt destinations and invocations are consumed evidence,
not retry instructions. Final Linux handle34043 exited0 without restart; initial
Linux handle76686 likewise completed. No model/numerical import attempts.

| Check | Outcome | Body seconds | Sampled RSS bytes |
|---|---|---:|---:|
| Windows initial | 174 pass | 2.901271800 | 77,369,344 |
| Linux initial | 174 pass | 21.208281470 | 68,816,896 |
| Windows stricter metadata | 172 pass / 4 fail | 2.892509100 | 84,017,152 |
| Windows corrected final | 177 pass | 2.761566100 | 80,936,960 |
| Linux final | 177 pass | 21.767245933 | 69,316,608 |

Tests total51.530874403s, within180s aggregate/45s256MiB each/3invocations per
platform. The failed invocation mistyped the actual `probe_auxiliary_updates`
receipt object as a scalar. Corrected mapping/scalar distinction and an explicit
malformed-shape case; failed receipt and `FAILURES.md` remain. Scoped Ruff lint,
format, default and Linux-target mypy all pass on four Python files, both initial
and final:10.479432200s total. Static RSS was not sampled. Formatting also ran
before cohorts. Python checks remained applicable after the final CSS-only change;
all final pages received actual browser qualification instead of repeating them.

## Actual HTML and browser behavior

Existing CLI `main()` was invoked with its ordinary argv under a numeric-import
blocker, RSS sampler and15s/output watchdog. Exact four argv lists per invocation,
input/output paths and durations are in `build.json` / `build-final.json`:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/build.py
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/build_final.py
```

Inputs are #360joint/current_only JSON, unchanged #358joint JSON, and a clearly
synthetic partial UI fixture. #360joint alone has its separately supplied work
file; association remains UNVERIFIED. No old failed scientific source was retried
or reconstructed. Initial four exports PASS .334651900s44,560,384B; final four
PASS .334409500s42,713,088B. Final sizes joint1,051,639/current-only988,546/
legacy-joint984,457/partial70,048B; each under existing16MiB. Inputs obey8MiB.
Original and final outputs coexist; total8exports .669061400s,0model work.

Playwright skill applied, existing npx/installed D: CLI and Edge reused, no install
or download. The ordinary permitted CLI file-page route succeeded; historical
#154/#175 browser-policy failures and their incomplete criteria remain unchanged.
No server/CDP/publication workaround. Explicit persistent D: profile
`D:\Dev\_Temp\circadian361-profile` began absent and was reused sequentially,
then closed. Browser source dialogs do not authenticate the source producer.

```powershell
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/browser_call.py open file:///D:/Dev/Repositories/Circadian/artifacts/runs/shared-linear-probe-inspector-20261031/joint.html --config output/playwright/shared-linear-probe-20261031/config.json --profile D:/Dev/_Temp/circadian361-profile --persistent
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/browser_call.py snapshot
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/browser_call.py run-code --filename output/playwright/shared-linear-probe-20261031/qa.js
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/browser_call.py close
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/browser_call.py open file:///D:/Dev/Repositories/Circadian/artifacts/runs/shared-linear-probe-inspector-20261031/joint-final.html --config output/playwright/shared-linear-probe-20261031/config.json --profile D:/Dev/_Temp/circadian361-profile --persistent
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/browser_call.py snapshot
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/browser_call.py run-code --filename output/playwright/shared-linear-probe-20261031/qa-final.js
.venv/Scripts/python.exe -B artifacts/runs/shared-linear-probe-inspector-20261031/browser_call.py close
```

All8commands EXIT0, total6.220696600s/90s. First30+final31assertions pass;
final four pages verify32curvedots/32receipts/18tasks/five methods, all1102joint
source cells, exact budget/zero via keyboard/resource dialog, actual joint old
3→4→4 vs control3→2→3, unchanged fresh2→2→2, legacy18vs once12return rows,
384vs288extra charges and current-only0retained TRAIN. Partial running work,
null/false/missing and inert hostile text remain visible. Zero pageerrors/external
requests observed during QA. Initial navigation is offline HTML with no external
resource declarations; no broader network monitor claim.

Visual review of initial desktop/narrow images found fragmented table words.
Comment6541769612 prospectively amended scope from4to8exports and6to8images/
snapshots before work. Four final exports plus one final narrow image and final
DOM/desktop+narrow layout assertions qualify the targeted CSS repair. Exactly
8snapshot-or-image captures including automatic opens/manual snapshots; all
initial failures/captures retained. One patch anchor and one shell parse failed
before execution; then a short stdin edit was used. No further checks were added
just to fill budget. Profile sampled high-water13,889,493B/128MiB during calls;
sampling gaps between calls and full process-tree memory are unmeasured. Profile
and outputs retained, not deleted. Final narrow screenshot shows whole words and
local horizontal scrolling, no page overflow. No real-device/accessibility audit.

## Review and remaining scope

Lead self-reviewed actual source, combined checks, procedure, costs and visual
captures. Finite independent actual-source/evidence review ACCEPTED the scoped
delivery with no blocker; [REVIEW.md](../../artifacts/runs/shared-linear-probe-inspector-20261031/REVIEW.md).
Six readonly calls consumed2.5574783s (one absent-path discovery exit1 and one
partially truncated read retained; remaining bounded reads completed coverage).
Independent final HTML inspection resolved joint1102/control1089/legacy1086/
partial30 linked cells with zero mismatches. Three real-report final pages differ
from initial only by targeted CSS; partial also uses its separately named fixture.
The reviewer inspected actual source/receipts/QA, not only this report. No
independent browser rerun/pixel/device audit, numerical replication or source
authentication; unchanged regression test bodies were not wholly re-reviewed.
The finite exception ended. Broader science/release requirements stay open.

Skipped: full suite, clean clone, new wheel/install, unchanged math/baseline/native
learning suites, newfits/predictions/teacher/games/physics/GPU, final evaluation,
remoteCI, real-device testing and release. All historical failures/SPENT captures
and scientific limitations stay. Scoped UI correctness does not improve #360's
scientific strength: fresh tasks already succeeded, only known-template retention
was observed; no broad reliability/generalization/matchedcompute/CPC benefit.
Original R4.7 topology/lineage/sleep data requirements, broader R2/R4/R5/R7 and
full goal remain OPEN. No commit/push/install/delete/global-setting changes.

Critical-path reassessment: do not start another inspector/package/capture/dose
sequel. Existing ordinary single-head recent/coverage replay is implemented and
has its own spent native evidence; do not rebuild it. Code inspection instead
finds shared-head cross-context joint packet composition in the isolated auxiliary
probe route (`ticket_linear_rehearsal`), while the ordinary single-head replay
route fits selected retained packets separately. Generic `StreamRehearsal` already
has joint composition, but its admission requires `ContextLinearFitSnapshot` and
its policy selects same-public-context data; it cannot admit shared `LinearFitSnapshot`.
Next implementation action: inspect and extend that existing rehearsal boundary
with an opt-in bounded shared-head joint TRAIN packet update, beginning with
role/arrived-consent/extra-row-accounting/default/failure tests and reusing
`compose_shared_packet`. Preserve source/actor/RNG and original clocks, separate
authority, no probe/final/monitor-label conversion or guaranteed benefit. No new
experiment allowance or tuning is implied; first reconcile existing paths to
avoid duplicate functionality. This moves useful private known-task evidence
toward ordinary candidate learning through the existing stream. Planned [#362](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/362) owns that next outcome with all five criteria unchecked and no numerical allowance.

Measured storage before final review/archival handoff: stage excluding HTML/cache
369,778B/16MiB, HTML6,188,706B/32MiB, dedicated cache initially0→13,812,523B/64MiB,
browser output541,665B/8MiB, retained closed profile13,828,542B/128MiB. Later
small handoff/review receipts are included in final storage readback. Scoped
tracked and four owned untracked-file whitespace checks pass.

[Canonical scoped acceptance](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/361#issuecomment-6541951259); live #361 CLOSED/all5criteria checked, #362 planned OPEN/all5unchecked. Full goal ACTIVE; independent exception ended.

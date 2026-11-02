# Saved ordinary learning probe inspector — 2026-10-30

R4.7-TW.PROBE.I / [issue329](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/329)
delivers an offline view through the existing dispatcher, source-cell projection,
shared HTML template and CLI. Prior turn #328 was PROGRESS; its accepted actual
LS observation and earlier repeated-update, R2.1 and matched controls were reused.
No learner, baseline, experiment or final evaluation ran during this increment.
HEAD remains d9119e80926a139a2f38460564d7eab1b6d2a2b0 with inherited dirty work preserved.

Implemented: new `src/app/reserved_probe_inspector_projection.py`,
`tests/test_reserved_probe_inspector.py`, narrow dispatch in
`src/app/key_door_inspector.py`, conditional rendering in
`src/infra/coherent_stream_inspector.html`. Existing schemas/defaults and other
viewer branches remain. No new runtime dependency. Pure app code imports no
outer adapter/infra or numerical library. [Usage](../ordinary-probe-inspector.md).

Executed behavior: all three saved phases, recorded success/change fields,
32 planned CE/accuracy readings, four CE charts, separate costs and five method
availability rows render. Every metric/dot opens its original JSON pointer.
Unknown/null/false/zero and missing checkpoint gaps remain distinct. Object
records open compact source dialogs. Offline JSON escaping remains inert.
All episode role/task headers are admitted before outcome access, including
later phases. Validation is not source authentication or recomputed science.

Local stage: `artifacts/runs/reserved-probe-inspector-20261030/`.
Browser receipts/scripts/snapshots/screenshots:
`output/playwright/reserved-probe-20261030/`.
Final pages: `actual-final.html`, `failed-final.html`, `partial-final.html` in
the stage. Original pre-resize pages and screenshots are preserved too.

The actual page embeds unchanged #328 `native.json`: old4→4→4/fresh1→2→2,
54 saved action rows, near-tie caveat and adverse return CE1.11439266347417.
No saved loss means recovery remains unproved. The failed #323 gzip was decoded
with an 8 MiB read bound, not rerun: private outcomes unknown, zero numeric curve
points, source12updates72rows still paid. A small synthetic partial fixture checks
null success, zero CE, null CE, missing checkpoint8 and hostile inert text.
Fixture behavior is not an adaptation observation.

## Commands and observed outcomes

All shells ran in physical `D:\Dev\Repositories\Circadian` with TEMP/TMP
`D:\Dev\_Temp`; Linux TEMP/TMP/TMPDIR `/mnt/d/Dev/_Temp`.
`check.py` blocks NumPy/Torch/torchvision and pytest plugin autoload, samples RSS,
and enforces each60s256MiB/shared180s including previous invocations.

```powershell
.venv/Scripts/python.exe -B artifacts/runs/reserved-probe-inspector-20261030/check.py artifacts/runs/reserved-probe-inspector-20261030/windows-combined-work.json tests/test_reserved_probe_inspector.py tests/test_context_serving_inspector.py tests/test_ticket_workflow_inspector.py tests/test_coherent_stream_inspector.py
wsl.exe bash -lc 'cd /mnt/d/Dev/Repositories/Circadian && export TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp && /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/reserved-probe-inspector-20261030/check.py artifacts/runs/reserved-probe-inspector-20261030/linux-combined-work.json tests/test_reserved_probe_inspector.py tests/test_context_serving_inspector.py tests/test_ticket_workflow_inspector.py tests/test_coherent_stream_inspector.py'
.venv/Scripts/python.exe -B artifacts/runs/reserved-probe-inspector-20261030/statics.py statics-final.json
.venv/Scripts/python.exe -B artifacts/runs/reserved-probe-inspector-20261030/build.py
.venv/Scripts/python.exe -B artifacts/runs/reserved-probe-inspector-20261030/rebuild.py
```

Build/check destinations are exclusive; these commands already consumed their
destinations and engineering allowances. They are evidence, not retry commands.

| Invocation | Passed | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
| Windows first, new cases | 36 | 1.043845800 | 55,398,400 |
| Windows initial combined | 147 | 2.739801500 | 80,154,624 |
| Linux initial combined | 147 | 18.358283342 | 71,077,888 |
| Windows final combined | 147 | 2.690205900 | 76,382,208 |
| Linux final combined | 147 | 19.183994437 | 70,823,936 |

All EXIT0, zero skips/failures/numeric-import attempts. Aggregate44.016130979s
of180s. Expected existing-output CLI refusals are assertions in passing legacy
tests, not failed export attempts. Final scoped Ruff/format/mypy default and
Linux target all PASS on the three changed/new Python files,5.286335700s.
Initial mypy found three typing errors; `statics-first.json` preserves them.
They were fixed before final checks. Scoped static commands aggregate12.600668s
plus the recorded initial formatting pass, within90s; no broad type claim.

First three CLI builds: EXIT0 .279494700s/54,362,112B; final three:
EXIT0 .142671800s/29,888,512B; no forbidden imports. Final HTML sizes
834,123/477,792/81,061B, each below existing16MiB writer bound. Original three-build
limit was explicitly amended **before** three final rebuilds after browser QA
found desktop SVG geometry shrinking labels on resize (issue comment6531995147).
The repair redraws only this view, removes the old chart area and deduplicates
its navigation link. Shared build90s/each30s256MiB limits were not renewed.

Playwright skill applied. Initial no-install npx resolution canceled (missing
package), global CLI absent, cached/Node imports unavailable. Discovery stopped;
comment6531801071 prospectively authorized ONE pinned, script-disabled devtool
install, no browser download, under `D:\Dev\Tools\CircadianInspector329`:

```powershell
$env:NPM_CONFIG_CACHE='D:\Dev\_Caches\npm'; $env:PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD='1'
npm install --prefix D:\Dev\Tools\CircadianInspector329 --ignore-scripts --no-save --no-package-lock @playwright/cli@0.1.22
```

EXIT0,4.2835006s,3packages/18,752,839B SDK. Initial no-install failure and install
logs retained; no repository/global dependency/settings edits. Existing Edge,
headless isolated profile, file URLs only. `browser_call.py` records every exact
command/stdout/stderr/exit and charges shared180s. Final workflow used:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/reserved-probe-inspector-20261030/browser_call.py open file:///D:/Dev/Repositories/Circadian/artifacts/runs/reserved-probe-inspector-20261030/actual.html --config output/playwright/reserved-probe-20261030/config.json
# goto each -final page, snapshot before selecting controls; exact calls in browser-work.jsonl
.venv/Scripts/python.exe -B artifacts/runs/reserved-probe-inspector-20261030/browser_call.py run-code --filename output/playwright/reserved-probe-20261030/actual-final-qa.js
.venv/Scripts/python.exe -B artifacts/runs/reserved-probe-inspector-20261030/browser_call.py run-code --filename output/playwright/reserved-probe-20261030/failed-qa.js
.venv/Scripts/python.exe -B artifacts/runs/reserved-probe-inspector-20261030/browser_call.py run-code --filename output/playwright/reserved-probe-20261030/partial-qa.js
.venv/Scripts/python.exe -B artifacts/runs/reserved-probe-inspector-20261030/browser_call.py close
```

Browser commands total8.570499200s, all EXIT0. Actual checks confirm32dots/32score
rows/5methods, source click/Enter, exact adverse return CE, unverified resource
dialog, no duplicate resize area/navigation, desktop1440 and narrow390 widths
equal page scroll widths, narrow358×210 viewBox, zero external requests/errors.
Failed case has18planned rows/54unknown outcome cells/no dots and paid12/12/72
source counters. Partial case has5first-panel dots, broken polyline lengths
[1,1,2,1], explicit null, zero via Space, inert hostile string/no image injection.
Lead visually inspected original and corrected narrow/desktop screenshots.
Browser closed; no matching temporary Playwright profile root retained in D TEMP.
Live profile-byte high-water was **not sampled**, so its prospective128MiB peak
allowance is not independently established; no exact profile-memory claim.
Retention and scientific claims are not inferred from this supporting limitation.

Before closeout, comment6532095833 prospectively allowed one sequential explicit-D
profile qualification, at most20s including failures and within original shared
180s browser allowance, same128MiB profile limit, one live browser. First open
failed because the reused config requested isolated mode with a persistent profile;
the subsequent readout also refused the absent browser, close confirmed not open.
`profile-check.json` retains all three commands/.973567200s, no browser/model work.
The fix was only a separate QA config with `isolated:false`, not product source.
`profile_check_final.py` includes failed time in its remaining allowance. Its
open/32-dot/title check/close all EXIT0,2.207330600s,89 file-size samples,
9,267,522B observed profile high-water, below128MiB. Profile retained at
`D:\Dev\_Temp\circadian329-bounded-profile`; no delete. Receipt
`profile-check-final.json`. This qualifies the final explicit profile at sampled
resolution, not the original discarded profile's unobserved peak or exact process
memory. Total browser work including both extra attempts11.751397000s/180s;
both sessions closed, never concurrent. Original failed/unknown history preserved.

## Review, limitations and next action

Lead self-reviewed source/procedure/results separately. One finite independent
reviewer accepted source and embedded actual/failed/partial values:630/17/225
nonmissing source cells each resolve exactly. Three read-only commands EXIT0,
.6341315s; after demonstrated resize repair, ONE prospectively allowed read-only
addendum EXIT0 .1306356s accepted final renderer/recorded QA. Total .7647671s,
no edits/project imports/models/tests/network/children; exception ended. This is
scoped independent acceptance, not independent whole-repository/release review.

Skipped: full suite, clean clone, wheel rebuild/install, GPU, physical robot/game
simulation, new native/CE/LS fits or predictions, model comparison, generalization,
matched-compute study, real device/accessibility audit and release. This is a
metadata viewer; accepted scientific evidence is reused, not strengthened. #325
wheel predates this source. Broader R4.7/topology/lineage/sleep, R2.1 reliable
adaptation/recovery, R5/R7, science/release and fullgoal remain OPEN. #323 stays
permanently FAILED/SPENT. No original budget, consent, counter or evidence changed.

Final measured retained storage before closeout: stage3,672,725B<8MiB,
browser captures/scripts/snapshots1,059,040B<16MiB, SDK18,752,839B<128MiB,
D npm cache39,784,501B<128MiB, explicit closed profile9,129,706B<128MiB.
The entire existing D npm cache is measured, not attributed as new install growth.
Scoped tracked whitespace check passed with LF/CRLF advisories; no commits,
pushes, deletions, global settings changes or unrelated resets.

Critical path: do not add another capture, inspector sequel or baseline rerun.
The ordinary BP CLI still exposes CE only as `--probe-initialization ce` inside
`--reserved-probe`; it does not expose the accepted #321 TRAIN-fitted CE predictor
for ordinary task execution. The useful conventional known-task neural baseline
therefore exists as an auxiliary accepted observation, while ordinary raw BP
task behavior remains negative. **Next implementation action:** add an explicit
opt-in TRAIN-fitted CE readout mode to the existing ordinary BP workflow, reusing
`fit_categorical_ce_readout`, `CategoricalCEReadoutPredictor` and existing task
episodes. Begin with role/cost/default/refusal tests and source/actor/RNG isolation;
do not install a probe checkpoint into the actor, fabricate SGD credits or silently
promote a fit. Freeze any genuinely new native qualification separately. Preserve
all existing negative defaults and the spent323 case; no hyperparameter/seed search.

GitHub completion receipts:329 canonical6532139645, closed/completed; tracker18
6532146186, parent21 6532146693, parent23 6532147155 all remain open.
One final documentation patch missed a wrapped anchor and made no change;
this receipt was appended directly. Final local closeout.json preserves it.

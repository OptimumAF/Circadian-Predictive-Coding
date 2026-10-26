# Recorded key/door inspector

## Saved FIFO36 fresh-task probe

Build a standalone page from the existing saved report:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/saved-fifo-fresh-return-20261019/native.json --output artifacts/fifo36-probe.html
```

Use a new output path. The page filters score curves and tables by source cohort
(0 or 128 updates), keeps F `probe_test` and A TRAIN roles explicit, and provides
source links for recorded measurements, signed metrics, preservation flags and
phase costs. Unknown values remain unknown. Original incomplete/spent history
and censored attempt labels remain separate from the successful disposable probe.

No autonomous games were planned in this assay; success and return are unavailable.
Mature A remained exact, so this shows retention. Cold A acquired during return;
neither observation demonstrates recovery from mature loss or circadian benefit.
Source links refer to the embedded original JSON, not an authenticated external
history. The builder executes no model/environment and reads no final dataset.

This exact protocol also permits up to16MiB pretty input, while compact projection
remains8MiB/500000nodes/depth64 and output16MiB. Other existing family limits stay
unchanged. Separate resource artifacts are refused for this family. Extend via
explicit saved-protocol validation in the app layer, with evidence/unknown and
exclusive-publication tests before changing the shared template.

Scoped tests: `python -m pytest -o addopts= -q tests/test_minigrid_saved_fifo_inspector.py tests/test_minigrid_saved_fifo_inspector_files.py`.
Full commands, failures and review: [increment results](../artifacts/runs/saved-fifo-inspector-20261019/RESULTS.md).
Interactive desktop/mobile/filter/source-dialog QA remains open: browser policy
refused the local file URL; no alternate route was attempted.

## H32 fresh-task learning and old-task return

The existing builder accepts the four saved H32 joint128/F16/A16 game protocols.
It shows benchmark fit, isolated F learning and A return for both cues, and uses
the existing public-view playback for native games. Source checkpoints, native
update counts and disposable phase counts remain separate. Missing planned
checkpoints/games are explicit unrun placeholders with no invented source link;
failed observations, unknown metrics and paid-work lower bounds remain visible.
False preservation flags are displayed as unproven, never converted to passes.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/original-wide-probe-games-study-20261018/native-bp.txt --output artifacts/bp-learnability.html
```

Use a new output path. The generated page is standalone, with no external assets
or model/environment execution. Inspect exact records through Source buttons;
Previous/Next/range/Play navigate recorded frames. Initial148 + current148 is
shown separately from the seven raw native action probabilities. No private map
or missing decision is reconstructed. Plots show observed points without filling
gaps; metric tables label unrun points and their unknown values.

This quartet is **incomplete and scientifically unqualified**. BP's partial F-red
success and F-blue negative terminal are retained; no comparative advantage,
full retention/recovery or final generalization follows. Training is inactive,
probe work stays separate, and final evaluation remains unavailable.

Raw input allowance is16MiB only for these exact four protocols (the pretty BP
file is8401101bytes); compact JSON remains bounded to8MiB/500000nodes/depth64 and
HTML output16MiB. Other input formats keep their8MiB allowance. Duplicate keys,
nonfinite values, corrupted native evidence and inconsistent complete claims are
refused before publication. Separate resource files are unsupported here;
embedded costs describe their recorded scope and may be unknown or lower bounds.

The pure `src/app/minigrid_learnability_inspector.py` projection shares native
frame validation with the fixed-A inspector; the existing dispatcher, file port,
CLI and MiniGrid HTML handle publication. Extend only by adding a tested explicit
saved protocol contract; keep learning and IO out of app projection code.
Focused command: `python -m pytest tests/test_minigrid_learnability_inspector.py -q`.
Final executed evidence and interactive-QA scope are recorded in
[the increment results](../artifacts/runs/learnability-inspector-20261018/RESULTS.md).


## Public-history and bounded memory reports

The existing builder now accepts the saved memory live, memory E16 capacity and
four-method matched-history live protocols. It displays all recorded methods,
TRAIN curves, raw action playback, separate probe origins/charges, observed table
capacity, whole-report resources and exact source links. No learner or environment
runs during publication. Missing, null, failed and inactive readouts stay distinct.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/bounded-action-memory-20261017/native.json --output artifacts/memory-inspector.html
.venv/Scripts/python.exe -X utf8 -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/action-memory-probes-20261017/native.json --output artifacts/memory-probe-inspector.html
.venv/Scripts/python.exe -X utf8 -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/matched-history-live-20261017/native.json --output artifacts/matched-history-inspector.html
```

Each output must be new. Separate resource files are refused for these schemas;
embedded time/RSS describe the whole report, not individual methods. Count
storage updates are not SGD. Raw TRAIN probabilities remain unmasked. Probe
updates are relative to disposable copies; benchmark clocks and serving state
remain separate. Probe end tables are not serialized, so their growth is not
presented as measured allocation. Topology, lineage, sleep and serving promotion
are unavailable where absent. Negative neural outcomes remain visible.

```text
src/app/public_history_inspector_projection.py  bounded saved-report projection
src/app/key_door_inspector.py                  shared dispatch/raw TRAIN selector
src/infra/public_history_inspector.html        offline presentation and controls
src/infra/key_door_inspector_files.py           existing bounded exclusive publication
tests/test_public_history_inspector.py         evidence, failures, pointers and files
```

Validation:443 combined inspector checks and57 closing checks pass; four-path
Ruff/format/mypy win32+linux and Node syntax checks pass. Independent source and
saved-output review accepted exact embedded evidence and150 episode pointers.
**Interactive QA remains open:** browser security rejected local-file navigation
and explicitly prohibited rerouting. No desktop/mobile layout, control behavior,
runtime-console or screenshot claim follows from static checks.

Open the [generated memory page](../artifacts/runs/r47-public-history-inspector-20261017/memory-live.html)
locally for manual QA: check desktop and narrow layout, change method/metric,
select an episode, use Previous/Next/range/Play, open a curve point with keyboard
and close its source dialog. Also check the
[probe page](../artifacts/runs/r47-public-history-inspector-20261017/memory-probes.html),
[four negative controls](../artifacts/runs/r47-public-history-inspector-20261017/matched-live.html),
and clearly labeled [engineering partial fixture](../artifacts/runs/r47-public-history-inspector-20261017/engineering-partial.html)
for null/missing states and console errors. Preserve pending acceptance until
those observations exist. [Commands, failures, scope and costs](../artifacts/runs/r47-public-history-inspector-20261017/RESULTS.md).

Tests: `python -m pytest tests/test_public_history_inspector.py -q`.
Run Ruff `check`/`format --check` and scoped mypy on the four Python paths above,
using `--follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL`
and both `--platform win32`/`linux`. Exact commands are in the stage receipts.
Extend via an explicit saved protocol/schema and malformed/partial fixtures;
reuse existing publication bounds/escaping. Add no model execution to the inspector.

## Coherent stream comparison

The same CLI now accepts the fixed independent-data replay-removal report:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/r28-removal-confirmation-20261017/native-stdout.txt --resource-artifact artifacts/runs/r28-removal-confirmation-20261017/native-resources.json --output artifacts/coherent-stream-inspector.html
```

Use a new output path. Open the standalone HTML directly in a browser. Default
visibility includes all four methods and both replay policies, all eight signed
outcome rows, four pair rows and A/B curves. Method/replay/metric controls filter
curves only. Click a score, cost or outcome, or focus a chart point and press
Enter/Space, for its exact source value. Close/Escape dismisses the source dialog.
Raw-score details are reversible. Narrow tables scroll locally.

Recorded null and missing are distinct, and null gaps break lines. Paid failed
work stays visible; planned limits do not become measured counts. Optional
resource association is labeled UNVERIFIED. Topology, lineage and sleep are
unavailable in this source. The builder performs no learning, prediction or
scientific metric recomputation and cannot establish a comparative benefit.

The new app module `src/app/coherent_stream_inspector_projection.py` validates
and projects the exact protocol. Existing file IO owns bounds/escaping/exclusive
publication; `src/infra/coherent_stream_inspector.html` owns presentation. Tests:
`-m pytest -q tests/test_coherent_stream_inspector.py`; integrated route regression
suite386pass, scoped Ruff/format/both-platform mypy and final browser QA pass.
Extend through an explicit saved schema plus behavioral malformed/partial tests;
keep model updates outside the inspector. No environment variables or dependencies
added. Full R4.7 remains open.

[Accepted saved comparison](../artifacts/runs/r47-coherent-stream-inspector-20261017/learning-comparison.html),
[commands, failures, limits and next action](../artifacts/runs/r47-coherent-stream-inspector-20261017/RESULTS.md),
[independent review](../artifacts/runs/r47-coherent-stream-inspector-20261017/REVIEW.md).

## Recorded game reports

Build an offline HTML page from an existing single return-policy, paired
return-probe, supported MiniGrid or sensor JSON report. The page displays every saved episode, recorded-step
playback, source JSON pointers, independent probe curves, training/final scores,
replay and resource costs. No learner or environment runs during publication.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/r21-return-fresh-task-probes-20261016/native.json --output artifacts/key-door-inspector.html
```

The output must be a new file. Open it in a browser; no server, network, account,
environment variables or new dependencies are required. Choose condition and
episode, then advance the recorded frame or use Play/Pause. Source buttons show
the exact saved record and its JSON pointer. Playback stops at the final frame.
The first recorded episode remains the default, including when it failed.

## Boundaries and limits

```text
src/app/key_door_inspector.py          pure validation and frame projection
src/app/acquired_inspector_validation.py explicit TRAIN readout consistency
src/infra/key_door_inspector_files.py  bounded JSON IO and exclusive publication
src/infra/key_door_inspector.html      standalone presentation and playback
scripts/build_key_door_inspector.py   explicit source/output CLI
tests/test_key_door_inspector_*.py     projection and file/CLI behavior
```

Why: reuse the existing v14 app → infra → CLI pattern while preserving its
separate strict schema. Projection derives map positions from recorded layout
and applied actions, without constructing an environment or predicting actions.
The map overview is separate from the eight local observation features. A
terminal post-observation need not have been used for a decision. Topology,
lineage and sleep proposals are explicitly unavailable, not invented.

Custom-game inputs are UTF-8 JSON ≤8 MiB, one/two conditions, ≤128 episode slots per condition
and ≤32 transitions per episode. Duplicate keys, nonfinite numbers, malformed
decisions, inconsistent observations/actions/rewards/counters and contradictory
unobserved work are refused before output. Missing partial metrics remain
unknown. Output is ≤16 MiB with escaped embedded JSON, text-only untrusted
content and no remote resources. Existing inputs/destinations are never
overwritten. This is a bounded saved-report slice, not every repository report.

## Checks and observed behavior

Prefix Python commands with `.venv/Scripts/python.exe -X utf8 -B`:

```text
-m pytest -q tests/test_key_door_inspector_projection.py tests/test_key_door_inspector_files.py tests/test_v14_dashboard_projection.py
-m ruff check src/app/key_door_inspector.py src/infra/key_door_inspector_files.py scripts/build_key_door_inspector.py tests/test_key_door_inspector_projection.py tests/test_key_door_inspector_files.py
-m ruff format --check <same five paths>
-m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 <same five paths>
```

Repeat mypy with `--platform linux`. Final combined checks passed:122 focused
cases, all five-path statics, and JavaScript syntax. After the final CSS/note/
favicon changes,22 file/CLI cases passed again; the checked JavaScript is exactly
unchanged. Saved final payloads match the three source artifacts exactly.

Browser evidence covers first failure, successful A return, wrong-key B return,
cue disappearance, action/source evidence, condition switching, empty failed
condition clearing, null charges, inactive probes/replay and explicit unknowns.
Final paired mobile width390 has no document overflow. Play to completion was
observed; a Pause-click check was not completed after a stale reference failure.
Desktop screenshot predates only the final wording/favicon/mobile CSS fixes.

The first mobile overflow, blocked file URL, stale browser references, one
read-only CLI evaluation failure and uncached optional HTML formatter attempt
are preserved in the evidence summary. Browser automation used localhost after
the tool refused a file URL; that temporary server and browser are closed.
Its cumulative active duration was not recorded, so the original five-minute
engineering ceiling is unverified; no further browser work is authorized here.

[Final paired inspector](../artifacts/runs/r47-key-door-inspector-20261016/inspector-v3.html),
[commands/outcomes/failures](../artifacts/runs/r47-key-door-inspector-20261016/RESULTS.md),
[independent review](../artifacts/runs/r47-key-door-inspector-20261016/REVIEW.md).
No new training or comparative experiment ran. Existing negative learning
results stand; interface correctness establishes no circadian advantage.

Extend by projecting another explicitly supported saved schema and adding
malformed/partial behavioral cases; keep learner updates out of this layer.
Full R4.7 still needs actual topology, lineage and sleep events when available,
and broader rolling outcomes. Useful retention, generalization and release
readiness remain open.

## Acquired-task TRAIN reports

Single acquired-task reports use the same builder and recorded playback:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/r21-acquired-replay-20261016/coverage/pilot.json --output artifacts/acquired-coverage-inspector.html
```

The view explicitly labels TRAIN diagnostics and keeps raw teacher fit separate
from actual recorded live success. Every A/C cue at wake192/256/320 remains,
including step-limit failures. Replay reports record native288/384/480 clocks;
where a report does not record a native checkpoint count, the view says
“Not recorded”. Exact source links retain `/acquired_return_diagnostics/...`
pointers; no TRAIN record is renamed into a final result.

The page shows the recorded initial-A and new-C acquisition premises, all signed
raw/live gain/loss metrics, replay composition and permanent charges, native
work, diagnostic callbacks, time and observed RSS. Missing metrics, topology,
lineage and sleep proposals stay unavailable. Existing final-policy reports and
paired off/on views retain their schema. Mixed TRAIN/final records are refused.

The historical browser ceiling described above remains unverified and closed.
The new acquired-task presentation has a separate prospective240seconds of
active browser-command work, documented in its scope; no training, scientific
budget renewal or further comparative experiment is involved.
[Current scope and evidence](../artifacts/runs/r47-acquired-task-inspector-20261016/RESULTS.md).

Current acquired-task verification:187focused tests/0skips, six-path Ruff/format/
Windows+Linux mypy and final JavaScript syntax checks pass. The real file CLI
published all three accepted sources unchanged: no-replay, recent48 and coverage,
with twelve exact episode pointers and132recorded frames each. Browser checks
used60.5207active command seconds within the separate240-second allowance, then
closed the owned Chrome session and temporary localhost server. Desktop1280 and
mobile390 had no document overflow. Both failure and success playback, timed
Pause stability, source evidence, clocks, metrics and costs were observed.

The tool blocked file URLs, one old element reference became stale, and its
installed CLI did not support the documented optional network command. All
three errors are preserved; no browser network trace is claimed. The output
itself remains standalone, with no remote assets. Negative learning results
remain unchanged; this is presentation correctness, not a new scientific run.

[Coverage TRAIN inspector](../artifacts/runs/r47-acquired-task-inspector-20261016/coverage-inspector.html),
[recent-only TRAIN inspector](../artifacts/runs/r47-acquired-task-inspector-20261016/recent48-inspector.html),
[no-replay TRAIN inspector](../artifacts/runs/r47-acquired-task-inspector-20261016/off-inspector.html).

## Recorded MiniGrid fixed-A run

The same builder supports exactly `minigrid_fixed_a128_acquisition_v1` from the
accepted saved TRAIN run. No optional MiniGrid/Torch installation or model call
is needed to inspect it. Use a new destination:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/r21-fixed-a-acquisition-20261016/native-result.json --output artifacts/minigrid-inspector.html
```

[Published MiniGrid inspector](../artifacts/runs/r47-minigrid-inspector-20261016/inspector.html)
is3518758bytes, within the unchanged16MiB output cap; source7047945bytes fits8MiB.
The file embeds its data, styles and script with no remote resources. Browser
verification used the temporary loopback server; direct-file opening was not
verified on this platform.

Choose any of five methods and all0/128 blue/red episodes. The page starts with
the first timeout. Current public7x7 pixels and the initial frame remain separate:
FF uses initial+current296, RNN keeps148 with an initial reference panel. The
agent faces up in this egocentric view; recorded global direction is metadata.
Unseen cells are shaded. This is not an invented map or rerun of the environment.
Seven raw action probabilities use first argmax with no action mask or repair.

Step/Play/Pause show only recorded frames. At a terminal post-observation there
is no next decision or probability row. Source buttons open exact observation,
296-input, decision, physics, metric and cost JSON records. All raw checkpoints,
positive and negative returns, both-cue acquisition flags and signed comparisons
remain available. Unknown actor versions, topology, lineage and sleep stay
unknown. An incomplete/empty method clears earlier pixels/actions and retains
paid costs without zero-imputing unavailable diagnostics.

```text
src/app/minigrid_inspector_projection.py  pure validation/public-view projection
src/app/key_door_inspector.py             exact protocol dispatch, old paths retained
src/infra/key_door_inspector_files.py     original safe IO, explicit template choice
src/infra/minigrid_inspector.html         recorded7x7/action/source interface
scripts/build_key_door_inspector.py       unchanged shared publication CLI
tests/test_minigrid_inspector_projection.py law/pointer/partial/cost behavior
tests/test_minigrid_inspector_files.py    safe publication and original limits
```

205 focused cases pass with no skips, including existing custom/acquired views.
Scoped Ruff/format and win32+Linux mypy pass; final emitted JavaScript syntax and
embedded source/pointers are verified. Exact commands are retained in
[the qualification receipts](../artifacts/runs/r47-minigrid-inspector-20261016/).
The standard prefix is `.venv/Scripts/python.exe -X utf8 -B`; run stage
`run_checks.py focused` or `statics` only after preserving existing receipts.
Existing test/source artifacts must be present; this is scoped checkout evidence.

Observed browser checks cover cue disappearance/retained history, all five
methods, blue−1failure/redgoal, Play/Pause/final stop, exact source dialogs,
RNN148 and synthetic construction failure/empty clearing. A requested390x844
viewport measured document client/scroll width375/375, with no document overflow.
Screenshots were emitted in the tool transcript, without standalone image-file
export. One body-focus shortcut failed; direct navigation verified the mobile
view. Browser console warnings/errors were absent. The temporary loopback server
ran210.1229524/600seconds, then stopped with exit0; tab closed/viewport reset.

All original16timeouts/2negative terminals/2goals and failed joint acquisition
remain unchanged. UI correctness establishes no circadian advantage, retained
skill or recovery. Extend by projecting a separately supported saved schema into
this builder, retaining source laws, bounds and honest unknowns. No training or
new capture framework belongs in the inspector. See ADR-0354.

## Recorded sensor evidence

The same CLI also accepts the recurring sensor reports. It builds an offline
sensor view with prearrival curves, heldout A/B checkpoints, reserved C probe
curves, signed retention, alarms, recovery nulls, memory and measured work.
The metric selector changes BCE/accuracy charts. Click recorded values to open
exact JSON pointers; the source dialog also offers the whole embedded report.
Missing evidence is unknown; recorded null stays null. Partial runs retain
available rows and their recorded status/reason. The inspector runs no learner,
inference, probe, simulation, SVD or training.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/r46-sensor-heldout-20261017/bp-result.json --output new-sensor-inspector.html
```

Optional `--resource-artifact artifacts/runs/r46-sensor-heldout-20261017/bp-resources.json`
embeds separately recorded native counters and resource observations. These old
resource files lack a shared unique run identity: the UI explicitly marks their
association with the source as unverified. It does not authenticate or silently
attribute an arbitrary resource file to this learner. Report-owned counts and
process measurements remain separate; planned protocol counts never replace
missing measured work. Both input files use the same8MiB duplicate/nonfinite
JSON refusal and exclusive16MiB output boundary. Inputs and existing outputs
are preserved.

```text
src/app/sensor_inspector_projection.py pure detached sensor projection/pointers
src/infra/sensor_inspector.html        standalone recorded charts/tables/source dialog
tests/test_sensor_inspector_*.py       projection and shared file/CLI integration
```

Why: accepted sensor and C-probe evidence already exists; reuse the shared
publication path instead of another dashboard or experiment. A single saved
report is displayed at a time. The known easy fixture has accuracy1 and no
recovery deficit; the UI preserves this finding and signed negative losses.
Inactive deployment/topology/sleep controls and missing actual work are explicit.
Full R4.7 cross-environment, structural and sleep coverage remains open.

277 focused passes, scoped static gates, six final saved-report CLI builds and
independently reviewed browser QA are recorded in [RESULTS](../artifacts/runs/r47-sensor-inspector-20261017/RESULTS.md).
The initial four test failures, typing errors and narrow chart defect are preserved. No new scientific observations are made.

### Stationary sensor controls

The same builder accepts all seven saved stationary reports, for example:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.build_key_door_inspector --source-artifact artifacts/runs/r46-stationary-sensor-20261017/native-incident.json --output new-stationary-inspector.html
```

The view identifies all-A data, absent B/return phases and at most eight dependent,
noise-free post-calibration opportunities. It preserves recorded nulls and
not_applicable recovery, partial unknown costs, scheduled optimizer steps versus
wake receipts, replay charges and incident skips. Stationary heldout/probe modes
are currently unsupported and refused. Optional separate resources may aggregate
multiple cases; their association is unverified and totals are not assigned to
this learner. Original brightness views remain unchanged. No new learner calls.
[329 scoped tests and browser evidence](../artifacts/runs/r47-stationary-inspector-20261017/RESULTS.md)
cover this compatibility slice; full R4.7 remains unfinished.

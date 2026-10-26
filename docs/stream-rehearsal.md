# Native rehearsal on a delayed stream

## Independent-data removal family

```powershell
.venv/Scripts/python.exe -m scripts.run_stream_rehearsal_pilot --describe --scenario removal-confirmation
```

`--run` executes this fixed local eight-arm family. No single-case flags are
accepted. Policy order is none/rotating; each uses BP, PC, neutral CPC and active
CPC, shared original initialization23 and fresh TRAIN7302/FINAL9102. These data
IDs were frozen before generation/scores. Original defaults remain7301/9101.
This is one initialization and an independent data pair, without significance,
generalization, equivalence or outcome-based configuration selection.

All arms use the existing delayed stream and final-only scores. None allocates
no replay component/session/storage. Rotating uses the original arrived-only
16row/384B cap/2rows every3wakes. Total192updates1200rows/64finalscorecalls4096rows,
one60sdeadline/RSS256MiB; no probes/health/SVD/promotion. Each no-replay arm has
18updates144rows, rotating30/156, so work and storage are unequal. Reports retain
all0/6/12/18 A/Bcurves, signed B gain/A forgetting/A return gain, initial/receipt
matching, perpolicy PC-neutral parity, all partial failures and permanent charges.

[Accepted evidence, all outcomes, commands and failures](../artifacts/runs/r28-removal-confirmation-20261017/RESULTS.md):
276focused passes/14CLIpreviews/13oldoutputs exact and one reviewed native family.
Rotating improves local B gain/retention, but BP return gain is lower; active CPC
gains remain below PC. Original R2.8 bounded acceptance permits an unresolved
finding: no verified minimal bundle/advantage/equivalence. Runtime full_R28_complete
stays conservative per-run metadata; canonical plan/review owns task acceptance.
[Architecture rationale](adr/ADR-0373-independent-data-replay-removal.md).

Extend by composing the existing stream and preserving separate RNG partitions,
fixed settings/metrics, final-only scoring and all failure slots. Do not tune the
next recipe from these final outcomes or start another seed/probe-horizon sweep.

## Explicit no-replay control

```powershell
.venv/Scripts/python.exe -m scripts.run_stream_rehearsal_pilot --describe --no-replay
.venv/Scripts/python.exe -m scripts.run_stream_rehearsal_pilot --describe --no-replay --diagnose-learnability
```

Replace `--describe` with `--run` to execute the described local protocol.
`--no-replay` applies only to `--scenario single` and excludes `--recent-only`.
The adapter passes `rehearsal=None` into the existing stream: no retention store,
replay helper/policy/session or consolidation event. Four original methods retain
18 wake updates each, original initialization/data/settings and finish-only
benchmark scores. Planned work is72updates/576rows, replaybytes0. Optional fixed
C probes remain separate and add256updates/2048rows (total328/2624); their health,
score and snapshot costs remain reported. No probe or final readout trains or
selects the benchmark learner.

Reports distinguish selected removal (`replay_disabled`) from successful complete
execution (`no_replay_verified`), including probe completeness. Failures retain
paid work and unexecuted arms. Shared replay-plan equality is null when no plans
exist. Existing default descriptors and calls are unchanged.

[Verification, exact commands and retained results](../artifacts/runs/r28-no-replay-control-20261017/RESULTS.md):
222 focused passes, 13 CLI previews/11 legacy outputs exact, independent review.
These tests use fake native math with the real stream. Compatible native baseline
and replay results are reused under their original source assumptions; there was
no new native wrapper run. The 36-arm historical readback is retrospective and
unequal work, with negative tradeoffs preserved; no minimal bundle, equivalence
or circadian advantage follows. Full R2.8 remains open.

Extend through the existing stream and explicit fixed protocols. Preserve the
absence of replay allocations, original defaults, independent probes and all
failure charges; do not choose settings using final scores. Architecture and
rationale: [ADR-0372](adr/ADR-0372-explicit-no-replay-control.md).

## Fixed long-stream scenario

```powershell
.venv/Scripts/python.exe -m scripts.run_stream_rehearsal_pilot --describe --scenario long-stream
.venv/Scripts/python.exe -m scripts.run_stream_rehearsal_pilot --run --scenario long-stream
```

This mode fixes64wake updates A16/B32/A16, four existing matched learners and the
same rotating policy. It rejects single-case probe/recent flags. Replay remains
every3updates/two rows with16rows/384arrayB and recent8/reservoir8 allocation.
The424-call protocol includes168replay calls, no probes/renewal. Final scoring
follows all training and replay for an arm.

Existing event records already expose retained `retention.sample_ids`, selected
IDs and actual replay receipts. A read-only `StreamRehearsal.retention` supplies
the detached current membership after a final non-event wake; existing report
schemas are unchanged. The new adapter reports endpoint64 only for complete
runs, None otherwise. Failed prefix/counters/charges remain in original reports.
Its512-ID train manifest is report-only and never passed to a learner/selector.

The [accepted long-stream result](reports/long-stream-rehearsal-20261016.md)
preserves all retention/exposure gaps and outcomes. This completes original R2.3
bounded implementation acceptance, without universal retention or learning-benefit
claims. `full_R23_complete=False` stays conservative per-run metadata; independent
criterion mapping in the plan owns repository task acceptance.

```text
src/adapters/long_stream_rehearsal_pilot.py  fixed data/native composition/reporting
src/app/stream_rehearsal.py                 existing runner helper + read-only port
scripts/run_stream_rehearsal_pilot.py       explicit long-stream CLI route
tests/test_long_stream_rehearsal_pilot.py   independent plans/ownership/failure oracles
tests/test_long_stream_rehearsal_cli.py     fixed routing/refusal behavior
```

Focused command: `python -m pytest -q tests/test_long_stream_rehearsal_pilot.py
tests/test_long_stream_rehearsal_cli.py`. Exact scoped Ruff/format/WinLinux mypy
commands and bounded native invocation are recorded in
`artifacts/runs/r23-long-stream-20261016/RESULTS.md`.
Extend by preserving the arrived-only boundary and matching all methods' exposure;
never use manifest context or final scores to improve replay selection silently.

The coherent four-method attribution stream can rehearse a shared bounded set
of arrived training rows. It uses the existing `ActorShadowRuntime.consolidate`
boundary. A private learner fork restores the detached candidate state, performs
the selected native updates and returns a complete state for commit. The source
learner and serving actor remain unchanged. An exception aborts the stream;
partial trajectories are not scored.

```powershell
.venv/Scripts/python.exe -m scripts.run_stream_rehearsal_pilot --describe
.venv/Scripts/python.exe -m scripts.run_stream_rehearsal_pilot --run
```

Add `--diagnose-learnability` to describe/run the separate fixed sixteen-step
reserved-task protocol. It attaches the existing observer at wake checkpoints
0/6/12/18, after any rehearsal at that boundary. Those native states include
replay updates too. A second independent template creates disposable probe forks;
probe updates never train the benchmark learner. The final benchmark data never
reach the observer, and benchmark scoring follows all observers in that arm.

This mode preserves the original reserved C task, data RNGs and score points
0/1/2/4/8/16. It costs376 optimizer calls:72wake+48replay+256probe, with2672labeled
row exposures,32benchmark and96probe score forwards,16feature passes and32SVDs.
The report separates wake/replay charges from probe attempts/completions, records
every curve and all failed/unreached slots, and measures outer diagnostic/probe
time. Reserved arrays use3840bytes plus512reference bytes per observer; snapshots,
activations, forks and Python metadata are additional. Native coupled importance
and neuron ages remain unavailable rather than inferred from proxy fields.

Probe endpoint signs are observations, never acceptance or selection criteria.
The retained nine negative trained PC-family sixteen-step endpoints stay intact.
This mode measures fresh-task ability alongside retention/recovery; it cannot
establish general plasticity or a circadian advantage from one fixed task/seed.

The [accepted integrated observation](reports/rehearsed-learnability-20261016.md)
retains all96probe points and unchanged benchmark reports. Nine trained PC-family
sixteen-step gains are negative, while their absolute initial/final C losses are
lower than the no-replay references. Keep both levels and gains when interpreting
fresh-task ability; the smaller gain alone does not establish plasticity loss.

## Recent-only control

Add `--recent-only` to select the explicit recent FIFO control, including in
combination with `--diagnose-learnability`. It uses the same sixteen-row/384-byte
array cap, period and two rows per event. The existing FIFO selector emits the
latest two distinct retained rows in their arrival order. Its policy has no
reservoir seed or partition allocation; partition/cursor report fields are None.
The rotating default and earlier protocols remain available without this flag.

This is a matched-exposure comparison of two policy bundles: retention
composition AND within-event order change. It does not isolate a reservoir or
rotation effect. Declared memory caps match; actual occupied rows, metadata,
process RSS and runtime can differ. Native work matches per method, while BP
and PC still have different native work. Preserve both policies' full outcomes
without tuning seeds, rates, memory quotas or the control order to select a win.

The [accepted matched-control observation](reports/recent-rehearsal-control-20261016.md)
shows a tradeoff: recent-only has lower absolute A loss, while rotation has lower
B and trained C loss. Larger forgetting/fresh-task gains use different starting
levels and do not establish an unconditional winner or better plasticity.

## Tracked learnability scenario

Use `--scenario tracked-learnability` with `--describe` or `--run` to compose
native unit histories and the existing isolated sixteen-step observer. It fixes
rotating replay and probes, so omit `--recent-only` and
`--diagnose-learnability`. Ordinary matched BP stays unchanged; PC/neutral/active
CPC use explicit tracked native classes and independent tracked probe templates.

The observer reports gradient-EMA and update-age summaries before probing,
including replay updates. At wake checkpoints0/6/12/18 the native ages are
0/10/20/30. Probes restore complete tracked state into disposable forks and never
update benchmark/source/template state. Final benchmark data remain unavailable
to the observer. All original modes keep their original descriptors and schemas.

This case costs the original376native calls/2672row exposures/128score forwards/
32SVD plus12native-history reads and282tracked history transitions; tracked
last-hidden storage is192arraybytes per width8 state, excluding metadata/copies.
Readout is in diagnostic timing and snapshot-byte accounting. Existing disk and
managed-capture schemas refuse tracked format2; no persistence migration.
See [verified results and R2.1 criterion mapping](reports/tracked-learnability-20261016.md).
This verifies instrumentation and isolation; it does not establish a learning gain.

## Paired initialization scenario

```powershell
.venv/Scripts/python.exe -m scripts.run_stream_rehearsal_pilot --describe --scenario paired-retention
.venv/Scripts/python.exe -m scripts.run_stream_rehearsal_pilot --run --scenario paired-retention
```

This fixed scenario runs original remaining initialization seeds41 and59, both
retention policies, and all four methods through the same stream helper. It uses
one shared initial parameter bundle per seed and reports all sixteen run slots
and eight within-seed/method pairs, including failures and unavailable results.
Complete pairs retain both policies' primary A/B BCE and accuracy levels at all
checkpoints as well as recent-minus-rotating derived outcomes. Known partial
work is reported separately; unobserved failed work is never filled with zero.

The planned cost is480 native updates (288wake/192replay),2496labeled-row
exposures and128score forwards. There are no fresh-task probes or feature/SVD
diagnostics. One global60second/256MiB RSS allowance bounds the entire matrix;
row wake/replay budgets remain separate. The scenario rejects single-case
`--recent-only` and `--diagnose-learnability` flags rather than changing this
protocol. All original single-case modes remain the default.

Why this: check whether the observed retention tradeoff repeats across the two
prespecified remaining initializations before selecting a mechanism. The data
remain fixed. This provides no task generalization, equivalence test, universal
winner, statistical confirmation, or circadian-advantage claim. Seed23 and all
prior negative observations remain separate development evidence. Equal caps
are not equal occupied memory or equal BP/PC computation; retention composition
and event order still change together.

The [accepted paired observation](reports/paired-retention-initializations-20261016.md)
retains all eight pairs: recent-only lowers A BCE but raises B BCE in both
initializations, with mixed accuracy. Seed23's differing BP return-gain direction
remains preserved. No policy is an unconditional winner.

## Default recent/reservoir case

The default fixed local demonstration uses BP, PC, neutral CPC and active CPC, seed23,
the existing eighteen-batch A→B→A stream and delayed labels. Every third wake
update selects two rows from a recent/reservoir buffer: at most16 unique rows,
384 float64 array bytes, two eight-row/192-byte partition quotas, seed53 and
rotating selection. Content IDs and selection are prediction/context independent.
Each method receives its own array copies of the same row proposals.

The complete case costs72 wake plus48 replay optimizer calls and624 labeled-row
exposures. Replay calls the original native training interface, advancing native
PC/CPC clocks and gates as those implementations define them. It does not
introduce a new sleep objective. Buffer array bytes exclude Python metadata,
event reports, candidate/checkpoint states, temporary copies and process RSS.
The helper caps event count, per-event rows and retention; no unlimited success
chain or seen-row ledger is introduced.

## Budgets and failures

Create the wake and replay `ToyBudgetSession` objects initially. The runner binds
the exact wake session; rehearsal requires a separate finite replay allowance.
Both sessions are checked around replay work. Why separate: the existing runtime
explicitly refuses a replay limit it does not enforce. That gate remains intact.
The adapter gives both sessions the remaining global wall allowance and the
same RSS sampler; sessions are never renewed after failure.

The entire event must fit the remaining replay budget before native admission.
One replay exposure is permanently charged immediately before each native call.
The legacy session field is named `replay_examples_completed`; this mode reports
it as **charged exposure**, distinguishing attempted, returned and committed
updates. A native exception can leave an attempted charge with no returned or
committed update. A postcommit deadline failure can leave a committed receipt
and a failed event. Reports retain these distinctions and all unreached slots.

## Evaluation and extension

Optional rehearsal runs after each newly committed wake batch and before the
corresponding checkpoint snapshot. Final-test batches never reach the helper.
All checkpoint scoring is deferred until that entire arm finishes training.
The no-replay API, schema and call order remain unchanged.

This is a small runnable integration, not a confirmation experiment. Compare
new-task B learning, old-task A retention and returning A recovery separately.
The retained no-replay runs have different exposure and compute, so descriptive
differences do not establish a replay effect. PC/neutral parity and shared replay
plans are correctness checks; they do not establish a circadian advantage.

For another policy, configure a new helper on the existing stream and freeze its
data, metrics and resource limits before native execution. Keep each helper
single-use, preserve failed charges, and require focused boundary tests. Do not
bypass the separate legacy CPC native/shared-buffer equality gate.

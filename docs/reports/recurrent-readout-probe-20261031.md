# Isolated recurrent-readout F/old-A/return probe

R2.1-RNN.RO.P / [issue394](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/394).
Software implemented and focused checks pass. At this software close, the new
native outcomes were unexecuted. A later separately frozen case is recorded in
[native switching/forgetting evidence](recurrent-readout-probe-native-20261101.md).
The full goal stays active.
Session started2026-10-31 and continued2026-11-01; stage retains its original date.

Existing #377 independently accepted both A goals for this frozen recurrent
readout. That baseline and prior R2.1/matched-family/replay evidence are reused.
This extension adds a new question through existing components, not a retry of
#390X's spent128-SGD reward comparison or a new learning algorithm.

## Working software

```powershell
python -m scripts.recurrent_readout_example --fresh-probe --describe
python -m scripts.recurrent_readout_example --fresh-probe --output new-readout-probe.json
```

The explicit option first requires both original A games to succeed and close.
The existing generic probe owns a different fork, applies two F probe_train fits,
reads F probe_test and old-A TRAIN monitors at0/1/2, then applies two explicit A
TRAIN return templates on that same fork. Two original two-update/zero-replay
budgets and one zero-update game budget are born before collection and never
renewed. The existing collector's16/2 packets are preserved; only its fixed first
two training copies fit. Complete packets remain bound outside the admitted
metadata-free raw148/K7 nine-row array bridge. TRAIN-only episode admission is
unchanged. Old affine296/old18 short-return and default16-update routes remain.

The existing native-game kernel reads F-blue/red probe_test then A-blue/red TRAIN
after each private phase, retaining false goals and separate fit versus game
outcomes. This is code behavior, not newly observed task competence. Original
source/benchmark/deployed state, data, actor, RNG, clocks, charges and prior paid
evidence remain guarded. Final data, promotion, tuning and probe selection stay
unavailable. There is no new solver, orchestrator, source loader or dependency.

Changed source: `src/app/fresh_task_probe.py`, `src/app/probe_return_phase.py`,
new `src/adapters/minigrid_readout_probe.py`, explicit option in
`scripts/recurrent_readout_example.py`, new `tests/test_recurrent_readout_probe.py`.
Usage is in `docs/recurrent-readout-policy.md`; README links it. Inherited dirty
and untracked implementations were preserved in stage `before/`, not replaced.

## Executed evidence and failures

All commands ran from physical D:\Dev\Repositories\Circadian with process-local
TEMP/TMP/TMPDIR at D:\Dev\_Temp. Stage
`artifacts/runs/recurrent-readout-probe-20261031/` retains protocol/amendments,
preimages, diffs, logs, SPENT markers and terminal receipts.

| Attempt | Cases/outcome | Body seconds | Peak RSS bytes | Host seconds/exit |
| --- | --- | ---: | ---: | --- |
| first |16PASS/7FAIL |3.0213354|87158784|3.1846797/1|
| repaired |23PASS |3.3418891|91426816|3.477857/0|
| final |111PASS |5.7282197|94867456|5.89633/0|
| kernel |1PASS |3.7023854|87453696|3.8474368/0|
| attachment |1PASS |2.8874091|86347776|3.1368841/0|
| review-fix |5PASS |5.2451624|88686592|5.3854757/0|
| baseline-window |4PASS |3.2289314|90882048|3.3939148/0|
| root-owner |5PASS |4.9274621|90202112|5.1767126/0|

There are119 distinct passing cases/173 executions:166PASS/7initialFAIL, no
test skips/errors. Total body32.0827946s. This is111 combined plus affected/new
focused checks, **not a fresh119-case combined run**. All processes recorded no
host expiry or forbidden runtime imports. The sampler starts before imports.
Literal NumPy arrays/scripted feature/fit/prediction ports and fake environments
are allowed; genuine solver/native activation/model/SGD/physics work is zero.

Initial failures were fixtures: TRAIN-only `_batch` instead of role-preserving
`_packet`, unstarted RSS sampler, and two rather than required six typed source
episodes. Correcting them did not weaken production checks. Initial four, then
one and two new owned type diagnostics are preserved and repaired with explicit
types, not suppressions. Final six-file format/lint/mypy all0/.9081062s. Types use
`--follow-imports=silent`, not repository-wide checking.

```powershell
./artifacts/runs/recurrent-readout-probe-20261031/run-check.ps1 -Attempt first -Selection recurrent_readout_probe
./artifacts/runs/recurrent-readout-probe-20261031/run-check.ps1 -Attempt repaired -Selection recurrent_readout_probe
./artifacts/runs/recurrent-readout-probe-20261031/run-check.ps1 -Attempt final
./artifacts/runs/recurrent-readout-probe-20261031/run-check.ps1 -Attempt kernel -Selection test_actual_game_kernel_reads_both_private_endpoints_and_closes_all_fake_owners
./artifacts/runs/recurrent-readout-probe-20261031/run-check.ps1 -Attempt attachment -Selection test_failed_memory_attachment_retains_the_first_original_budget_birth
./artifacts/runs/recurrent-readout-probe-20261031/run-check.ps1 -Attempt review-fix -Selection 'returned_observation_window or last_snapshot_source or adapter_routes_both_endpoints or actual_game_kernel_reads_both_private_endpoints'
./artifacts/runs/recurrent-readout-probe-20261031/run-check.ps1 -Attempt baseline-window -Selection test_firstparty_real_stream_probe_composition_preserves_source_and_checks_prior_success
./artifacts/runs/recurrent-readout-probe-20261031/run-check.ps1 -Attempt root-owner -Selection test_firstparty_real_stream_probe_composition_preserves_source_and_checks_prior_success
```

The runner retains exclusive SPENT/log/receipt identities; these commands describe
executed attempts, not permission to overwrite or rerun them.

```powershell
$files = @('src/app/fresh_task_probe.py', 'src/app/probe_return_phase.py', 'src/adapters/minigrid_readout_probe.py', 'scripts/recurrent_readout_example.py', 'tests/test_recurrent_readout_probe.py', 'artifacts/runs/recurrent-readout-probe-20261031/check_once.py')
.venv/Scripts/python.exe -B -m ruff format --check @files
.venv/Scripts/python.exe -B -m ruff check @files
.venv/Scripts/python.exe -B -m mypy --follow-imports=silent --cache-dir artifacts/runs/recurrent-readout-probe-20261031/mypy-cache @files
```

## Review and scope limits

Independent review initially required changes: returned game evidence was pinned
after callbacks. The corrected adapter pins before callbacks, restores original
returned values on refusal, preserves F2 charges and admits no A update after
tampering. Pure live-source/ref checks bracket snapshot/parent ports; late source
mutation refuses before another snapshot. Lead follow-through applied the same
repair to first-party A admission. Value flips and equal-valued replacements are
rejected; a final original-source replacement after private work keeps paid probe
results but cannot claim global preservation. Failed RSS attachment retains the
created budget's original birth. No unlimited success-record chain is retained.

Source/procedure/result self-review is separate from the finite peer's verdict
in `REVIEW.md`/`reviewer-timings.json`. Original two-read acceptance covers the
unchanged adapter/core component; a separately bounded final delta read covers
the late first-party repair. Legacy test authorship, reused old core bodies and
no numerical replication remain explicit limits. No native GO is claimed here.

Original agent engineering attempt caps were amended prospectively for specific
demonstrated integration/receipt/callback defects, with measured costs and old
SPENT attempts preserved. Final eight pytest/nine type attempts still hold the
original180-execution/90s body/512MiB and120s types/32MiB cache totals. Stage64MiB
and logs1MiB remain bounded. These are engineering amendments, not renewed
scientific allowances. All #390X FAILED/SPENT, #391 PARTIAL, #392 authentication/
browser gaps and earlier negative findings remain unchanged.

Skipped full suite, Linux, clean clone, wheel/install, genuine learning/games,
browser/device, GPU/remote CI and release. No download/publication/deletion or
old scientific JSON/parameter reconstruction occurred. Nothing here establishes
native F acquisition, retention, recovery, reliability, generalization,
matched-compute benefit or a circadian advantage.

Exact next action after software acceptance: inspect the existing readout native
producer, then freeze/review **one** complete first-party `--fresh-probe` local
case, accounting for fresh source fits plus F/return fits, all diagnostic/prefix
work, collection, ten game episodes, memory/storage/CPU and failures. Forecast
six fits54rows/four F teacher owners plus ten game owners is not an allowance.
Use unchanged seed17/metrics/all fixed cues, preserve all failures and cheaper
accepted affine controls; do not retry #390X or start a capture/wheel/audit chain.

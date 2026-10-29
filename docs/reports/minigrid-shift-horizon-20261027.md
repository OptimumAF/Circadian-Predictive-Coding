# Bounded MiniGrid shift-observation horizon

[#232 / R2.7-MG.H](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/232)
delivers scoped software through the existing adapter and stream. Source,
procedure and results were self-reviewed; this is not independent acceptance.
Full repository, R2.7 deployability and reliable adaptation requirements remain
open. Historical plans/log retain evidence; live handoff belongs in GitHub.

## Working behavior

`MiniGridShiftDiagnostics(max_arrivals=12)` observes a longer finite TRAIN
schedule through the existing protected `run_reference_window` callback. The
keyword-only count must be an exact integer in1..64, validated before allocating
a reader or scalar slots. `describe(max_arrivals=...)` reports the same count.
Reports have exactly that many detached slots; an arrival beyond the bound is
refused before another prediction. One reader is retained, with no success-state
history. The six-arrival default keeps its original report schema and values.

Why this: the adapter previously rejected index6 regardless of the stream's
schedule. Default three-batch reference/two-batch window/two-decision persistence
cannot produce an onset before6, so the original B3–4 phase cannot be detected
in-phase. A longer explicitly declared schedule can provide a clean reference
and monitoring opportunities before context return. The option changes capacity,
not thresholds, reference selection, persistence, feature summaries or policies.

Changed files:

- `src/adapters/minigrid_shift_diagnostics.py`: validate count, allocate bounded
  slots, enforce count and publish configured capacity/storage scope.
- `tests/test_minigrid_shift_diagnostics.py`:16 new behavioral cases alongside
  the original31 in this file.
- `docs/attribution-reference-window.md`: existing API and longer-adapter usage.

The ordinary fixed-six API/CLI is unchanged. There is no new runner, callback,
budget, controller feedback, fitting algorithm, dependency or environment setting.
The existing README/architecture links still point to this adapter's usage.

## Executed checks

Artifacts: `artifacts/runs/minigrid-shift-horizon-20261027/`. Original dirty source
and tests were copied into `preedit/`; `combined-source.patch` shows the actual
delta. `PROTOCOL.md` freezes the software scope and resource limits before work.

Windows and cached Linux each passed47 distinct cases (16new/31original), then
four affected cases on the final combined tree:51 instances/platform, zero test
failures/errors/skips. The four repeats cover explicit keyword refusal, both
twelve-arrival modes and late failure after a small test-call/type correction.
The adapter implementation stayed unchanged after the initial full-file run.

The twelve-arrival no_learning and periodic streams match observer-disabled
controls' decisions, applied receipts and final candidate state. Source, actor,
training inputs and Python/NumPy RNG remain unchanged. Original budget object,
limits, clock, sampler and start identity remain binding; no replay is performed.
no_learning observes all12 packets with0 updates and12 pending packets; periodic
performs6 scripted updates. Full296-feature readonly K7 predictions occur before
each arrival's choice/fit; only current nine TRAIN rows enter the diagnostics.

Invalid counts refuse before reader construction. Counts1/12/64 allocate exactly
the declared slots and detached report mutation cannot enlarge live storage.
An injected lost eighth prediction retains7 complete records,1 failed record,
4 not_run slots, attempted8/returned7 predictions and unknown-return accounting,
with0 updates in that case. Beyond-bound input refuses before additional work.
Existing mutation, clock, metric-failure, joint-row exclusion and default API
checks also pass. These are scripted learner fixtures plus real pure statistics;
they are not learned policy performance or native detector evidence.

| All attempts per platform | Windows | Linux |
| --- | ---: | ---: |
| Body seconds | 6.3540709000 | 13.9307474360 |
| Parent seconds | 9.5508645000 | 41.0180037997 |
| Sampled peak RSS, bytes | 79,839,232 | 74,870,784 |
| Scripted owners / snapshots | 150 / 2,183 | 150 / 2,183 |
| Scripted fits / rows | 42 / 432 | 42 / 432 |
| Scripted predictions / rows | 110 / 1,062 | 110 / 1,062 |
| Arrival callbacks / K7 metric calls | 110 / 93 | 110 / 93 |

Original caps: CPU1,120s cumulative body/180s parent/512MiB RSS after imports;
400 owners/12,000 snapshots/160 scripted fits3,000rows/240 predictions3,000rows/
240 callbacks/240 metrics. All held. RSS does not cover initial imports or a
process tree. Constructor/fit/predict on the genuine Native class and LAPACK
entry are patched to refuse. Genuine learned work, random model initialization,
gradient/latent/teacher, physics, source payload reads, private/probe/final data,
and old scientific raw artifact reads are0. No optional simulator/Torch import.

The finite qualifier and supervisor reuse the existing ordinary pytest procedure.
Initial Windows command:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/minigrid-shift-horizon-20261027/supervise.py win-initial .venv/Scripts/python.exe -B artifacts/runs/minigrid-shift-horizon-20261027/qualify.py win-initial
```

Linux uses the same supervisor with `wsl -d Ubuntu -- env` and CPU1 variables,
`timeout 120s`, cached Python
`/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python`,
and the same qualifier. Exact commands, final test selectors and terminal stdout
are in `{win,linux}-{initial,final}-parent.json`; per-test outcomes in XML/logs.
All receipts are exclusive writes: do not repeat against their existing names.

Four-file Ruff format/check and default/Linux-target mypy pass. Initial mypy
reported4 test-only errors: intentional positional-invalid call, callback list
annotation and optional failed-report narrowing. The failed receipt is preserved
in `statics.json`. Explicit `Any` only at the intentionally invalid runtime call,
an annotated callback list and a non-None assertion repair these without ignores.
`statics-final.json` records all four checks EXIT0. Total static/format time over
all attempts is26.0787179002s of120s. One inherited cache file111,280,128B was
mtime-selected, below160MiB; this is touched storage, not growth or two separate
copies. No new cache tree was created inside the study stage.

For ordinary development use `python -B -m pytest -q
tests/test_minigrid_shift_diagnostics.py`, and `python -m ruff check`,
`python -m ruff format --check`, `python -m mypy --explicit-package-bases
--follow-imports=silent` on the changed source/test files. Add `--platform linux`
for the second type target. The executed finite versions also cover the two
stage helpers, with exact commands and times in the static receipts.

## Review, preserved evidence and next action

Self-review inspected the source delta, existing stream admission/receipt path,
actual fixtures, counters, failure states and terminal results. New behavior is
limited to explicit horizon/storage; original core stream, metrics, detector,
learner, solver, native producer, API and CLI sources are unchanged. Default
reports remain compatible; #230 evidence is reused for unchanged boundaries,
with its affected31 tests re-executed here. Source pins for the adapter naturally
change in this increment; frozen historical pins/qualification are not refreshed.

Full suite, clean clone, wheel/install, UI, remote CI/GPU, native execution,
comparative experiment and independent review were skipped. No scientific
allowance was created or renewed. #231's native attempt remains failed/spent
with all signal outcomes unknown; its budget correction is still an unapplied
proposal. #227 and every earlier failure/negative result remain preserved. No
claims about learning, retention/recovery, generalization, efficiency, CPC benefit
or repository/release readiness follow from these passing fixtures.

Next precise action: scope one small **data-only observability diagnosis** using
the existing `public_feature_summary` on the four original A/B public TRAIN
episodes. Freeze a single budgeted source parse, four summaries and scalar
separation/reachability metrics before reading the payload. Determine whether
the eight channel-mean/direction features distinguish the layout change at all
before planning another learned study. Use no learner construction, prediction,
fit, solver, physics, threshold tuning, final/probe data or native231 retry.
Why this: longer monitoring cannot repair a summary that discards the relevant
spatial information. This directly tests a concrete observable-signal dependency
instead of adding another API, package, capture or audit increment. R2.1's
original bounded diagnostics and isolated probe are already accepted; reuse them
and the matched BP/PC/neutral/active evidence rather than reconstructing it.

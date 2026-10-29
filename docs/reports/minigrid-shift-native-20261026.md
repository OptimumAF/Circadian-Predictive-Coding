# Native MiniGrid shift observation: failed before stream admission

Live task: [#231 / R2.7-MG.N](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/231).
Parents R2.7/#21, #23 and #33 remain open. The full repository goal is incomplete.

## Observed outcome

The sole authorized four-case attempt failed before fitting or prediction.
`ActorShadowRuntime` rejects any non-None replay limit, including the producer's
explicit `max_replay_examples=0`. This is a producer admission error, **not an
empirical shift-detector result**. No native retry was made or qualified.

The stationary/no_learning case failed at stream construction; the remaining
three cases were not run. All false-onset, B-detection, return-detection and
latency outcomes are **unknown**, not zero or misses. Final source/actor/RNG and
budget-binding success assertions were not reached. Their preservation must not
be inferred from the absence of fitting. The failure retains its original
producer, protocol, qualification, launch marker and terminal receipts.

The declared pre-run structural limitation remains valid independently of this
failure: the default detector uses reference batches1–3, a two-batch window and
two persistent monitoring decisions. Its earliest possible onset is batch6.
With original B at batches3–4, in-phase B detection is impossible and the
reference includes the first B packet. This was disclosed before execution;
it is not an observed native miss or grounds for post-hoc threshold changes.

## Execution and permanent costs

All evidence is under
`artifacts/runs/minigrid-shift-native-20261026/`.

```powershell
.venv/Scripts/python.exe -B artifacts/runs/minigrid-shift-native-20261026/execute.py
```

One invocation, outer EXIT1 / checked child EXIT3. `native-parent.json` records
`failed-spent`, 1.6325858999s. `native-work.json` records 1.2672832999s and
55,971,840B sampled peak RSS, covering before optional imports through scalar
serialization/write and sampler exit. Final small work receipt is outside RSS
sampling and inside parent time. Caps were90s body/120s parent/512MiB RSS.

Known attempted/returned work is identical:2 empty native owners,1 snapshot,
0 fits/fit rows,0 LAPACK solves/solve rows,0 predictions/prediction rows,
0 arrival callbacks and0 metric calls. There is no outstanding native work.
No physics, teacher, probe, final evaluation or model random draw was executed.

One native public TRAIN read of94,705B decoded6 episodes/54rows. Together with
the pre-GO opaque hash read, permanently charged source access is2 reads/189,410B.
Resource SHA256 is
`5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508`.
Only four original A/B episode IDs were selected; no raw model artifact was
written. `terminal.json` is a3,568B explicit scalar projection.

## Passing software evidence and limitations

Unchanged #230's32 distinct literal checks/platform are reused at their original
scope; they establish software boundaries, not native usefulness. Before GO,
`check_protocol.py` passed8 pure checks on Windows and cached Linux. Windows
body/parent2.0276003000/2.4349297001s, peak RSS63,680,512B; Linux
2.674016848/8.18162319995s,63,688,704B. Each used12 synthetic packets and no
models, budgets or source payloads. Exact commands are in
`protocol-win-parent.json` and `protocol-linux-parent.json`.

Four producer helpers passed final Ruff formatting/checks and default/Linux
target mypy. An initial24 typing diagnostics were repaired without ignores;
the failed receipt remains. All original static attempts have a conservative
74.9697484967s upper bound within120s. The initial asynchronous mypy duration
is an upper bound including intervening work, not an isolated runtime. The
source/procedure self-review missed the native budget admission prerequisite;
the pure checks did not construct budgets by design. This limits that review.

After the failure, a prospective engineering-only amendment was posted in
[comment6432500084](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/231#issuecomment-6432500084).
It authorizes4 literal admission checks/platform,30s body/60s parent/256MiB RSS,
32 scripted owners/512 snapshots/24 scripted updates/216 synthetic rows;
no genuine native work or new scientific allowance. The original pure-check
budget was not silently reused for budget construction.

`proposed-budget-fix.patch` removes only the explicit unsupported replay limit.
It is **not applied** to the spent producer. `check_budget_admission.py` verifies
that exact diff, compiles only the original/corrected budget expression and
executes the real stream with scripted learners. Four checks/platform passed:

- The original zero replay limit is refused before updates.
- Corrected no_learning completes six readonly arrivals with0 updates/replay.
- Corrected periodic completes six arrivals with3 scripted updates/27rows and
  no replay. Original limit/session/clock/sampler binding remains unchanged;
  detached callback input mutation cannot change the live training packet.
- Six separately recorded synthetic update charges exhaust the original update
  cap; a seventh attempt is refused without renewal.

Windows body/parent0.3460196999/0.6147011998s, peak RSS48,148,480B; Linux
0.92756554/5.4126887000s,49,991,680B. Each platform:7 scripted owners,
161 snapshots,3 scripted fits/27rows,12 callbacks and6 explicit synthetic budget
charges. Genuine native constructors and solver entry are patched to refuse;
no payload reads, native predictions, physics or experiment relaunch occurred.
These checks qualify a proposed admission correction, **not the full native
producer or a successful scientific study**. They do not prove that an unset
limit generally enforces zero replay; this specific stream path executes none.

```powershell
.venv/Scripts/python.exe -B artifacts/runs/minigrid-shift-native-20261026/check_budget_admission.py
```

The above writes a receipt exclusively; do not repeat against existing receipts.
The exact Windows and Linux invocations, CPU environment, finite parent timeout,
stdout and outcomes are in `admission-{win,linux}-parent.json`. Cached Linux
uses `/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python`.

Final one-file Ruff format/check and default/Linux-target mypy all EXIT0;
exact commands/times are in `admission-statics.json`, total10.3263944001s.
Initial formatting plus patch-file generation has a conservative1s upper bound
(enclosing tool invocation reported0.6s). New type caches total28,131,786B below32MiB. They were
initially created inside the stage (temporary total28,190,577B) then moved,
without changing evidence, to
`.mypy_cache/minigrid-shift-native-admission-20261026/`. This keeps these
engineering caches separate from the original10MiB study artifact allowance.
`admission-postflight.json` preserves that initial measurement. All23 original
qualification source/protocol pins still match; none was refreshed.

Source/procedure/result review is **self-review**, not independent acceptance.
Full suite, clean clone, new package/install, UI, remote CI/GPU, native Linux,
scientific retry and independent review were skipped. No production source,
historical plan/log, baseline, metric, threshold, seed or unrelated dirty file
was changed by this failure repair. Earlier #227 and other failed/spent runs,
and all negative BP/PC/neutral CPC/active CPC evidence, remain unchanged.

## Critical path and exact next action

The original bounded R2.1 diagnostics/isolated fresh-task probe implementation
already exists and was accepted (including the tracked observer criterion map
at `artifacts/runs/r21-tracked-observer-20261016/RESULTS.md`). Reimplementing it
or repeating matched baselines would not resolve this gap. Existing native
fresh-task/return results remain descriptive, with joint retention unresolved;
this admission check supplies no new acquisition, retention or recovery claim.

#231 remains open with the native/scientific criteria unmet. Its one attempt is
spent, not permission for a renamed repeat. The supported-budget patch is
retained as an unlaunched proposal for a separately justified future protocol.

Next implementation action: scope an opt-in **bounded arrival count** in the
existing `MiniGridShiftDiagnostics` adapter, preserving the six-arrival default,
so an independently frozen stream can contain an uncontaminated reference and
enough monitoring arrivals within a new context before it returns. The existing
stream already handles finite schedules; the adapter hard-codes six slots and
rejects index6. Verify scalar-storage bounds, original budget/callback isolation
and unchanged defaults with scripted learners first. This addresses a concrete
timing dependency of useful shift observations; it adds no new orchestration,
capture framework or threshold tuning. Do not launch a native experiment as
part of that software increment. R2.7 deployability, reliable adaptation, CPC
advantage and repository/release readiness remain open.

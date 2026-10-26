# Owned-checkpoint key/door learnability

The existing CLI also accepts optional
[decision/class diagnostics](probe-decision-diagnostics.md). Its original
aggregate-only default remains unchanged. The opt-in retains ordered reserved
episode decisions, class support and confusion from the same predictions; it
does not add native updates or predictions.

The existing attribution CLI can measure reserved fresh-task fit at fixed stream
checkpoints using complete native state and disposable learner forks:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --describe --scenario torch-key-door-learnability
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-learnability
```

Preview the diagnostic variant before executing a separately budgeted case:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --describe --scenario torch-key-door-learnability --probe-decision-diagnostics
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-learnability --probe-decision-diagnostics
```

`--probe-decision-diagnostics` is valid only for this scenario. Python callers
can use `run_learnability_pilot(decision_diagnostics=True)` or the corresponding
descriptor option; both require an exact bool. The selected protocol has ID
`actual_key_door_owned_checkpoint_decisions_v1`. It retains original scientific
metrics, seeds, data, checkpoints and native work bounds. The helper forwards
the flag to all four probes, verifies the requested context maps are present,
and preserves charged failed reports before refusing dependent work.
Python report/temporary NumPy allocations add time and memory; input-array byte
counts exclude them, while total invocation time and sampled RSS include them.
This is no claim of zero diagnostic overhead or useful adaptation.

The first [frozen diagnostic measurement](reports/key-door-decisions-20261016.md)
matches all original results and exposes all-left actions at every reserved
endpoint. The earlier aggregate-only record remains preserved; every class and
intermediate decision is retained in the new report.

The scenario runs only recurrent behavior cloning, replay off/on. Original
`torch-key-door` and `torch-key-door-recurrent` routes keep their own behavior.
There is no new dependency, configuration variable or stream runner.

## Fixed question and isolation

At wake checkpoints 0 and 8, restore the complete owned observer checkpoint on a
disposable fork using existing `run_fresh_task_probe`. This includes original
optimizer momentum, gradients and counters. The four cohorts each receive a
separate original 16-update budget, with scores at 0/4/8/16 updates. Source,
checkpoint, serving actor and original benchmark budgets must remain unchanged.
The helper checks immutable probe session/limits/clock/progress/start bindings,
exact integer charges and the observed monotonic clock. It preserves failed work,
primary and secondary errors and all unreached slots; failure stops that arm.

Collect 16 reserved E training episodes, alternating cues, and two E test episodes
at rotation 3/branch 3/swapped keys/appearance 0.75. All are complete eight-step
teacher trajectories, with explicit probe roles, unique IDs and disjoint arrays.
Training/test values share deterministic traces: this measures supervised fit,
**not unseen-trajectory generalization**. Benchmark A/B/C/D final records never
enter probe training, selection, promotion or tuning. No probe work is deployed.
Within each isolated arm, original benchmark final scoring/live evaluation occurs
only after all benchmark updates and both probes. Scores never cross into training.

Metrics are paired E mean CE at update 0 minus CE at update 16, plus both endpoint
accuracies. Preserve all curves and negative values; no desired-quality gate.
The seed 17, D8/H8/K5, rate .03/momentum .9, eight ordinary teacher episodes,
original replay selection/32-transition permanent charge and 16-transition/4032B
retention cap are fixed. Replaying the existing recurrent prefix is necessary to
observe its opaque state in process; parameter-only old JSON cannot restore it.
Old BP/PC/neutral CPC/active CPC experiments remain accepted references, unrerun.

## Work and memory

One prospectively frozen local case: 84 SGD calls/672 training rows, including
64 probe calls/512 rows; 112 classification calls/896 rows, including 32 probe
calls/256 rows. Eighty fresh live sessions, at most 960 live forwards, at most
2528 recurrent timesteps, 34 teacher episodes/272 steps. One CPU thread in each
runtime, GPU 0, 90-second body/120-second parent, sampled process RSS <=768MiB,
JSON <=8MiB. No extra probes, renewal, SVD, backbone, sweep or post-result update.

Each probe reports detached feature/target payload bytes and complete episode
IDs/transition counts. Episode records, Python objects, native/copy state and
temporary graphs are excluded from logical bytes; sampled full-process RSS is
reported separately. Slot time includes input/state detachment and probe execution, ending before
final isolation readback. The overall invocation includes that readback cost.
Benchmark parameter, optimizer, gradient and activation tensor bytes stay separate.
Partial native work may require the execution receipt; it is never zero-imputed.

## Structure and focused checks

```text
src/adapters/key_door_learnability.py       bounded slots, probe composition, isolation
src/adapters/key_door_pilot.py              existing stream observer and recipe
scripts/run_coupled_attribution.py          explicit scenario selection
src/app/fresh_task_probe.py                 reused disposable fork and metrics
tests/test_key_door_learnability_pilot.py  reserved packets, charges, failures, CLI
tests/test_checkpoint_probe_decision_route.py  direct opt-in and refusal behavior
tests/test_key_door_decision_route.py      unchanged defaults and four-probe CLI path
```

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_checkpoint_probe_decision_route.py tests/test_key_door_decision_route.py
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_key_door_learnability_pilot.py tests/test_recurrent_key_door_pilot.py tests/test_key_door_pilot.py tests/test_categorical_fresh_task_probe.py tests/test_fresh_task_probe.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/adapters/key_door_learnability.py src/adapters/key_door_pilot.py scripts/run_coupled_attribution.py tests/test_key_door_learnability_pilot.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/adapters/key_door_learnability.py src/adapters/key_door_pilot.py scripts/run_coupled_attribution.py tests/test_key_door_learnability_pilot.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/key_door_learnability.py src/adapters/key_door_pilot.py scripts/run_coupled_attribution.py tests/test_key_door_learnability_pilot.py
```

Use the same type command with `--platform linux`. Reuse accepted evidence when
source/assumptions are unchanged. The scientific case is separately budgeted;
repeating the command is another experiment, not free validation.

Why this: a small failed exposure is not proof of lost learnability. This
composition measures reserved fit without extending the final-scored benchmark
learner. Keep future algorithms behind the existing learner/stream ports; any new
scientific protocol needs independent review before execution. Separate recurrent
RL is now implemented and evaluated through its own route; useful navigation and
full R4.2 remain open.

[Prospective scope and artifacts](../artifacts/runs/r21-key-door-learnability-20261016/SCOPE.md)
and [ADR-0329](adr/ADR-0329-probe-owned-key-door-checkpoints.md).

## Accepted original aggregate-only evidence

116 distinct focused cases qualify via114unchanged passes plus two repaired
fault-injection fixtures. The original failed attempt is preserved. Final four-path
Ruff/format/Windows+Linux type checks and independent source/GO/ACTUAL accept.
The sole frozen native case completes84SGD/1738unrolled timesteps in15.6646296s,
with313487360B sampledRSS. All13frozen source/controller texts and every original
benchmark field match accepted prior evidence, except additive probe work fields.

All four E curves reduce CE and reach50% accuracy, equal to a training-label
majority reference. Its empirical-prior reference also has lower CE. These are
explicitly posthoc descriptive references, not new tuned or registered matched
comparisons. All80benchmark episodes remain illegal/zero success/return. No
useful navigation, learned cue memory or circadian advantage is established.
[Results, failures and exact commands](../artifacts/runs/r21-key-door-learnability-20261016/RESULTS.md),
[independent review](../artifacts/runs/r21-key-door-learnability-20261016/REVIEW.md),
[every curve, cost and next action](reports/key-door-learnability-20261016.md).

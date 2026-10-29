# R7.1-F — combined rate and gate TRAIN actions

[Issue176](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/176)
qualifies one bounded wake action that applies both existing native effects in
one staged update. This is implemented and executed functional behavior, reviewed
by the implementing agent. It is not independent acceptance or evidence of
learning utility, retention, recovery, generalization, efficiency or CPC benefit.

Why this: the existing private native update already accepted both scalars. A
narrow optional public port exposes that operation through the existing causal
inbox and shared TRAIN admission without performing two updates or adding a runner.

## Implementation and boundaries

- `src/core/learner_ports.py`: optional `RateGateControlledLearner` capability.
- `src/adapters/torch_feature_learner.py`: validate both effects; delegate original
  public methods for unity/unchanged, otherwise stage the existing joint operation once.
- `src/app/{learner_step,experience_inbox,resource_sharing}.py`: joint capability
  refusal before payload/resource/budget callbacks, original admission and receipts.
- `scripts/ticket_workflow_native.py`: existing counted learner observes the joint
  operation exactly once, including degenerate delegation paths.
- `tests/test_combined_rate_gate_actions.py`: 29 new behavior checks; the bounded
  qualification selects 17 affected existing pure checks as well.

BP/PC still refuse nondefault gates; neutral CPC refuses active. Replay/sleep
combinations remain unsupported on this path. Base policy/config, optimizer LR,
momentum, chemistry, state formats, source/actor isolation and RNG behavior remain
native. Lost returns stop the inbox and remain paid/unknown; late stops preserve
completed receipts. No final/probe/training/promotion authority is added.

## Executed checks

Stage: `artifacts/runs/combined-rate-gate-20261025/`. Frozen `PROTOCOL.md`,
pre-execution and result `REVIEW.md`, commands/outputs/JSON/XML receipts retained.

| Scoped correctness | Windows | Cached WSL Linux |
| --- | ---: | ---: |
| Tests passed / failed / skipped | 46 / 0 / 0 | 46 / 0 / 0 |
| Body seconds | 9.404927600 | 10.752831837 |
| Parent seconds | 10.2497514 | 16.3151832 |
| Sampled RSS bytes | 344,834,048 | 362,770,432 |

Each platform performed 31 physical native head updates / 62 TRAIN rows,
54 latent iterations, 4 SGD/backward calls and 4 TRAIN forwards / 8 rows,
including direct references and deliberately failed staged work. There were
123 head constructor entries / 2,091 initializer-slot upper bound (not exact
random draws), 128 snapshots / 468,737 typed scalars and 5,384,341 JSON bytes
plus the same counting UTF-8 bytes. No evaluation, teacher, physics, encoder,
probe, final, replay, sleep or promotion work. Global RNG equality checked.
Software RSS sampling covers test bodies after runtime imports; it does not
establish import-time or process-tree peak memory.

Exact cohort commands:

```console
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/combined-rate-gate-20261025/qualify.py windows-1
wsl -d Ubuntu -- env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= timeout 140s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/combined-rate-gate-20261025/qualify.py linux-1
```

Both cohorts passed on their first attempt. Full-state default, rate-only,
gate-only, direct joint native math, active/default equivalence, failed-stage
atomicity and counted-fork observers passed. Literal regressions cover roles,
future labels, pause/serving/resource deferrals, unsupported capabilities,
quota exhaustion, lost returns and completed work followed by a late stop.

Final nine-file Ruff, format and default/Linux-target mypy passed; case-only
checks also passed after moving its existing RSS sampler before runtime imports.
Exact argument arrays and times are in `statics-final.json` and
`case-statics-final.json`. Initial mypy failed with two optional-runtime closure
errors; an explicit non-None assertion repaired them without suppressions.
An initial post-format patch context miss wrote nothing; narrow patches succeeded.
The failed static receipts remain. No production/test source changed after the
passing cohorts. Docs changed during closeout; whitespace validation recorded
in `final-metadata.json`.

## Sole functional native example — complete, allowance spent

After software gates and self-review GO, one cold literal gated-CPC source
(seed17, 2 features, hidden3, K2, base LR .03) entered the original shared runtime.

```console
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/combined-rate-gate-20261025/case.py
```

It exited0: body2.150851200s, parent2.690791000s, sampled RSS226,852,864bytes
including NumPy/Torch imports. Three native attempts/returns, six TRAIN rows and
six latent iterations; calls were (.015, neutral), (.015, active), then
(.03, unchanged). Each had one admission, native update, receipt and budget charge.
Source and actor complete states and Python/NumPy/Torch RNG stayed unchanged.
Candidate finished3updates/6rows under the original policy/base LR/schema.

Observed clocks1/4/7 and arrived=applied3/6/9 preserve chronology. Future labels,
one paused and one serving poll deferred; seven decisions yielded six returns
and one quota refusal. Original three-update/60second/512MiB budget, clock,
sampler, immutable envelope and cumulative quotas retained their identities,
starts and permanent charges. All replay/sleep counts zero, unknown work false.
Three diagnostic losses (.1250352561, .1250325292, .1250298172) are retained;
they are not task utility or comparative evidence and selected no settings.

`actual.json` contains scalar polls, receipts, policy and preservation proofs,
not weights. The same-process terminal result was consumed; no extra raw reader
or historical model/data source was used. Native allowance is spent, no retry.
All prospective time/RSS/work/storage limits held; final metadata records sizes
and static/cache costs without reopening raw native evidence.

## Limits and next action

Full suite, clean-clone/current-wheel install, browser, physics, independent
review and comparative/utility experiments were skipped because this increment
changes a narrow native port/admission composition. No dependency/install/GPU/CI,
commit/push/reset/delete/subagent work. Prior failures and negative comparisons
remain valid. Earlier #174 wheel predates these changes and #175 viewer remains
open for actual UI checks under #154's browser restriction.

Full R7.1/R7.5/R7.7, R2/R4/R5, scientific and release parents stay open.
Reuse completed fresh-task diagnostics and A4 old/new/return evidence; do not
repeat them or automatically add another action combination. The live tracker
records one next substantive source-level action against the remaining roadmap.
Historical plans/log remain frozen under AGENTS.md's issue workflow.

Extend only through the existing optional native ports and original causal
admission; preserve one-update accounting and reject unsupported compositions.

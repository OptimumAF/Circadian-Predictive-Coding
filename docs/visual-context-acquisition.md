# Native visual context acquisition and return

One fixed local case now connects the public RGB teacher, the ordinary
`ActorShadowRuntime`, the actual stable serving actor and detached candidate
readout to the existing PickCube smoke. Production code is unchanged.

## Files and responsibilities

```text
artifacts/runs/visual-context-return-20261019/
  PROTOCOL.md       Frozen sequence, definitions, ownership and resource limits
  native_case.py    One fixed case composing existing public runtime/CLI APIs
  native_guard.py   Reused Linux process/time/storage guard
  RESULTS.md        Commands, outcomes, costs, limitations and next action
  REVIEW.md         Independent source, gate and actual-data acceptance
tests/test_visual_native_context_case.py  Fake IO with real NN/runtime objects
```

This is an executable experiment case, with no new simulator runner, learner,
dependency, checkpoint codec or generic orchestration framework. No pretrained
weights or new environment variables are needed. The accepted optional Linux
simulator environment and existing fixed visual recipe are reused.

## Fixed sequence and observed result

Reuse accepted A17 teacher/reference; collect B19 once before fitting anything.
Fit A observations once and append all A pairs once. Create the existing stable
actor/private candidate runtime. Run B with the actual unchanged actor. Record
B features at logical tick0, withhold labels until2, and append the full B packet
once under a one-update budget. Read an owned candidate snapshot through the
existing frozen learner path on B, then fresh A. No promotion occurs.

| Endpoint | Owner | Steps | Sparse return | Final success | Ending |
| --- | --- | ---: | ---: | --- | --- |
| A reference, reused | Accepted A-only learner |29 |1 |true |terminated |
| B before update | Actual A-only stable actor |50 |0 |false |truncated |
| B after update | Detached private candidate |30 |1 |true |terminated |
| Fresh A return | Detached private candidate |29 |1 |true |terminated |

The B teacher qualified with30 steps. Both candidate rollouts exactly reproduce
their complete teacher trajectories, including all789D observations/actions/end
flags. B signed return/any/final improvements are all+1. A reference minus return
retention losses are all0. Initial visual observations, frozen A statistics,
retained A prototype prefix and all source/actor/candidate states were checked.
Independent readback recomputed all109 native predictions from saved arrays.

The candidate learns a second reset context through stored nearest memory.
Serving remains A-only; it has not acquired B, and no deployed policy improved.
These are the same PickCube task and camera, with explicit public proprioception
and goal coordinates alongside pooled RGB. This does not establish a distinct
task, neural plasticity, visual necessity/generalization, camera/object/dynamics
domain drift, matched BP/PC/CPC benefit or a circadian advantage. The failed B
baseline and all earlier negative results remain intact.

## Isolation, costs and limits

Candidate rows29/29/59 and charges0/0/1 at ticks0/1/2; one empty post-update poll
does not renew the charge. A-only source and serving actor remain publication1;
candidate is publication2. No final/probe files, SGD, replay training, refit,
eviction or promotion. All teacher qualification, source/model and runtime
completion gates precede further work; complete negative task outcomes are valid.

The four new episodes use139 controls/695 known controller physics steps and143
visual projections. One fit reads29rows/22,881values with12,624numeric statistics
bytes. Two appends ingest59pairs in total. Prototype payload grows183,976→374,296
bytes (+190,320); one four-array NPZ is559,342file bytes.109prediction rows use
3,488raw output bytes,4,931distance candidates/3,890,559coordinates, and111
normalizations/168rows.143frames represent7,028,736raw input bytes and112,827
returned feature values/902,616numeric feature bytes. No rawframe archive.

Guarded runtime44.656s/launcher47.311s stayed within90s/105s and the four-episode
200control/1000physics limits. Observed highwater sum1,144,291,328bytes and final
CLI self peak1,081,245,696bytes stayed within the declared2GiB observation limits.
Sampled observations cannot prove whole-lifetime process-tree memory. SDK hidden
constructor/reset physics/renders, allocation overhead and snapshot-copy work
are not fully counted by numeric payload/distance counters. Wall time includes
their execution; the reused A acquisition costs remain in prior evidence.

## Verification and safe extension

55 new cases pass. Final Windows459 pass; Linux55new+404boundary cases pass on
the same unchanged final tree, with exact matching testcase-ID union. Zero
failures/errors/skips; four-file lint/format and two-source types pass. Independent
source/gate admission preceded the single native case; actual-data review is
recorded separately. Full-suite/clean-clone/release gates remain unrun.

```powershell
python -m pytest -q -p no:cacheprovider tests/test_visual_native_context_case.py tests/test_stable_actor_robot_boundary.py tests/test_maniskill_rgb_observations.py tests/test_continuous_stream_demo.py
python -m ruff check artifacts/runs/visual-context-return-20261019/native_case.py artifacts/runs/visual-context-return-20261019/native_guard.py tests/test_visual_native_context_case.py
python -m ruff format --check artifacts/runs/visual-context-return-20261019/native_case.py artifacts/runs/visual-context-return-20261019/native_guard.py tests/test_visual_native_context_case.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux artifacts/runs/visual-context-return-20261019/native_case.py artifacts/runs/visual-context-return-20261019/native_guard.py
```

The one-use native command is preserved in PROTOCOL/RESULTS; existing evidence
must not be overwritten or a new trial silently admitted. The next functional
gap is serving publication: the existing smoke explicitly refuses PromotableActor,
whose public serving_snapshot includes generation/configuration/cache/metadata.
Extend that existing boundary with frozen version/generation guards and actual
public predictions, using the existing promotion controller. Do not substitute a
copied learner or broadly accept subclasses. First verify committed candidates
change subsequent controls and mid-rollout commits refuse in fake IO; any new
native promotion or domain-change study needs a separate frozen protocol.

[Full evidence and exact commands](../artifacts/runs/visual-context-return-20261019/RESULTS.md).

# Private learning from provisional robot corrections

The existing robot rollout and actor/shadow runtime now compose into a fixed
TRAIN collection → private update → frozen rollout recipe. The original source
learner and serving weights remain unchanged. The source policy keeps its
default-MSE action objective; PC uses its native energy update. Existing model
hyperparameters, normalization and momentum are preserved.
It does not require a default-size policy for fake-boundary tests; the scientific
visual protocol separately fixes789 features, hidden64 and the original settings.

## Programmatic use

```python
from pathlib import Path
from scripts.maniskill_continuous_smoke import (
    run_frozen_rollout, run_corrective_adaptation,
)
from src.adapters.continuous_correction_demo import run_continuous_correction_update

# A separately budgeted collection from an owned TRAIN BP or PC/CPC learner:
collected = run_frozen_rollout(Path("fresh-collection.json"), learner=source,
    train_seed=19, visual_observation=True, corrective_queries=True)
if collected.report["complete"]:
    batch = collected.corrections
    updated = run_continuous_correction_update(source, batch,
        now=batch.labels[-1].labels_arrived_at)

# Or the fixed composition: creates this directory exclusively.
summary = run_corrective_adaptation(source,
    output_directory=Path("fresh-adaptation"), visual_observation=True)
```

These examples are alternatives and perform actual work. They require a separate
frozen local native budget; they do not request rerunning accepted evidence.
No model loader, deployment, reset/retry loop or second simulator runner is added.
The existing CLI `main` delegates to the same body and keeps its report/arguments.
`FrozenRolloutResult` returns an owned report and correction batch only after
complete collection, cleanup and original/private model state guards pass.
Any failed collection releases no usable batch.

## Fixed update behavior

The update API admits exact TRAIN `NumpyContinuousBCLearner` and
`NumpyContinuousPCLearner` owners, including neutral and active CPC, with default
MSE base policy and already frozen observation statistics. It validates and owns the paired batch,
matching spec and complete label arrival before any model forks. A future release,
held-out role, invalid/tampered batch or mixed objective refuses early.

Eight complete correction batches train a detached candidate through the existing
`ActorShadowRuntime` and one original eight-update budget. The candidate starts
from every source parameter, momentum value and cumulative counter. No statistics
fit, reference replay, absent-context data or extra diagnostic forward occurs.
Source and stable actor remain exact. The last original observation and arrival
ticks remain on every receipt; distinct exposure identities distinguish repeats.
The API does not create later label arrivals or renew allowances.

Each previous success's full candidate state, receipts and paid counts validate
before more work. Failed updates retain the runtime's permanent receipts, partial
progress and native failed-work report when available. They release no candidate.
The final empty poll and detached restore recheck full state, owners, receipts
and charge. Successful return reuses `ContinuousLearnabilityResult`.

For BP with `n` correction rows, work is8 updates/8*n exposures/n unique/7*n repeated;
8 forward and8 backward batches,8*n normalizations,8 parameter-gradient arrays,
dense MACs `8*n*(2*D*H+3*H*A)`. Raw per-owner parameter+momentum bytes are
`16*(D*H+H+H*A+A)`; four named owners and eight bounded inbox payloads are
reported. Copies, validation, temporary snapshots, elementwise math and process
overhead remain separate. Query/controller/visual/prediction costs come from the
rollout reports, not the optimizer counts.

## Fixed native composition and artifacts

`run_corrective_adaptation` admits the frozen source before native creation.
Default visual mode requires789D. Explicit `observation_projection` now admits
public21 sources while preserving raw789 collection evidence; see
[public-vector corrections](public-vector-corrections.md) for the API, costs and
synthetic acceptance scope. It creates an exclusive output directory:

```text
collection.json    Source TRAIN19, student controls plus provisional query labels
update.json        Eight private updates, all paid receipts and costs
updated-17.json    Frozen candidate on TRAIN17
updated-19.json    Frozen candidate on TRAIN19
summary.json       Stage outcomes/costs and full source preservation
```

The returned full update report is saved in memory before file writing, so a
write failure still preserves paid receipts in the final summary. Any failed
stage stops later work. Raw student actions/rewards stay separate from labels;
the nominal teacher is not qualified as an expert at off-reference states.
No guards31/37, probe23 or final29 enter collection/update or selection.

## Checks and limits

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_continuous_correction_demo.py tests/test_pick_cube_corrections.py tests/test_maniskill_bc_smoke.py -k 'not direct_reference'
.venv/Scripts/python.exe -m ruff check src/adapters/continuous_correction_demo.py scripts/maniskill_continuous_smoke.py tests/test_continuous_correction_demo.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/continuous_correction_demo.py scripts/maniskill_continuous_smoke.py tests/test_continuous_correction_demo.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/adapters/continuous_correction_demo.py scripts/maniskill_continuous_smoke.py
```

The direct eight-step reference runs once globally under the original engineering
budget, then is excluded from repeats/platform checks; accepted old references
are reused. Tests cover pretrained momentum continuation, exact arrival/ownership,
corrupted preceding receipts, paid postcommit failures, empty polling, release/
cleanup, default compatibility and the complete fake-boundary pipeline.

The native study is a single fixed pre/post intervention with eight extra updates,
not equal-compute evidence or a test of query necessity. An improvement would
support only that combined recipe; a failure remains a valid result. Useful
adaptation, qualified correction quality, generalization and circadian advantage
require additional evidence. Keep all earlier negative cases and unfinished
roadmap requirements. [Protocol, observed results, exact commands and review](../artifacts/runs/corrective-adaptation-20261019/RESULTS.md).

Next extension must use existing learner/runtime ports and predeclare any change
to source/control data, exposures, replay, seeds, endpoints or physical work.

## Exact PC/CPC continuation (2026-10-20)

The same adapter and native composition now support PC, neutral CPC and active
CPC. `RecordedContinuousPC` retains at most eight native update reports. Full
parameters, momentum, wake state, counters and frozen statistics continue from
the source; the original source and serving actor remain unchanged. Native
event counts and row-weighted work replace the BP dense arithmetic formula.
Normalization, payload and wake memory are reported separately; copying,
validation and process overhead are additional. No latent relaxation occurs during serving.

Original budget/timer identity, receipts and prior returned native work validate
before another update. A rewritten report, replaced facade or late coherent wake
change refuses release. Paid native returns survive final metadata/clock failure;
missing metadata remains unknown. No failed charge is refunded or resealed.

The final combined selection passed131cases on Windows and pinned Linux, with
scoped lint/format/type checks and independent acceptance. It covers all three
PC families, continued momentum/chemistry, paid failures and the existing fake
19→17→19 pipeline. These are correctness checks, not native usefulness evidence.
[Protocol, commands, failures and results](../artifacts/runs/continuous-pc-corrections-20261020/RESULTS.md).

The separate frozen native question completed BP, PC and neutral CPC: all nine
50-control episodes had zero success. A whole-work parity assertion included
neutral-only chemistry costs and failed; active CPC was withheld under the
original stop rule. The four-family cohort is incomplete, with no retry or
advantage claim. Common work and trajectories are exact; full failure retained.

## Explicit geometry-readiness labels (2026-10-20)

`readiness_corrections=True` selects distinct provisional records in the existing
collection/private-update recipe. Query closing progress follows preceding actual
student gripper commands, not proposed close labels. Projection preserves record
family and arrival ticks. Wrongmode/family refuses before privatework; poisoned
query reports retain paid history. Original fixed default remains unchanged.
[API, checks and scientific limits](readiness-corrections.md).

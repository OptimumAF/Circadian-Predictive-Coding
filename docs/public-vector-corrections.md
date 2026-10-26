# Corrective adaptation with public21 learner inputs

The existing correction collector and eight-update private runtime now accept an
explicit public21 learner. The executed trajectory and geometry queries remain
raw789. This closes a composition gap; no new objective, fitter or runner is added.

## Boundaries and ownership

```text
src/core/pick_cube_public_corrections.py       pure owned raw/model batch views
src/adapters/continuous_correction_demo.py     arrival admission and private updates
scripts/maniskill_continuous_smoke.py          collection and frozen rollout pipeline
tests/test_pick_cube_public_corrections.py     projection/pairing/ownership behavior
tests/test_public_vector_correction_demo.py   adapter and fake-port integration
```

The core module imports only core contracts. It validates the complete raw TRAIN
episode, default RGB recipe and exact public projection before producing a model
batch. Every current and next observation is checked, including the pixels that
will be dropped. Student actions, rewards, end flags, context, episode identity,
provisional targets, stages and steps stay unchanged. Observation and arrival
clocks retain their original step-derived values.

Why this: predictions need21 coordinates, while query pairing needs the original
raw observation and executed actions. Keeping both views preserves that evidence
without relabeling student actions or qualifying a provisional correction batch as
an expert trajectory. Final evaluation is never an input to this path.

## API

Given a frozen TRAIN default-MSE continuous MLP with21 input coordinates and its
existing fitted statistics, pass the original collected raw batch:

```python
from src.adapters.continuous_correction_demo import run_continuous_correction_update
from src.core.pick_cube_public_vectors import public_vector_projection

result = run_continuous_correction_update(
    source,
    raw_batch,
    now=raw_batch.labels[-1].labels_arrived_at,
    observation_projection=public_vector_projection(),
)
```

The updater admits original label arrival before deriving its own model view and
before forking any learner. It never accepts a caller-supplied projected view as
proof of pairing. The source stays unchanged; a successful candidate continues
the full original optimizer state through exactly eight complete-batch updates.
There is no statistics refit, extra geometry query or diagnostic forward.

The existing pipeline also accepts the explicit option:

```python
from pathlib import Path
from scripts.maniskill_continuous_smoke import run_corrective_adaptation

report = run_corrective_adaptation(
    source,
    output_directory=Path("artifacts/my-prospectively-budgeted-case"),
    observation_projection=public_vector_projection(),
)
```

That pipeline collects source TRAIN19, performs eight private updates and runs
the frozen candidate on TRAIN17 then19. **Real native execution requires its own
prospective experiment protocol.** No native case ran in this implementation
increment. Existing default raw789 APIs, reports and omitted-option calls remain
unchanged. Actor, clock, camera and implicit-source incompatibilities remain.

During collection the exact raw tuple underlying the21D prediction is passed to
the current-port query check. Recipe, projection and complete learner state are
checked before control. Final correction release also compares the returned raw
episode and labels/clocks with retained execution/query evidence, then rechecks
owners after the final getter. Corruption leaves paid work visible and releases
no usable batch. Runtime failures similarly preserve permanent update receipts.

## Accounting and limits

ForN correction rows, one model view accounts for2N current/next projections,
2N*789 validated source values,2N*21 projected values and nominal float64 bytes.
Paired label payload is separatelyN*21 observation andN*4 target values; labels
reuse the matching projected transitions without extra projections. These are
logical payload counts, excluding copies, revalidation, Python overhead and
process memory. Existing update accounting uses21D model rows; collection retains
raw/query/render/control work separately. None of these costs renews a budget.

Run the focused selection from the repository root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m pytest -q tests/test_pick_cube_public_corrections.py tests/test_public_vector_correction_demo.py tests/test_continuous_correction_demo.py tests/test_pick_cube_public_vectors.py tests/test_pick_cube_corrections.py -k "not match_direct_reference_from_full_pretrained_momentum"
```

The unchanged independent direct-gradient reference is reused. Exact Windows and
pinned Linux receipts, scoped Ruff/format/type commands and independent review
are in [the implementation evidence](../artifacts/runs/public-vector-corrections-20261019/RESULTS.md).
These checks use synthetic data and fake simulator boundaries. They do not prove
native learning benefit, task generalization, circadian advantage or release
readiness. Earlier p8/p32 native failures and all negative results remain valid.

Next extension: freeze one small native correction protocol against the accepted
saved public21 source and existing uncorrected endpoints. Keep seeds and metrics
fixed, report added correction exposure/query/compute costs, and use this existing
pipeline. Do not add a new orchestration framework or tune the simpler controls.

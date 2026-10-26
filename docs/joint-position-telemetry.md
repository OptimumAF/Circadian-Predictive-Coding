# Joint-position telemetry

R4.5-JOINT-POSITION adds optional physical position observations to the existing
single-episode ManiSkill CPU port. It changes no learner inputs, actions, rewards,
physics or default factory options. The optional simulator remains lazily imported.

## API

```python
from dataclasses import asdict
from src.infra.maniskill_continuous import ManiSkillPickCubeEpisode

with ManiSkillPickCubeEpisode(
    "joint-example-17", "pick-cube", seed=17, monitor_joint_positions=True
) as episode:
    episode.step((0.0, 0.0, 0.0, 0.0))
    print(asdict(episode.joint_position_summary))
    print(asdict(episode.joint_position_work))
```

Run native callers only in the separately installed pinned Linux simulator
environment. Default `False` requires no joint access and returns `None` for both
properties. No environment variable or dependency was added. No final-test access
is enabled; existing role/seed restrictions still apply.

On opt-in, one `get_active_joints()` call verifies exact Panda order, and one
`get_qlimits()` call snapshots physical bounds after reset/context application.
Seven arm positions use radians; two prismatic fingers use meters. Numeric limits
are copied from the native array, with strict finite/shape/order validation.
The existing state42/RGB789 encoding begins with qpos9, so sampling adds no qpos
query, sensor packet, geometry access or control step. A changed robot/helper or
admission binding permanently refuses more controls and episode export.

Each summary includes owned limits, `samples`, `violating_samples`,
`violations_per_joint`, `lower_peak` and `upper_peak`. For each joint, the lower
peak is the most negative `min(0, qpos-lower)` and upper the largest positive
`max(0, qpos-upper)`. Exact boundary contact is allowed; **zero tolerance** means
even a tiny representable exceedance is recorded. Signs and units are retained;
there is no mixed-unit score. Counts include reset and accepted next encodings,
at most51 samples. Retained numeric state is O(joints), without a sample history.

`joint_position_work` separates name/limit attempted and returned accessor calls,
accepted samples and sampled coordinates. Returned calls count even when later
validation rejects their value. Constructor exceptions retain partial work in
`error.joint_position_work`; a cleanup error does not replace the original error.
Summary/work properties are pure snapshots available after close/failure. A valid
encoding can count before subsequent reward/flag rejection; that failed interaction
still cannot produce an admissible episode. Reporting/copying/Python overhead is
not measured by these numeric counters.

## Frozen rollout API and CLI

The existing `scripts.maniskill_continuous_smoke.run_frozen_rollout` accepts
`monitor_joint_positions=True` alongside a frozen `learner` or `actor`. Its report
adds `joint_position_summary`, `joint_position_work`, `joint_position_exact`,
`joint_position_validation_samples` and `joint_position_validation_coordinates`.
The same option is available to `main` and as `--monitor-joint-positions`:

```powershell
python -m scripts.maniskill_continuous_smoke --output new-joint-trace.json --monitor-joint-positions
```

This command uses the existing zero-action TRAIN policy unless an owned frozen
model is supplied through the Python API. Native execution uses a small
prospective budget and the pinned simulator environment. Existing destinations
are never overwritten.

Default calls add no fields/options and never read joint report properties. When
enabled, initial owned bounds and paid work are retained. Final reports are read
after cleanup, then every initial/next raw qpos9 sample is recomputed
with the core monitor. A completed `n`-control trace requires `n+1` samples and
records validation of `9*(n+1)` coordinates. This is extra pure diagnostic work,
not extra native observations or learner forwards. Clock/projection features do
not enter this validation: it uses the retained full raw42/789D episode.
The validation fields describe the submitted reconstruction batch; on an early
failure they bound attempted work, and only `exact=True` confirms all samples.

Only unchanged reset-time bounds, coherent monotonic counters and exact signed
extrema/counts can set `joint_position_exact=True`. Failed construction/interaction/
cleanup or malformed reports remain incomplete with `exact=None`; valid observed
partial work is retained, bounded by reset plus returned controls. Source/actor
state and existing mass/light metadata are rechecked after reporting callbacks.
No usable corrective batch is released after a final validation failure.

[Integration protocol and scoped evidence](../artifacts/runs/joint-rollout-integration-20261019/RESULTS.md).

## Scope and extension

Why this: normalized action bounds alone do not report physical joint deviations.
This observes declared reset-time bounds at control observation times; SDK limits
are themselves cached. Later limit changes, intermediate physics substeps,
velocity/torque/collision constraints and general safety remain unobserved. The
monitor never clips, suppresses violations, chooses a learner or changes training.

Pure behavior lives in `src/core/joint_position_limits.py`; native binding remains
in `src/infra/maniskill_continuous.py`. New task/controller support must declare
joint order/units and observation provenance explicitly. To expose telemetry
through a higher-level result, add an explicit opt-in and owned optional fields;
preserve current default calls/output and keep constraint readouts outside learner
features and evaluation/promotion choices. Do not add a new episode runner.

[Frozen protocol, checks and observed outcomes](../artifacts/runs/joint-position-telemetry-20261019/RESULTS.md)
and [independent review](../artifacts/runs/joint-position-telemetry-20261019/REVIEW.md).

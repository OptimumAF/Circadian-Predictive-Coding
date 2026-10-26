# Continuous-action CPU behavior cloning

`NumpyContinuousBCLearner` implements the existing `NativeLearner` contract for
continuous actions. This is an implemented baseline component for R4.4. A frozen
64-update pilot on one qualified PickCube TRAIN demonstration reduced training
loss, but both untrained and trained probe policies failed. It is not a qualified
strong robot baseline. [Teacher and pilot evidence](pick-cube-task-teacher.md).

## Modules

```text
src/core/continuous_bc.py           Pure prediction, loss and analytic gradients
src/adapters/numpy_continuous_bc.py Policy, momentum, snapshots and native port
tests/test_continuous_bc.py         Numeric oracle and boundary/continuation tests
```

The network has one tanh hidden layer and tanh output coordinates mapped into
each declared physical action interval. A convex blend returns exact bounds at
saturation; arithmetic outside the bounds is refused, not clipped. Constant
axes predict their fixed bound and contribute zero loss/gradient.

For each nonconstant axis, normalize the target by its declared center and
half-width. The named objective is half squared error averaged over rows and
nonconstant axes in those dimensionless coordinates. It is neither raw physical
MSE nor task reward. Raw default features retain supplied coordinates. Optional
[explicit TRAIN-fitted frozen standardization](continuous-bc-standardization.md)
is owned in the policy and full snapshots; it never refits during model use.
No unit inference, regularization or action clipping is introduced.
Bounds must have finite representable centers and positive representable
half-widths on nonconstant axes. Tiny subnormal intervals may be refused even
when the generic demonstration record can store them.

Defaults: hidden64, seed17, learning rate.01, momentum.9. Explicit configuration
permits hidden1..256, seed0..2**32-1, rate(0,1], momentum[0,1), and a fixed training
role (`train` or `probe_train`). These are implementation defaults, not a tuned or
qualified robot baseline. Initialization uses a local NumPy RNG; subsequent
updates have no stochastic state. Float64 array batches must have1..256rows.

## Usage

```python
from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.core.continuous_demonstration import ContinuousActionSpec

spec = ContinuousActionSpec(2, ("m/s",), (-0.5,), (0.5,))
learner = NumpyContinuousBCLearner(ContinuousBCPolicy(spec, hidden_dim=4))
# episode is a complete ContinuousEpisode with the identical spec and train role:
# diagnostic = learner.train_episode(episode)
# prediction = learner.predict(float64_features)
checkpoint = learner.snapshot_state()
fork = learner.fork()
learner.restore_state(checkpoint)
```

`train_episode` checks complete records, full spec/units/bounds and matching role;
evaluation records are refused. Raw `train_batch` arrays carry no provenance:
their caller must own the data-release authority, just as for existing native
ports. Neither method selects probe/final data or grants runtime promotion.

Snapshots contain full policy, four parameter arrays, four momentum arrays and
update/example counters. Copies/forks are detached. Restore validates policy,
shape/dtype/finiteness and coherent counters before replacing state. The update
uses `v = momentum*v + gradient`, `theta = theta - rate*v`; failed numeric
proposals raise `TrainingUpdateFailed` with native work and publish no state.
Diagnostics report the pre-update normalized objective. No disk codec or new
runner is provided. A single owner is responsible for serialized mutations.

## Executed checks and limits

### Optional row weights

`continuous_bc_loss_and_gradients`, `train_batch` and `train_episode` accept
keyword `row_weights=None`. Explicit weights must be an exact float64 NumPy
vector with one strictly positive finite value per row. The objective averages
row losses with masses `w / sum(w)`; active-axis normalization and the optional
binary gripper's stable logit2z cross entropy retain their definitions. Scaling
all weights by the same positive factor leaves the objective unchanged, within
floating-point precision. Uniform positive weights take the exact original
arithmetic branch.

```python
import numpy as np

# TRAIN supervision only; caller owns release authority for these arrays.
# diagnostic = learner.train_batch(features, targets,
#                                  row_weights=np.array([3.0, 1.0], dtype=np.float64))
# diagnostic = learner.train_episode(episode, row_weights=weights)
```

Weights normalize by their maximum first, avoiding overflow for large finite
values. Unrepresentable positive normalized mass is refused before a forward or
native gradient attempt. Explicit weights select `WEIGHTED_OBJECTIVE` or
`WEIGHTED_MIXED_OBJECTIVE`, including all-ones weights. The default keeps existing
objective IDs and callback signatures. Weights remain batch supervision; snapshots
and policy do not retain them. Update/example counters count actual rows rather
than weight mass. Frozen standardization, momentum and atomic publication still
apply. A failed gradient or source check retains its attempted/completed work.

Additional modules:

```text
tests/test_continuous_bc_row_weights.py        Scalar oracle and full derivatives
tests/test_numpy_continuous_bc_row_weights.py  Real native updates and failure isolation
```

Why this: accepted corrective diagnostics show better mean correction fit with
worse first-action and original expert fit. Positive row weights supply a
conventional learning capability for testing that hypothesis. They do not select
a schedule or establish useful physical adaptation. The existing recorded mixed
continuation now carries fixed weights, provenance and accounting; see the
[qualified runtime path](push-cube-learned-continuation.md). A frozen weighted
study remains open. Keep all observations and charge real row exposure;
do not use weights to relabel heldout data or omit unsuccessful examples.

The current scoped qualification and exact commands are recorded in
[weighted BC evidence](../artifacts/runs/weighted-continuous-bc-20261021/RESULTS.md).
To extend safely, pass owned positive weights through the existing continuation
boundary and record the objective/weights/costs there; preserve this default path
and qualify failure accounting before freezing a comparative experiment.

### Earlier unweighted qualification

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_continuous_bc.py tests/test_continuous_demonstration.py tests/test_demonstration_episode.py
.venv/Scripts/python.exe -m ruff check src/core/continuous_bc.py src/adapters/numpy_continuous_bc.py tests/test_continuous_bc.py
.venv/Scripts/python.exe -m ruff format --check src/core/continuous_bc.py src/adapters/numpy_continuous_bc.py tests/test_continuous_bc.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.11 --platform win32 src/core/continuous_bc.py src/adapters/numpy_continuous_bc.py
# Same two-source check passes with --platform linux.
.venv/Scripts/python.exe -X utf8 -B -m artifacts.runs.continuous-bc-learner-20261019.example
```

229combined tests pass (64BC,110continuous records,55categorical). Independent
central finite differences cover every parameter/bias; tests cover momentum,
full-state continuation, RNG/isolation, malformed state/inputs and actual SGD
overflow atomicity. The hand-authored example executes three model updates and
two diagnostic derivative evaluations, checks exact fork continuation and a small
loss decrease. It is not collected robot data or scientific learning evidence.
Initial fixture/type failures remain retained. First-attempt RSS was unmeasured;
the final combined run reports peak72,634,368B. No full-suite/release claim.

Parameter count is D*H+H+H*A+A. Parameter+momentum float64 payload is16times that
count in bytes; copies, inputs, activations, metadata and process RSS are additional.
Native proposal attempts and finite-difference work, including failed checks, are
reported separately. [Exact commands, failures, costs and review](../artifacts/runs/continuous-bc-learner-20261019/RESULTS.md).

The actual ManiSkill controller/data integration and successful task teacher are
now implemented and scoped evidence is accepted. A TRAIN-only diagnostic method
adds one forward with per-axis/stage residuals, exact state preservation, and
explicit work counts. [Definitions, usage, checks, and saved-state findings](continuous-bc-diagnostics.md).
Strong BC task quality, matched comparisons, retention/recovery, visual features,
and R4.4 parent acceptance remain unfinished. The original hand-authored example
does not establish those outcomes.

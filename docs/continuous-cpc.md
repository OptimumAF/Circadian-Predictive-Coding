# Matched continuous CPC modes

The existing `NumpyContinuousPCLearner` now supports PC, neutral CPC and active
CPC through one optimizer and atomic state path. The optional configuration uses
the existing coupled CPC wake equations, with the accepted bounded continuous
energy and inference. No second learner loop or stream runner is introduced.

```python
pc_policy = ContinuousPCPolicy(base_bc_policy)  # unchanged default PC
neutral_policy = replace(
    pc_policy, circadian=CoupledCircadianConfig.matched_pc_control()
)
active_policy = replace(pc_policy, circadian=CoupledCircadianConfig())
learner = NumpyContinuousPCLearner(neutral_policy)
```

`base_bc_policy` still determines the matched one-hidden architecture, seed,
normalization, learning rate, momentum, continuous geometry and training role.
All three modes share the same task energy/precision, inference and candidate
forward checks. The optional BC binary-gripper likelihood remains unsupported.
No dependency, environment variable, optimizer default or BP source changed.

## Chemistry and gradient selection

For the final **accepted** hidden free state (including a valid inference fallback):

```text
activity = mean_rows(abs(hidden_latents))
next_chemicals = decay * old_chemicals + buildup * activity
gates = clip(exp(-sensitivity * next_chemicals), min_plasticity, 1)
```

Gates multiply hidden incoming weight columns, hidden biases, and output weight
rows. Output bias is unchanged. Then the shared optimizer computes
`velocity = momentum * old_velocity + gated_gradient` and
`parameters = old_parameters - learning_rate * velocity`. Old velocity is not
gated; smaller gates do not guarantee frozen weights.

Neutral mode sets `min_plasticity=1`. It still performs real chemistry, exponent/
clip calculations and gradient products. Unit gates give exact PC common-state,
prediction, native diagnostic and inference identity; CPC-only state and work
remain separately present. Active-mode unit tests check this math against a
handwritten oracle. They are not a comparative benefit study.

## State and atomicity

`ContinuousPCPolicy` adds its last optional `circadian` field (default `None`).
`ContinuousPCSnapshot` adds its last optional `wake_state` field (default `None`).
Existing positional construction/default PC behavior remain compatible. These are
trusted local snapshots; no portable wire/checkpoint compatibility is claimed.

PC requires absent wake state. CPC requires complete owned `CoupledWakeState`:
same validated configuration, finite nonnegative float64 chemistry and gate
vectors of width H, exact derived gates, and a native wake count equal to the
learner's completed-update count. Zero updates require zero chemistry/unit gates.
Missing, replaced, incoherent or incompatible wake/configuration is refused.

Prospective chemistry and gradient gates precede the parameter proposal. The full
parameter/momentum/counter/wake candidate validates before the single publication.
Failures publish none of it and retain actual work. Snapshot/restore/fork copy
all arrays/configuration and do not draw new RNG values. No history chain is
retained. `get_chemical_state()`, `get_plasticity_state()` and `wake_updates` expose
owned readouts in CPC mode and refuse PC mode rather than inventing chemistry.

## Work and evaluation boundaries

The original 16 native update counters are unchanged and exactly match between
PC and neutral CPC. CPC adds `chemistry_attempts`, `chemistry_returns`,
`gate_applications_attempted`, and `gate_applications_returned`. These record the
transition and gradient-selection boundaries, including failure. They do not
exhaust CPC computation: snapshot/validation recomputes derived gates, copies wake
vectors, and checks coherence outside these four counters. Initialization,
metadata, byte copies, wall time and whole-process memory also need separate
accounting in any future experiment. Persistent CPC numeric state adds two
float64 vectors (16H bytes), excluding metadata and temporary copies.

The task diagnostic remains preparameter inferred energy, with task loss separate;
it is not a postupdate or shared BP loss metric. Prediction remains target-free,
and episode training keeps exact train/probe-train role checks. Final data must
remain unavailable to training, promotion, tuning and probe selection; probes
must use detached private learners. No dataset or native simulator is accessed.

## Example and checks

```powershell
python -m scripts.continuous_cpc_example
python -m pytest tests/test_continuous_cpc.py tests/test_numpy_continuous_pc.py
python -m ruff check src/core/continuous_cpc.py src/adapters/numpy_continuous_pc.py tests/test_continuous_cpc.py scripts/continuous_cpc_example.py
python -m ruff format --check src/core/continuous_cpc.py src/adapters/numpy_continuous_pc.py tests/test_continuous_cpc.py scripts/continuous_cpc_example.py
python -m mypy --follow-imports=silent src/core/continuous_cpc.py src/adapters/numpy_continuous_pc.py tests/test_continuous_cpc.py scripts/continuous_cpc_example.py
```

The fixed PC/neutral example performs two updates per arm including fork/restore
continuation, four total. It requires exact common-state identity, not a gain.
It contains no active performance arm, new task, final/probe data or native work.

```text
src/core/continuous_cpc.py          owned wake helpers and existing gate math
src/adapters/numpy_continuous_pc.py one optimizer, optional config/wake state
tests/test_continuous_cpc.py       neutral, active arithmetic and state checks
scripts/continuous_cpc_example.py  bounded exact-control continuation
```

Next, extend the existing continuous learnability runner's learner selection,
diagnostics and work reporting for BP/PC/neutral/active CPC. Its current BC-specific
construction/conditioning/work formulas cannot be relabeled as PC/CPC costs.
Freeze and review a small common A/B/A protocol before comparative execution.

[Protocol](../artifacts/runs/continuous-cpc-bridge-20261020/PROTOCOL.md) ·
[Results and limitations](../artifacts/runs/continuous-cpc-bridge-20261020/RESULTS.md) ·
[Independent acceptance](../artifacts/runs/continuous-cpc-bridge-20261020/REVIEW.md).

## Stream integration followup — 2026-10-20

The existing continuous A/B/A runner now selects BP/PC/neutral/active CPC with
common physical task diagnostics and separate native update costs. This closes
the runner prerequisite above; useful reserved fresh-task probing and scientific
benefit remain open. See [family API and evidence](continuous-mlp-learnability.md).

# Observed damage from tolerant linear merge

Preview the fixed engineering example:
`python -m scripts.run_linear_width_merge --describe`.
Run without `--describe` for both linear MLP/residual families' accepted, rejected
and failed decisions, complete states/lineage/probes and detached continuation.
The CLI sets CPU thread counts to one before imports and caps output to2MiB.
No new dependencies or environment configuration.

## Modules and API

```text
src/app/linear_merge_gate.py              owned observed-damage decision
scripts/run_linear_width_merge.py         analytic fixture and JSON CLI
tests/test_linear_merge_gate.py           independent references and fault controls
tests/test_linear_width_merge_cli.py      actual example behavior
```

`gate_linear_merge(initial, lineage, probes, limits=..., budget=...)` returns a
frozen `MergeDecision` with status, reason, complete state, accepted-only lineage,
completed diagnostics and planned probe count. `initial` is an exact supported
`LinearScalingState` or `LinearResidualState`; lineage is the existing
`LinearWidthLineage` describing the widened input configuration.

`MergeDamageLimits` is an exact public alias of the existing `PruneDamageLimits`:
`max_prediction_change`, `max_half_mse_increase`, `max_energy_increase`.
`MergeProbeDiagnostic` similarly aliases `PruneProbeDiagnostic`; these names share
the same classes, validation and record schema, rather than defining new types.
Inherited admission errors may mention pruning. The shared helpers perform no
pruning when used by this gate.

Why this: the core merge's branch tolerance, rtol1e-10/atol1e-12, admits small
off-image components. Downstream matrices can amplify them, so coordinate
compatibility is not a prediction or energy bound. Preserve that core behavior
and counterexample; application acceptance needs explicit observed damage.

## Admission and observations

Before external budget callbacks, the gate validates/copies the complete source,
lineage, limits and probe batches. Full family/configuration must match the
lineage's widened geometry. Probes must be a tuple of1..8 train/validation batches
with validation present, globally disjoint row IDs and no shared array storage.
Existing row/input/target/hidden activation caps apply. Final-test payloads are
refused without copying or evaluating them.

The core merge gets detached state/lineage copies; the returned candidate must
have the exact family and full declared original configuration. Each source and
candidate evaluation uses fresh state/features/targets, and copies relevant
results before resource callbacks. The native evaluator reports target-free
predictions, half-MSE `||y-prediction||²/(2B)`, and the full equilibrium sum of
represented residual penalties. Energy is checked independently of prediction
and half-MSE; comparing only the final output residual is insufficient.

For every declared probe, require all of:

- maximum absolute prediction change <= its limit;
- `max(0,candidate_half_mse-source_half_mse)` <= its limit;
- `max(0,candidate_energy-source_energy)` <= its limit.

Limits are finite, nonnegative and inclusive. Every probe completes before a
damage rejection; a good batch cannot hide a later failure. Resource checks occur
before candidate construction, around each evaluation, after each completed
diagnostic and before publication. These are safe-boundary checks, not preemption.

Accepted decisions return the detached candidate and copied lineage. Rejected
decisions return the entire owned source and no accepted lineage. Candidate,
numerical, evaluation, callback and resource faults return failed with the same
full-source fallback and known completed diagnostic prefix; invalid public
preflight raises. A failure after a complete source/candidate diagnostic is
recorded retains that diagnostic. Nothing publishes a model or changes caller-
owned state.

This is trusted in-process composition. Defensive native/result checks do not
authorize arbitrary untrusted helpers. Observations bound only the declared
probes, not unseen inputs, future SGD, global function error or serving quality.

## Example and checks

Both families have a benign mean-field H2 split/merge accepted at fixed1e-12
limits, followed by one SGD step using separate train-role data. The original
legacy MLP amplification fixture and a residual H2 fixed-identity analogue both
remain core-compatible but fail the fixed1e-6 prediction limit, even when
half-MSE and energy improve. A deterministic zero-time budget fails before core
candidate work. Full fallback state is retained; no scientific result is selected.

```powershell
python -m pytest -q tests/test_linear_merge_gate.py tests/test_linear_width_merge_cli.py
python -m ruff check src/app/linear_merge_gate.py scripts/run_linear_width_merge.py tests/test_linear_merge_gate.py tests/test_linear_width_merge_cli.py
python -m ruff format --check src/app/linear_merge_gate.py scripts/run_linear_width_merge.py tests/test_linear_merge_gate.py tests/test_linear_width_merge_cli.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/app/linear_merge_gate.py scripts/run_linear_width_merge.py tests/test_linear_merge_gate.py tests/test_linear_width_merge_cli.py
```

Add future selection behavior at this application boundary with independent
observations and explicit role/budget controls. Preserve the pure mathematical
merge and its limits; never tighten compatibility merely to remove a valid
negative example or treat this decision as serving authorization.

# Raw coupled topology through the runtime

Preview the bounded engineering fixture without model work:
`python -m scripts.run_coupled_raw_topology_runtime --describe`.
Run without `--describe` for full JSON states, refusal diagnostics, promotion
reports and rollback. The CLI sets CPU thread counts to one before imports and
bounds output to2MiB. No new dependencies or environment configuration.

## Modules

```text
src/core/coupled_scaling_topology.py           owned raw envelope/config chain
src/adapters/topology_coupled_scaling_learner.py opt-in native ports/reconstruction
src/app/coupled_scaling_prune_consolidation.py accepted gate -> candidate replacement
src/adapters/coupled_scaling_topology_demo.py   fixed runtime/guard composition
scripts/run_coupled_raw_topology_runtime.py    CLI
```

The format1 envelope contains complete `CoupledScalingState` and at most64
ordered raw prune records. Consecutive full source/target configurations join;
the last target matches current state. Raw tensors, scaling scheme/gamma/width/
depth/input, precisions, CPC policy, chemistry, gates and wake count remain
together. Copy/restore revalidate every mutable array. Earlier valid states and
shorter histories allow rollback; lineage is metadata, not authority or provenance.

Why this: raw width changes alter weight/bias factors. The existing effective
topology wrapper cannot restore those coordinates. The new adapter composes the
existing strict raw learner and pins every configuration field except width,
including the original gamma value without lossy float conversion. Learning and
inference settings belong to the adapter. A complete private learner/history
replacement validates before one payload swap. Build/fork own all mutable state.
Existing strict raw restore still rejects configuration changes.

## Consolidation and serving

Construct `CoupledScalingPruneConsolidator(removed_units, probes, limits=...,
budget=..., inference_steps=..., inference_learning_rate=...)` and call
`runtime.consolidate(event_id, operation)`. Probes must be bounded, storage/ID
disjoint train/validation batches with validation present; final-test roles are
rejected before payload access. Inputs/limits are copied before external callbacks.
History capacity and coordinate geometry are checked before running the gate.

Accepted evidence must contain matching source/target configuration and requested
coordinates, selected CPC history, finite representation error within the fixed
8eps relative bound, and all declared probe diagnostics. Fixed-cap inference
records require exact native types/work counts, no fallback, finite nonnegative
energies/norm, final energy matching the recorded inferred energy and no increase
from initial energy. Damage increases equal `max(0,candidate-source)` exactly;
no numerical tolerance can conceal damage at a zero limit. Probability change
cannot exceed one. These checks validate trusted gate output; they are not an
independent recomputation or authorization for an untrusted transform.

Only acceptance returns a full replacement with appended history. The receipt
diagnostic `coupled_scaling_prune_max_probability_change_v1` is observed local
damage, not training loss. Rejection/failure raises `CoupledScalingPruneRefused`
with an owned full-source decision. Runtime attempt IDs stay spent on refusal.
Existing stop behavior on restore failure and post-commit budget checks applies;
a post-commit budget error does not undo a completed consolidation. Arbitrary
consolidation remains refused under managed retention.

Candidate work is separate from serving. Bind `prototype.build_from_state` once
and use the same callable in the existing guard evaluator/controller. Digest the
whole raw envelope. Supply every warm/training/prune-selection sample key in
`training_ids`, including validation labels; the controller cannot infer arbitrary
selection IDs. Independent `inner_guard` old/new samples remain disjoint.
Promotion requires a fresh candidate revision and accepted guards. Commit swaps
the serving bundle; rollback restores the previous model/configuration, cache,
version and metadata with a new generation. No codec or managed-retention support.

## Example and checks

The fixed mean-field N3/H2 fixture uses gamma1.5 and PC/neutral/active CPC. Two
real raw wakes precede declared disconnected-coordinate surgery. The gate rejects
connected pruning, fails a deterministic exhausted budget, then accepts removal
`(0,1)` under prospective1e-12 local damage limits. One delayed label continues
the candidate at width2. A separate resource refusal precedes accepted promotion,
serving and exact old-bundle rollback. Raw PC/neutral parameters stay exactly equal.

Native tensor bytes exclude wrapper/heap/copy overhead; synthetic utility
thresholds and zero clocks test transactions, not deployment quality or timing.
Pruning rebinds factors and does not promise general function/energy or future
trajectory preservation. No scientific comparison, held-out final data or tuning.

```powershell
python -m pytest -q tests/test_coupled_scaling_topology.py tests/test_topology_coupled_scaling_learner.py tests/test_coupled_scaling_prune_consolidation.py tests/test_coupled_raw_topology_runtime.py
python -m ruff check <changed-files>
python -m ruff format --check <changed-files>
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 <changed-files>
```

Extend structure through a separately derived raw mapping/lineage contract and
independent mathematical tests. Raw split/merge remains unfinished; do not relax
the fixed precision/configuration bindings to fit an unproved transformation.

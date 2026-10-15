# Mixed raw structure through the runtime

Preview the fixed local example without model work:
`python -m scripts.run_coupled_structural_runtime --describe`.
Run without `--describe` for full raw/CPC states, tagged histories, refusal
diagnostics and serving transactions. CPU thread settings are one before imports;
JSON output is bounded to2MiB. No new dependencies or environment variables.

## Modules

```text
src/core/coupled_structural_state.py           owned rooted state and transition tags
src/adapters/structural_coupled_scaling_learner.py fixed-origin learner reconstruction
src/app/coupled_structural_consolidation.py    split/merge/prune candidate transforms
src/adapters/coupled_structural_demo.py        fixed data/runtime/serving composition
scripts/run_coupled_structural_runtime.py      CLI
```

`CoupledStructuralState(1, origin_config, raw_state, transitions=())` owns the
entire raw state and up to64 `CoupledStructuralTransition(kind,lineage)` records.
Exact tags derive complete configuration moves:

| Kind | Lineage type | Source → target |
|---|---|---|
| `split` | `CoupledScalingSplitLineage` | declared source → doubled width/half precision |
| `merge` | `CoupledScalingSplitLineage` | declared split target → declared source |
| `prune` | `CoupledScalingPruneLineage` | declared source → width minus one |

The full origin must match the first source, every target must join the next
source, and the final target must equal the current config. Empty history requires
current config equal to origin. Admission checks tags/types/cap before raw/native
allocation. Copy/restore revalidate every mutable config, lineage and array,
including complete CPC policy/chemistry/gates/wake count. Earlier valid states
and shorter histories permit rollback. Metadata describes configuration
reachability; it does not prove that past transforms ran or grant authority.

Why this: the prune-only topology adapter pins precision. The verified nonlinear
split changes precision under a specific mapping, so a separate full-origin-bound
adapter carries its explicit chain. Existing strict raw and prune-only adapters
retain their original restore behavior.

## Learners and consolidation

`StructuralCoupledScalingLearner(initial_raw, learning_rate=..., inference_steps=...,
inference_learning_rate=..., inference_policy=None)` pins the entire original
configuration and the fixed adapter settings. Restore/build/fork accept only the
same origin and a valid full chain. A complete replacement learner/history is
prepared privately before one payload assignment. Settings are not snapshot
fields and do not change automatically after structure. In particular, no
source/split inference or raw SGD trajectory identity is claimed.

Pass these operations to the existing `runtime.consolidate(event_id, operation)`:

- `CoupledScalingSplitConsolidator()` uses the verified dyadic core map.
- `CoupledScalingMergeConsolidator(split_lineage)` owns explicit declared merge
  geometry and requires its full target configuration before the kernel runs.
- `CoupledScalingStructuralPruneConsolidator(removed_units, probes, limits=...,
  budget=..., inference_steps=..., inference_learning_rate=...)` reuses the existing
  isolated raw prune gate. Train/validation admission, complete fixed-cap inference
  evidence, representation checks, observed limits and owned refusals remain.

Only accepted transforms append a tag. All check capacity before callbacks and
validate exact result/configuration/lineage/error and mapped CPC trace/count.
Split/merge receipt diagnostics name effective representation error; pruning
reports observed probability damage. The temporary prune-only envelope used
internally never replaces or erases the captured mixed history.

Merge need not undo the latest recorded split: an explicitly declared exact-pair
map can be valid without recovering historical weights. Conversely, split from
N3 toN6, followed by one-coordinate pruning toN5, cannot be inverted with the
original split lineage. That attempt refuses before the merge kernel, keeps the
split/prune tags and spends its event ID. A later valid operation can proceed.
Core exact-pair checks still reject physically incompatible matching geometry.

These are trusted kernel/delegation boundaries. Defensive result validation does
not independently recompute mathematical observations or authorize untrusted
transforms. Runtime attempt counters and budgets remain external. Existing
restore-fault stop behavior/post-commit budget checks apply; a post-commit budget
error does not undo a completed consolidation. Managed retention still refuses
arbitrary consolidation.

## Serving and example

Bind `prototype.build_from_state` once and give that same callable to the existing
guard evaluator and promotion controller. Digest the whole origin/raw/transition
envelope. Declare every warm/training/prune-selection key in `training_ids`,
including validation labels; generic runtime ports cannot infer arbitrary selection
IDs. Old/new `inner_guard` samples remain disjoint. Final-test data is excluded.
Serving publication remains separate from candidate acceptance.

The fixed mean-field N3/H2/gamma1.5/beta1 PC/neutral/active fixture starts with two
genuine wakes, then declared disconnected units. The main path splits, merges,
records damage and budget refusals, prunes, and trains one delayed-label update.
A detached branch prunes after split and refuses the original inverse merge.
Independent resource refusal precedes accepted promotion, serve and full old-bundle
rollback. Model/config/CPC/history/cache/version/metadata restore exactly except
the expected generation increment. PC/neutral raw arrays remain exactly equal.

Utility thresholds cover the probability domain, clocks are zero, and the byte
metric counts native tensor arrays only. They test transactions, not deployment
quality, throughput or comparative learning benefit. Representation tolerance is
not a global function bound; pruning damage is local observed probe damage.

```powershell
python -m pytest -q tests/test_coupled_structural_state.py tests/test_structural_coupled_scaling_learner.py tests/test_coupled_structural_consolidation.py tests/test_coupled_structural_runtime.py
python -m ruff check <changed-files>
python -m ruff format --check <changed-files>
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 <changed-files>
```

Exact checked paths/commands and results are retained in
artifacts/runs/coupled-structural-runtime-20261015/statics.json and RESULTS.md.
Add a future transition only after defining its full config/state map, admission
and refusal semantics, independent preservation/damage controls and serving
recovery. Preserve default bindings and external authority/budget boundaries.

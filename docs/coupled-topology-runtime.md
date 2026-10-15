# Native coupled topology through the runtime

`python -m scripts.run_coupled_topology_runtime --describe` previews the fixed
engineering example without model work. Run without `--describe` for full JSON
containing source/candidate/serving states, refusal diagnostics, promotion reports,
coordinate lineage and rollback. No environment variables or new dependencies
are required; the CLI sets native CPU thread counts to one before imports.

## Modules and boundaries

```text
src/core/coupled_topology.py              complete geometry/state/lineage record
src/adapters/topology_coupled_learner.py   opt-in width reconstruction and native ports
src/app/coupled_prune_consolidation.py    accepted damage decision -> consolidation state
src/adapters/coupled_topology_demo.py     fixed runtime/guard composition and JSON
scripts/run_coupled_topology_runtime.py  CLI, thread settings and output bound
```

`CoupledTopologyState` contains format1, native family, input/hidden geometry,
complete `CoupledSnapshot` and up to64 ordered `CoupledPruneLineage` records. It
owns arrays and validates consecutive source/target coordinates against final
geometry, family, input and precision. Restore revalidates mutable arrays and
reconstructs an independent native model. The structural domain remains bounded
to16 hidden layers/1M total parameter entries before allocation. CPC chemistry,
plasticity, fixed policy and native wake count survive reconstruction.

`TopologyCoupledLearner` implements the existing native/forkable learner ports.
Its constructor accepts the same fixed update settings as `CoupledLearner`.
`snapshot_state`, `restore_state`, `fork` and `build_from_state` preserve complete
topology/lineage and independent ownership. Restore/build pin family, input,
exact precision tuple and CPC fixed policy. Precision length therefore pins depth;
hidden widths may change. Training settings belong to the adapter and cannot be
redefined by the structural state. A complete replacement is validated privately
before a single learner/geometry/lineage payload assignment.

Why this: the existing adapter restores into fixed geometry. An explicit new
adapter lets the runtime reconstruct width changes while retaining the strict
behavior and defaults of existing `CoupledLearner`. State restoration may return
to an earlier valid topology and shorter coordinate history. These records are
metadata, not append-only audit history, data authorization or promotion authority.

## Consolidation and promotion

Construct `CoupledPruneConsolidator(layer, unit, probes, limits=..., budget=...,
inference_steps=..., inference_learning_rate=...)` and pass it to
`runtime.consolidate(event_id, operation)`. It accepts declared train/validation
probes, bounds/checks matrices before copying, then uses the existing isolated
pruning gate. Accepted decisions append one coordinate record and return a full
`ConsolidatedState`. The receipt diagnostic `coupled_prune_max_probability_change_v1`
reports the largest observed probability change; it is not training loss or
promotion utility. `CoupledPruneRefused.decision` returns a detached rejected or
failed decision. Such refusals produce no successful consolidation receipt.

The existing runtime serializes candidate work and charges attempted event IDs,
including refused transforms. Its existing stop behavior on restore failure and
post-commit budget checks remains intact. A post-commit budget error does not undo
the completed consolidation. Arbitrary consolidation still refuses managed
retention. No checkpoint codec or managed-retention support is added here.

Bind `builder = prototype.build_from_state` **once**, then pass that same callable
to `PromotionGuardEvaluator` and `ServingPromotionController`. Existing guard
policy, complete-state digest, candidate revision checks and serving slot swap
authorize publication. The example's digest callback canonically encodes the
entire wrapper, including native wake state and ordered lineage. Parameter-only
digests would omit part of this state boundary.

Supply `training_ids` containing every sample used for wake training **and local
prune selection**, including validation probes. The generic controller adds its
observed wake IDs; callers must declare selection IDs it cannot observe. The
example uses consistent `(episode, sample)` keys and disjoint `inner_guard` data.
Final-test data never enters these decisions. Labels in native predictions are
never supplied to `predict`; guard utility reads labels only after prediction.

## What the example demonstrates

PC, neutral CPC and active CPC each receive two real wakes, then a declared
parameter-only disconnection of the last unit. Native CPC history is retained.
Each runtime records a damage rejection and budget failure before accepting the
disconnected prune. Serving state/cache/version stay unchanged through candidate
work. An arrived training update continues at the new width, then a separate
resource guard refuses promotion. A passing guard prepares and commits the
complete new bundle. Rollback restores the previous topology, native state,
metadata and cache, with the expected incremented serving generation.

The example utility limits cover the full probability domain, and guard clocks
are deterministic fixtures. They exercise transaction behavior and do not
establish deployment quality, latency or circadian learning benefit. Gate energy
is fixed-step guarded relaxed energy, not equilibrium; equal iterations are not
equal compute. Broader native split/merge and nonlinear scaling R1.8 remain open.

## Focused validation

```powershell
.venv/Scripts/python.exe -m pytest tests/test_coupled_topology.py tests/test_topology_coupled_runtime.py tests/test_coupled_topology_runtime_cli.py -q -o addopts= -p no:cacheprovider
.venv/Scripts/python.exe -m ruff check src/core/coupled_topology.py src/core/coupled_width_pruning.py src/adapters/topology_coupled_learner.py src/app/coupled_prune_consolidation.py src/adapters/coupled_topology_demo.py scripts/run_coupled_topology_runtime.py tests/test_coupled_topology.py tests/test_topology_coupled_runtime.py tests/test_coupled_topology_runtime_cli.py
```

Use `ruff format --check` on the same nine-file scope. Scoped mypy uses
`--follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL
--platform win32` and `--platform linux`. Exact executed commands, outcomes,
budgets and skipped wider gates are retained in
`artifacts/runs/coupled-topology-runtime-20261015/RESULTS.md`.

Accepted R1.6d evidence:54 distinct focused cases/0 failures/errors/skips,
nine-file Ruff/format/Windows+Linux mypy and independent source/state/outcome review
pass. Saved CLI74,928B retains all three native variants. Joint32 training calls,
76 guarded inferences and356 energy evaluations stay within the prospective
64/192/1536 allowance; unused32/116/1180 closed. These are engineering checks.

Extend by adding a reviewed native transfer and tests for full state/lineage,
continued updates, refusal isolation and existing publication/rollback boundaries.
Keep local structural acceptance distinct from guard approval, and preserve
strict defaults, selection-data exclusions and managed-retention refusal.

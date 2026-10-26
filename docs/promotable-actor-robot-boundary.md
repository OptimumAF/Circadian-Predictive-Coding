# Promotable actor at the existing robot boundary

The PickCube smoke now accepts an exact nearest-BC `PromotableActor` through
the existing `main(actor=...)` argument. Each command uses that actual actor's
public, versioned `predict`; no replacement learner is forked or restored.
Publication remains the responsibility of `ServingPromotionController`.

## Files and dependency direction

```text
scripts/maniskill_continuous_smoke.py       IO, commands, receipts, cleanup
src/adapters/promotable_robot_binding.py   Exact admission and bundle comparison
tests/test_promotable_actor_robot_boundary.py  Fake robot IO and real app controller
```

The script depends on the adapter, which depends inward on existing app/core
serving interfaces and the nearest learner adapter. App/core/learner/runtime
implementations are unchanged. No dependency, orchestration framework, cache
codec or publication mechanism is added.

Why this: the accepted native context case acquired B in a private candidate,
while actual A-only serving still failed B. Connecting the existing publication
owner to the existing robot boundary makes committed serving observable.

## Usage and admission

```python
from scripts.maniskill_continuous_smoke import main

# Existing runtime.actor is an exact PromotableActor owning a warm TRAIN NN.
# sys.argv supplies --output fresh-trace.json; output reservation is exclusive.
exit_code = main(actor=runtime.actor, train_seed=17)
```

The existing StableActor, private nearest/MLP learner, default zero-policy,
state/RGB and explicit clock routes remain. Actor and learner arguments are
exclusive. Unlisted subclasses, foreign backends/storage, cold/non-TRAIN models,
invalid typed snapshots, controller units/bounds and versions refuse admission.
The new adapter inspects exact private slot/bundle/backend types and backend
identity only; model/bundle values and predictions use public locked methods.

The binding retains one owned serving snapshot: generation, version, complete
nearest model, serving configuration, cache, metadata and last cache tick.
Full observations before/after prediction, immediately before/after controls,
and after cleanup refuse any change. Commit/rollback changes the generation;
equal-state backend replacement is also refused. A subsequent fresh invocation
can admit a successfully committed version. No automatic commit, rollback,
budget renewal, cache update or hidden restoration occurs.

The last precommand observation follows clock/recipe reads, since those reads
may run trusted callbacks. Checks observe serial boundary events; they do not
hold the serving lock across a simulator command and do not promise atomicity
against an independently publishing thread between observation and command.

## Auxiliary values and bounded work

Auxiliary comparison accepts exact native None/bool/int/finite float/str/bytes,
lists/tuples/string-keyed dictionaries and finite numeric NumPy arrays (including
complex metadata). Equal-valued wrong types, custom/object graphs, cycles and
nonfinite data refuse. The local comparator permits at most4096 visited/queued
value nodes and1MiB of numeric/text payload per snapshot, including dictionary
keys. Container expansion and text/integer/array comparison are preflighted.

Cache entries additionally require native full SHA256 keys, exact
`CachedPrediction`, native expiry within `(last_cache_tick, last_cache_tick+TTL]`,
finite float64 Nx4 bounded actions, declared capacity, and1MiB aggregate cache
payload before array scanning. The combined auxiliary comparator still applies.
Valid populated caches remain unchanged: robot prediction calls `predict`, never
`serve`. Metadata is descriptive and grants no controller authority.

These limits cover the local comparator and cache scans. Existing public
`serving_snapshot` first deepcopies trusted state; the limits do not bound all
copy allocation/latency or certify arbitrary Python graphs/callbacks. Complete
model snapshots retain the existing nearest learner's separate validation and
prototype cap. Snapshot/copy overhead is separate from distance work.

## Receipts, evidence and extension

Policy is `frozen_promotable_nearest_continuous_bc`; scope is
`actual_promotable_actor_no_private_fork`. Receipt fields include
`actor_generation`, `actor_generation_exact`, `actor_configuration_exact`,
`actor_cache_exact`, `actor_metadata_exact`, `actor_last_cache_tick_exact`, and
`actor_serving_bundle_exact`. The exact flags are cleared before observation and
all become true only after complete successful comparison. Existing state,
version, backend, returned-version, attempt/return and partial-work fields remain.
Distance counts are derived from validated exact NN/query shapes; no nearest
indices or native distance reports are invented for actor outputs.

```powershell
python -m pytest -q -p no:cacheprovider tests/test_promotable_actor_robot_boundary.py tests/test_stable_actor_robot_boundary.py tests/test_serving_promotion.py tests/test_promotion_guard_evaluation.py
python -m ruff check src/adapters/promotable_robot_binding.py scripts/maniskill_continuous_smoke.py tests/test_promotable_actor_robot_boundary.py
python -m ruff format --check src/adapters/promotable_robot_binding.py scripts/maniskill_continuous_smoke.py tests/test_promotable_actor_robot_boundary.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/adapters/promotable_robot_binding.py scripts/maniskill_continuous_smoke.py
```

Tests use fake robot IO and the real existing runtime/evaluator/controller with
explicitly synthetic, independently declared disjoint guard queries. One delayed
private update preserves serving; a real controller commit changes later controls;
mid-rollout commit/rollback and late corruption refuse with completed work kept.
Final/probe roles, overlapping identities and future labels remain unavailable.
[Exact executed checks and limitations](../artifacts/runs/promotable-actor-robot-boundary-20261019/RESULTS.md).

No new native or comparative experiment was admitted. This new actor route and
native publication are not physically validated; native promotion first needs
legitimately allocated arrived/disjoint robot `inner_guard` data. The current
collector now has an explicit [guard-only path](guard-only-teacher.md) with frozen
fresh31/37 allocation and arrival-gated records. A subsequent [native guard case](native-guard-promotion.md)
qualified both teachers but rejected the candidate; no committed native serving
occurred. Do not relabel TRAIN trajectories or loosen guards to force publication;
keep final/probe/training authorities separate. Correct serving
integration establishes no generalization, domain recovery or circadian benefit.

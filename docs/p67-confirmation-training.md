# P6.7b1 unscored training composition

Prospective contract, 2026-09-30. P6.7b1 implemented and verified with sealed
development fixtures. The later complete P6.7b2/P6.7b unscored execution is
verified separately in [training results](p67-confirmation-training-results.md).

## Boundary

Inputs are the complete frozen P6.7a scope plus unchanged original family
configurations. Outputs are held after-A/after-B models, arrived role handles
and finite unscored facts. No artifact IO, source selection, outer/final
scoring or uncertainty analysis belongs here. No new algorithm or dependency.

All six families and every original arm remain required. The production
entry validates the complete manifest before generating data. Complete all
A work and verify held/live A state before constructing the first B source;
then train B and recheck every A/B checkpoint globally. Final fields and
outer arrays stay unopened. Role hashes/counts are metadata; train/inner
content is allowed only for the original wake/guard behavior.

## Facts and checkpoints

Preserve original unscored family role/cost/decision facts without renaming
or dropping unequal/rejected work. Gating/replay add no metrics: their
method facts are original scored method records with `development` omitted.
Derive replay exposure from actual applied boundary IDs and check CPC clocks.

Checkpoint records include model type, uniform parameter hash, width/count,
full canonical state and its digest, clocks, retention, lineage and selector
facts when applicable. Use the existing circadian snapshot canonicalizer;
baseline state binds all instance fields, including traffic history, and
capture checks intact parameter aliases/exact tensor shapes. Equal bytes
cannot establish alias relationships. Retain initial and both trained records.
No actual tensor values are exported by canonical array fingerprints.

Sleep/schedule helpers have no full rejection witness in their old raw
schema. Capture complete before/after states separately around unchanged
calls and require equality on rollback/skip. Combined/parent already retain
complete before/proposed/after facts; preserve them intact. Never put an
observer/callback into model state to capture telemetry.

## Acceptance before confirmation execution

Use first original development seeds in manifest order: gating/replay 41,
sleep 67, schedule 79, combined 263, parent 347. These are implementation
fixtures, without new scientific scores or reserved sources. Compare exact
legacy unscored facts, parameters/full checkpoint state and all raw costs;
for scored pilots, block their accuracy helper and verify remaining facts.
Raising outer/final, all-A-before-B, sealed roles, independent copy and late
full-state corruption tests must pass. Record exact commands and skips.

P6.7b1 is complete for all-six composition fixtures. P6.7b2 still owns strict
independent JSON validation, saved source/reference binding, actual joint
update/wall/RSS and artifact lifecycle failure gates, and two complete
560-cell unscored runs. The parent P6.7b/P6.7, original matrix, P6.11 analysis
and independent final scoring remain open; do not treat this app as a gate
for scientific publication until b2 passes.

## Implemented structure and verified commands

```text
src/app/
  continual_confirmation_state.py      checkpoint/role facts, complete live gates
  continual_confirmation_simple.py     gating/replay/A-boundary sleep phases
  continual_confirmation_periodic.py   schedule/combined/parent phases
  continual_confirmation_training.py   frozen scope and joint arrival/copy orchestration
tests/
  test_continual_confirmation_state.py
  test_continual_confirmation_training.py
```

Production `train_confirmation` rejects changed/partial scope before any
model/source; each adapter's original resolved configuration is compared
again before source arrival. Private phase fixtures use only the original
first development seed and expose no production bypass flag. Gating/replay
reference accuracy callbacks return placeholders with actual outer fields
omitted, then metric fields are removed. New composition raises on scorer/
pilot-runner calls and outer/final reads. The four later preflight references
need no scoring stub. No scientific accuracy is computed by these fixtures.

All 56 fixture cells exactly match raw legacy facts and full initial/A/B state.
Sleep/schedule forced rejection proves complete restoration; schedule observes
12 actual rejected replay executions and zero baseline replay on its one-seed
fixture. Late A parameter/traffic/noise/selector changes stop the first B
source; late A/B cursor and role-label changes are refused. Copies have
independent arrays. Chemistry, memory, lineage, alias/shape and finite-state
regressions also pass. Selected tests need no ignored artifact fixtures.

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q -ra --maxfail=1 tests/test_continual_confirmation_state.py tests/test_continual_confirmation_training.py tests/test_continual_confirmation_manifest.py tests/test_p67_confirmation_scope.py tests/test_continual_gating_pilot.py tests/test_continual_replay_factor_pilot.py tests/test_continual_sleep_factor_preflight.py tests/test_continual_schedule_factor_preflight.py tests/test_continual_combined_factor_preflight.py tests/test_continual_parent_factor_preflight.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/app/continual_confirmation_state.py src/app/continual_confirmation_simple.py src/app/continual_confirmation_periodic.py src/app/continual_confirmation_training.py tests/test_continual_confirmation_state.py tests/test_continual_confirmation_training.py
.\.venv\Scripts\python.exe -m mypy
git diff --check
```

**122 passed in 35.76 s, zero skipped**, including **53 new** tests; mypy
371 files and other static gates pass. A final type-narrowing-only correction
also passes all 18 checkpoint tests in .25 s. Original twelve development
bundles/twenty source usage files and scope SHA `622feead...b76c5772f` revalidate
without files or scientific sources changing. No confirmation or resource
claim; full CPU suite/CUDA/large sweeps/final scoring were skipped.

The later independent JSON and bounded execution gates are now complete;
see `docs/p67-confirmation-validation.md` and `docs/p67-confirmation-execution.md`.
Both full reserved processes independently bind scope/source/request and
live state, validate every cost/resource/artifact identity, and repeat exact
560-cell result bytes under unchanged caps. The historical fixture evidence
above remains the b1 evidence. Keep original scientific helpers intact;
P6.11a's predeclared analysis/correctness gate now passes; next freeze and
implement P6.7c's separate reference/checkpoint/final-release boundary.

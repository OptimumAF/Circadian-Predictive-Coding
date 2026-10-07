# P6.7c1c2: Final execution observer increment

Declared and implemented: 2026-10-01 (ADR-0160). This component alone left
**P6.7c1c2 unchecked**; the subsequent [full worker](p67-confirmation-scored-worker.md)
now completes c1c2/c1c/c1 correctness.
This increment implements the independent observer needed by the complete
scored worker. Full request/source/process/artifact correctness, actual
reserved final scoring and seed reports remain required. No actual reserved
final value was opened and no scientific acceptance criterion changed.

## Structure and responsibilities

```text
src/app/continual_confirmation_final_observation.py   pure whole observation links
src/infra/continual_confirmation_final_runtime.py     actual source/model observations
tests/test_continual_confirmation_final_observation.py
tests/test_continual_confirmation_final_runtime.py
docs/adr/ADR-0160-observe-final-source-reads-and-model-calls.md
```

The infra observer takes already held training and an external budget-check
callback. It retains an independent ordered model/role schedule outside model
snapshots. Temporarily wrap the original source's final fields and guard outer
input/labels; preserve train/inner/ID/hash/seal metadata. Call each actual
input/label field exactly once inside its scheduled release and bind the final
view to the same arrays, declared IDs and captured content. All releases
precede any prediction. Cached arrays support late checks without source reads.

Wrap the actual BP/PC/circadian prediction methods, including inherited parent
controls. Require each scheduled held model/input object and one actual call.
Independently derive the correct count from the returned float64 probabilities
at the unchanged 0.5 threshold, or record the existing nonfinite/FP exception
policy. Compare that outcome with the original adapter result. Record attempts,
returns, numerical exceptions, examples and model-kind partitions separately
from app totals and rollback state. Every planned numerical-failure endpoint
is still attempted; source/contract/resource/other errors abort.

Block optimizer calls, outer access, unscheduled source fields and stray/extra
predictions. A caught forbidden operation remains in the independent counters
and prevents successful verification. Restore all owned method/source/outer
guards on every exit, including cancellation and budget failure. Preserve other
changed role metadata so the enclosing complete state gate can reject it.

The pure app verifier first validates the whole fixed scored JSON, then derives
all 120 release/240 read/1,680 prediction events and totals from exact endpoint
records and independently declared model types. Require closed fields, exact
order and recursive types/values. The model-kind partition is BP 420/PC 450/
circadian 810 calls; total examples are 67,200. It has no arrays, live models,
IO or resource authority. Both pure and runtime observations explicitly deny
source provenance; app scoring still denies external execution authority.

Why this: a port invocation alone cannot prove that the intended prediction
ran, received the intended model/input, or produced the saved count. Actual
method/field observation supplies those facts without changing algorithms.

## Composition and validation

The prospective full worker will supply the unchanged `ExecutionObserver`
resource check. Keep the final observer active, with all views retained,
through scientific JSON serialization and the subsequent whole training-state,
view, source/request/reference and resource checks. The full fixed app API
remains mandatory; there is no partial scientific CLI or manifest override.

```python
# Inside the future bound worker, after complete request/reference/training gates:
final_observer = FinalExecutionObserver(trained, budget.checkpoint)
with final_observer.observe():
    scored = evaluate_confirmation(
        trained, manifest, final_observer.release,
        final_observer.evaluate, final_observer.checkpoint,
    )
    final_observer.verify_result(scored)
    # Serialize and validate whole scored JSON; recheck frozen bindings/budgets.
    # Recheck ALL held model state after serialization, then retained final views.
    verify_scoring_training_state(trained, manifest)
    final_observer.verify_result(scored)
```

This illustrates component composition; complete worker/publication/readback
gates are still unimplemented. The observer checks model identity, not every
parameter/controller/RNG value. A late parameter-mutation test explicitly
requires the separate complete c1a live-state gate. Generic supplied-inventory
support permits private development fixtures only; the public scientific
scoring/observation validators reject nonfixed/partial scope.

Run the development/fabricated component checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q -ra --maxfail=1 tests/test_continual_confirmation_final_runtime.py tests/test_continual_confirmation_final_observation.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src/app/continual_confirmation_final_observation.py src/infra/continual_confirmation_final_runtime.py tests/test_continual_confirmation_final_observation.py tests/test_continual_confirmation_final_runtime.py
.\.venv\Scripts\python.exe -m mypy
```

No new dependency, environment variable, algorithm, seed, metric or budget.
Original limits remain 16,000 optimizer calls/600 seconds/5-ms sampled observed
512-MiB RSS. Deterministic fake RSS/clock tests prove limit behavior, including
post-serialization stops; they are not new process-memory or runtime measurements.

## Available source freeze and repaired regression

The conservative available closure extends c1c1's unchanged 90 pins with the
two production modules: **92 files**. These are component freezes, with both
`full_scored_worker_bound` and `scored_request_bound` false. They cannot satisfy
the full c1c2 source/request freeze or authorize an actual reserved final.

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| Initial `artifacts/runs/p67-confirmation-final-observer-source.json` | `2548a3c5b8ec8d7640e4e586f689a2f126b8b0fef311ea244432df9523430445` | 11,470 |
| Initial source map | `563d866c1a049e6d225716921dca98b1f92c2aec690ce2d9299a96307e66572a` | — |
| Authoritative `artifacts/runs/p67-confirmation-final-observer-source-v2.json` | `2bc058e2859ade591a462f0674f3ea5d72c5f1205065246f63aa1f01c771f169` | 11,845 |
| Authoritative source map | `28aa619ed9c808872b5353c5644e80b3cab9aff8659db73020186f46a987417a` | — |
| Pure app source | `263561557dc26aa44923872d016bc406fa50b3577eb289c7b7a4c7428b354c9d` | — |
| Current runtime source | `b8e656687c10275f73210661d942c453f5bdb00625d0f24fd176be3de8d8db09` | — |
| Metadata-only `artifacts/runs/p67-confirmation-final-observer-validation.json` | `469715f65a28f638d03e7a83becd34102ebb01e5f19cc6aea803c01d5d5fc7e3` | 4,310 |

Initial freeze UTC **2026-10-01T09:32:06.192421+00:00**, before fabricated
scored component fixtures; the first two development checks passed. A new
callback regression then failed: `verify_result` returned before detecting
content drift caused by its final budget callback. Context-exit verification
would catch it later, but immediate method verification must fail. Move budget
checks before final content/link checks. Preserve the failed initial runtime
pin `e9ca6ff6...831f8`; V2 supersedes the original record and changes that pin
only. All other 91 pins remain exact. V2 froze at
**2026-10-01T09:45:32.046559+00:00**, before rerunning scored fixtures. The
callback regression and original two cases pass on V2. No subsequent
production repair or scientific setting/failure-policy/cap change.

## Evidence scope and remaining gate

First six development seeds supply genuine 56-cell trained held models and
all three original complete development proofs. Original final/outer fields
remain raising sentinels; replacements contain fabricated final arrays only.
Observe 12 releases, 12 input/12 label reads, 168 real predictions and 6,720
examples, partition 42/45/81. Prediction and guard removal leave complete
model/controller/selector/RNG state unchanged.

A full-size fixture reuses those development models with fabricated final
arrays and reserved-seed **metadata tags only**. Public app composition runs
under an explicit global-state spy. Independently observe all 120 releases,
240 source fields, 1,680 real predictions and 67,200 examples; require whole
pure JSON/observation agreement, including a numerical failure. This is not
reserved model/data construction or reproduction of the saved full train
result. Without the spy, the public global gate rejects it before any source
field. The original genuine development state also rechecks after reuse.

Tests cover missing/extra/changed actual predictions, forged correct counts,
copied/substituted final arrays, repeated/reversed field reads, forbidden and
caught calls, resource/source/contract/cancellation failures, all numerical
failure combinations, retained content including resealing, source/model
bindings, late observed/app endpoint/count drift and post-serialization gates.
Pure tests use an independent arm-name/count oracle and forbid live model/
source/train/prediction/RNG/file activity. All command outcomes, static gates,
source/test hashes and selected regression evidence are in the current log.

Final selected regression: **168 new/781 related passed in 200.26 s**, zero
skipped, including all 596 prior scored/reference/analysis/training gates and
17 original optimizer/resource cases. Runtime focused expansion passed 54
in 53.25 s; pure observation passed 109 in 18.17 s. Five further caught-call/
source/post-serialization-budget cases pass in the final combined gate.
Ruff/four-file format/mypy **412 files**/diff pass. Independently recompute the
entire static closure and all 92 current file hashes, V1-to-V2 one-file repair,
unchanged scientific manifest and all six original train reference bytes/
markers. The existing c1c1 readback report SHA/length also remains exact;
no new full reference decode or actual resource benchmark this increment.
The validation artifact above stores test/byte evidence only and denies
scientific execution and full worker/request authority.

Full CPU/CUDA/clean-clone/actual CI, new reserved training/scoring, sweeps and
actual intervals remain skipped. No selected test skip or new dependency.

The subsequent [complete actual confirmation/repeat](p67-confirmation-scored-results.md)
now passes both independent readbacks and exact scientific result-byte
equality through the 97-source worker. **Current exact next action: P6.11b**,
publish every original seed/interval/contrast and joined raw cost fact under
the unchanged contract; preserve all null/negative/inactive/undefined cases.

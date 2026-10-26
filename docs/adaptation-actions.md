# Bounded adaptation choices

`ResourceSharedRuntime` can accept a finite action catalog. A controller selects
only an ID; the original actor/shadow runtime, serving gate and budget retain
authority. Default wrappers still use `train_ready()` as before.

Given an existing ordinary TRAIN `runtime` and `gate`:

```python
from src.app.resource_sharing import ResourceSharedRuntime
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope

envelope = AdaptationEnvelope(
    actions=(AdaptationAction("wait"), AdaptationAction("wake-one", wake_updates=1)),
    max_decisions=8,
    max_wake_updates=2,
)
shared = ResourceSharedRuntime(runtime, gate, action_envelope=envelope)
shared.train_action("wait")                 # retains pending labels; zero native work
result = shared.train_action("wake-one")    # at most one causally ready update
print(result.training.updates, result.training.deferred_reason)
print(shared.adaptation_snapshot().progress)
```

Explicit mode refuses `train_ready()` so that it cannot bypass the declared
choices. `train_action` accepts no model, label, callback, permission or budget
argument. Catalogs contain1..32 exact immutable records; duplicates, malformed
fields and choices beyond their hard envelope fail. Returned configuration and
progress snapshots are detached. Trusted local Python is not a security sandbox.
The action ledger belongs to this live wrapper; no durable action-ledger format
is supplied. A future continuation extension must retain its original charges
and bindings, alongside the existing runtime's supported recovery requirements.

## Supported effects and accounting

| Choice | Current behavior |
| --- | --- |
| Zero wake, unchanged other fields | Explicit no-update; no native state/payload read or sharing admission |
| Positive wake bound | Existing causal inbox, serving priority and original budgets |
| Replay amount | Optional fixed whole TRAIN bank executor; [contract](categorical-replay-actions.md) |
| Learning-rate multiplier | Per-update rate+wake with optional native port; [contract](learning-rate-actions.md) |
| Neutral/active gate setting | Optional one-update native plasticity port; [contract](gate-actions.md) |
| Sleep allowance | One fixed-width NumPy component event; [contract](component-sleep-actions.md) |

Structural renewal and combined effects remain unsupported. Sleep events count
separately from TRAIN updates and consume original consolidation/shared quotas.

Unsupported choices never silently become wake or no-update. Full R7.1 remains
open until its remaining executors and acceptance criteria are satisfied.

Known catalog choices with decision quota available spend one opportunity before
execution or refusal. Empty polls, future-label polls, resource deferrals and
no-update all count. Invalid unknown IDs and calls after quota exhaustion cannot
obtain another opportunity. Future controller comparisons must also record all
outer opportunities and overhead; these counters alone are not that experiment.

Wake bounds are maxima, not promises to train. The original per-poll sharing
limit may reduce the returned prefix. A requested bound must fit the remaining
adaptation envelope before execution; select a smaller declared action instead
of implicitly changing the proposal. Sharing admissions are permanently charged
before native/budget callbacks, including failures. Native attempts/completions,
returned polls and receipts stay distinct. A post-update budget stop preserves a
completed receipt; uncertain native failures latch unknown work. No budget,
clock, model version or consent authority is renewed. Refused/failed choices
remain in the bounded counters; there is no growing success-record history.

The original gate still prioritizes serving, pause and resource availability.
Probe owners cannot enroll actions, and heldout events cannot enter the ordinary
TRAIN inbox. No final evaluation, probe update, promotion or controller training
occurs here; replay requires the separate fixed-bank executor. Native prediction/serving ports
retain their original behavior. Finish or resume work through the original owner;
creating controller data does not restore any old runtime authority.

## Executed checks

```console
python -m pytest -q tests/test_adaptation_action.py tests/test_resource_sharing.py -k "not match_complete_native_state_after_partial_poll_pause_and_resume"
python -m ruff check src/core/adaptation_action.py src/app/resource_sharing.py tests/test_adaptation_action.py
python -m ruff format --check src/core/adaptation_action.py src/app/resource_sharing.py tests/test_adaptation_action.py
python -m mypy --explicit-package-bases --follow-imports=silent src/core/adaptation_action.py src/app/resource_sharing.py tests/test_adaptation_action.py
```

Final79checks/platform passed, with two unchanged full native parity cases
deliberately deselected and reused. The native test
`test_native_no_update_and_bounded_wake_preserve_original_authorities` is the
executed example: two fixed TRAIN samples with future labels, no-update, empty
poll, serving deferral, bounded wake, envelope refusal, another no-update/wake,
then decision quota refusal. Exactly2updates/4rows per platform, original actor/
source/RNG unchanged, exact receipt order and native clocks. It proves admission
correctness, not useful task fit or an adaptation-controller benefit.

Evidence: `artifacts/runs/adaptation-action-admission-20261023/`. First scoped
mypy errors were repaired before model work; all charges are retained. No
comparison, warmup, dependency install, sweep or native retry was run.

Why this: bounded controller selection needs an executable admission boundary.
The existing shared runtime already owns that boundary. Reuse it, keep unsafe or
unsupported effects explicit, and establish fixed/heuristic causal references
before a learned controller. Add other executors only through their existing
learner/runtime ports with meaningful refusal and permanent-cost tests.

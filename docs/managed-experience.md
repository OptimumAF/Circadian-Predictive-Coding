# Managed local experience admission

Install `ManagedExperienceOwner` on a fresh `ResourceSharedRuntime` before registering
any source, label or consolidation. Declare metadata before delivering opaque payloads.
The original candidate budget, clock, serving gate and native update path are retained.

## Modules

```text
src/core/data_lifecycle.py       immutable declarations and admission limits
src/app/managed_experience.py    original consent authority and registration fence
src/app/experience_inbox.py      permanent hooks before copy and ready selection
tests/test_managed_experience.py metadata, bypass, opt-out and handoff controls
```

Why this: CPC may retain replay payloads even when arrival metadata does not request
replay. Only consented `replay` retention is currently supported. `transient` and
`audit_only` declarations are refused until actual native purge semantics are implemented.
Approved-record and replay-record quotas count lifetime declarations; opt-out never
refunds those grants. These are record counts, not native replay byte quotas.

Provenance is caller-declared local metadata, not authenticated evidence. Synthetic
and unverified records default to refusal; explicit policy flags can permit them.
Training still requires both training and replay consent and source permissions.
Held-out roles and unknown declarations are refused before payload copying.

```python
from src.app.actor_shadow import ActorShadowRuntime
from src.app.managed_experience import ManagedExperienceOwner
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.data_lifecycle import DataConsent, DataProvenance, LifecycleDeclaration, LifecycleLimits
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.resource_sharing import SharingLimits
from src.core.learner_ports import TrainingDiagnostic

class LocalLearner:
    def __init__(self):
        self.state = 0
    def fork(self):
        learner = LocalLearner()
        learner.state = self.state
        return learner
    def train_batch(self, features, targets):
        self.state += 1
        return TrainingDiagnostic("local_fixture", float(self.state))
    def predict(self, features):
        return self.state
    def snapshot_state(self):
        return self.state
    def restore_state(self, state):
        self.state = state

clock = LogicalClock()
budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
runtime = ActorShadowRuntime(LocalLearner(), actor_version="actor-0", candidate_version="candidate-0", clock=clock, budget=budget)
gate = ServingPriorityGate(SharingLimits(1, 1, 1), resource_available=lambda: True)
shared = ResourceSharedRuntime(runtime, gate)
owner = ManagedExperienceOwner(shared, limits=LifecycleLimits(1, 1))
owner.declare(LifecycleDeclaration(("episode", "sample"), DataProvenance("local", "subject", True, False), DataConsent(True, True), "replay"))
owner.record_label(LabelArrival("label", "sample", "episode", 3, "actor-0", [1]))
owner.record_experience(Experience("sample", "episode", 1, "actor-0", [0], "train", ExperiencePermissions(True, True)))
owner.opt_out("subject")
clock.advance_to(3)
assert shared.train_ready().updates == ()
assert runtime.train_ready() == ()
assert budget.updates_completed == 0
```

The underlying public runtime cannot register events outside this owner, even for a
declared key. Direct public training polls retain consent eligibility. Supported
same-process checkpoint handoff preserves the exact original hooks and catalog;
declaration/opt-out mutations invalidate previously captured checkpoints. A second
manager cannot renew policy or grant counts. Candidate contention and reentrance
are explicit refusals. Arbitrary private field mutation is outside trusted ownership.

**Opt-out stops future arrived training. It does not delete inbox or native replay
payloads, caller-held copies, checkpoint copies, or learned parameter influence.**
R3.6b must implement native/inbox erasure, payload-free tombstones and checkpoint
invalidation/non-resurrection controls before the full R3.6 task can pass. No durable
cross-process consent restoration or arbitrary consolidation filtering is claimed.
Legacy runtimes without this optional manager retain their existing behavior.

## Validation commands

```powershell
python -m pytest tests/test_managed_experience.py -q -k 'not apply_one_native_wake'
python -m ruff check src tests scripts
python -m ruff format --check src/core/data_lifecycle.py src/app/managed_experience.py src/app/experience_inbox.py tests/test_managed_experience.py
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
```

Native controls are separately budgeted once, two fixed wakes, no predictions or
sleep. Extend retention categories only after proving payload erasure and checkpoint
non-resurrection under the same ownership and cumulative budget constraints.

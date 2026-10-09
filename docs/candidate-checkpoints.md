# Complete owned candidate checkpoint handoff

R3.5b transfers a complete candidate into one replacement owner within the same
process. This supported path preserves the original clocks, budget and resource
adapters instead of constructing fresh quotas. It does not provide restart-from-
disk recovery. That extension requires a separately specified monotonic time and
resource accounting policy; a serialized cursor alone cannot supply it.

```text
src/app/candidate_checkpoint.py    bounded capture/inspect/restore authority
src/app/actor_shadow.py            candidate revision/retirement/materialization
src/app/experience_inbox.py        internal complete validated history materialization
src/app/resource_sharing.py        paused checkpoint lease; serving remains available
tests/test_candidate_checkpoint.py fake/refusal and both-native continuation controls
```

## Ownership and operation

Create `CandidateCheckpointController` around the actual `ResourceSharedRuntime`
handle. Configure a trusted independent native builder, complete native state
digest and complete native training policy digest. Each digest returns lowercase
SHA-256. Policy covers learning rates/inference settings as applicable; native
state includes parameters, aliases, RNG, replay and traffic. These are local
trusted probes, not an arbitrary graph/source security certificate.

Pause the original `ServingPriorityGate` before capture or restore. Entry refuses
active training, currently admitted serving, another checkpoint or a busy
candidate. Once preparation starts, new serving remains available through the
original actor. Resume is refused until the checkpoint lease exits. The original
gate retains its resource callable, limits, spent admitted attempts and deferrals;
the controller never creates a new gate. A request still active at final recheck
can refuse restore safely; retry is bounded by preparation attempts.

`capture()` returns an identity token. `inspect()` returns a detached observation
of full native state, validated inbox source/label/applied/duplicate/future/stopped
history, base/candidate versions, consolidation attempts/receipts/limit, budget
observations, sharing state and logical event tick. Private trusted local payloads
are copied and integrity bound, with metadata validated first. No unpickling of
caller input occurs. Copying or serializing a token does not grant authority.

Restore binds the current owner/revision, original builder/probes, exact actor
version/generation, native state/policy and complete observation integrity.
Registration, polling, consolidation attempts or changed configuration invalidate
captured authority. Foreign, copied, discarded, used, corrupt or stale tokens are
refused. Pending tokens and total preparation attempts have explicit positive
capacities. Charge preparation attempts before native callbacks, including errors;
discard/re-capture never renews the controller's lifetime attempt quota.

Build and restore an independent native model, validate full state and policy
before and after preparation, and materialize all inbox/consolidation history
without replay. The replacement retains the same original `ToyBudgetSession`,
wall callable, progress and optional live `ProcessRssSampler`, exact original
`LogicalClock`, gate/resource probe and actor. Elapsed time and RSS telemetry may
advance during preparation; they are never restored backwards. Other budget
configuration, cumulative work and progress values must match. Unknown event-clock
or resource adapters are refused. Calls to wall/resource probes can legitimately
advance telemetry even when native preparation fails; no live candidate native
state/history is changed by that failure.

Under the serving read gate, recheck generation and publish the new runtime
pointer while retiring the old candidate. No callback/copy occurs after this
point. Old candidate training/registration/consolidation/snapshot callbacks refuse
retirement. Its actor stays available. Existing pending promotion tickets remain
bound to the old runtime and cannot commit through either owner; prepare a fresh
guard ticket for the replacement. Actor model/cache/metadata/version/generation
are retained, not rolled back. All wrapper serving/training must continue through
the retained handle. Direct private field mutation or parallel external ownership
of budget/probes/models is outside this trusted exclusive ownership contract.

Stopped inbox/candidate histories remain stopped in the replacement; uncertain
native work is not made retryable. Retained consumed IDs and original quotas
prevent reapplying completed labels or renewing failed consolidation attempts.

## Runnable local example

```python
import hashlib
import pickle
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.resource_sharing import SharingLimits

class Learner:
    def __init__(self, state=0): self.state = state
    def fork(self): return Learner(self.state)
    def train_batch(self, features, targets):
        self.state += targets
        return TrainingDiagnostic("fixture", float(self.state))
    def predict(self, features): return self.state
    def snapshot_state(self): return self.state
    def restore_state(self, state): self.state = state

def digest(state): return hashlib.sha256(pickle.dumps(state)).hexdigest()
clock = LogicalClock()
budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda:0.0)
old = ActorShadowRuntime(Learner(), actor_version="actor-0", candidate_version="candidate-0", clock=clock, budget=budget)
gate = ServingPriorityGate(SharingLimits(2,2,1), resource_available=lambda:True)
shared = ResourceSharedRuntime(old,gate)
for sample in ("s1","s2"):
    old.record_experience(Experience(sample,"e1",1,"actor-0",0,"train",ExperiencePermissions(training=True)))
    old.record_label(LabelArrival("label-"+sample,sample,"e1",3,"actor-0",1))
clock.advance_to(3)
shared.train_ready()
gate.pause()
control = CandidateCheckpointController(shared,build_learner=Learner,state_digest=digest,policy_digest=lambda learner:digest("fixture-policy"))
new = control.restore(control.capture())
assert new is shared._runtime and new is not old and new._budget is budget
assert new.actor is old.actor and shared.predict(0).prediction == 0
gate.resume()
assert len(shared.train_ready().updates) == 1 and budget.updates_completed == 2
assert new.candidate_snapshot().state == 2 and old._retired
```

## Validation and extension

```powershell
python -m pytest -q tests/test_candidate_checkpoint.py tests/test_inbox_cursor.py tests/test_resource_sharing.py
python -m ruff check src tests scripts
python -m ruff format --check src/app/candidate_checkpoint.py src/app/actor_shadow.py src/app/experience_inbox.py src/app/resource_sharing.py tests/test_candidate_checkpoint.py
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
```

Native continuation controls use the existing fixed fixtures and compare complete
state/receipts after two identical arrived updates, interrupted and uninterrupted.
They do not select metrics/seeds or test scientific advantage. Actual idle versus
training serving p50/p95 remains the next R3.5c gate. Add another supported adapter
only with full state/policy/ownership, chronology and spent-resource continuation
controls; keep durable process recovery unchecked until its full policy passes.

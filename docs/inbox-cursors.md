# Complete owned inbox cursor capture

R3.5b1 provides the domain format and quiescent capture needed before complete
candidate checkpoint handoff. R3.5b remains open: native model, candidate/actor
identity, cumulative budget/time/resource/sharing state and old-owner retirement
must still be restored together. This record grants no public resume authority.

```text
src/core/inbox_cursor.py       validated versioned complete inbox histories
src/app/experience_inbox.py    quiescent owned cursor capture
tests/test_inbox_cursor.py     corrupt metadata, payload ownership, native capture
docs/adr/ADR-0204-validate-inbox-history-before-checkpoint-copy.md
```

## Supported metadata

`InboxCursor` retains format version1, candidate learner version, identity capacity,
complete source/label/applied histories, last observed event tick, stopped flag and
the budget's observed completed-update count. Immutable tuple histories preserve
pending source-only/label-only/future pairs as well as consumed IDs. Unique label
event IDs can be reconstructed exactly from retained labels; capture checks the
owner's duplicate-event and source/label/applied indexes before copying.

Validate exact record types, all declared metadata fields, train role/training
permission, exact permission booleans, source/label/event/applied ID uniqueness,
capacity, matching paired model versions and observation/arrival order. Every
applied receipt must reference its full source and label, candidate version/event
ID/times must match, update numbers must form the committed1..N sequence, applied
ticks must be between label arrival and last observed tick, and the native diagnostic
must retain its original valid definition/value. Pending future arrival ticks can
exceed the last observed tick. Labels may precede source transport delivery.

An open cursor requires completed work count equal to its committed receipts,
making the supported exclusive inbox budget explicit. A stopped cursor can retain
greater observed completed work after uncertain native/receipt failure, but never
less than recorded receipts. Stopped capture remains an observation of poisoned
state; it does not permit native retry or reopening the inbox.

`validate_inbox_cursor(object)` accepts unknown input and refuses unsupported exact
type/format before field or payload access. Construction validates metadata too.
Opaque payloads are never copied or inspected by core validation. A reader of a
trusted stored record must call the validator again: some serializers bypass
dataclass constructors. Serialization/security/IO/source provenance stay outside
this module; this is not arbitrary Python graph certification.

## Capture and ownership

`ExperienceInbox.capture_cursor()` refuses an active drain, including admission/native
callbacks. Use a quiescent exclusively owned inbox; an enclosing runtime must hold
its candidate gate when composing this with native state in the next handoff task.
The standalone inbox has no independent thread-safe transport registration gate.

Build and validate metadata first, then deepcopy the entire record. Bad held-out or
unpermitted metadata is rejected before a poisoned payload can run a copy callback.
Returned source/target arrays can be edited without changing the owner. Capture
does not train, predict, sample clocks/resources, restore models or change budgets.
The current inbox's ownership and spent ledgers remain intact.

Why this: restoring just native weights loses pending/future labels, consumed event
IDs and applied work, which can replay training or reset its quota. A complete
validated history contract is necessary before an outer atomic restore can safely
retire the old owner and transfer cumulative budgets and clocks.

## Local example

```python
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.inbox_cursor import validate_inbox_cursor
from src.core.learner_ports import TrainingDiagnostic

class Learner:
    def __init__(self): self.count = 0
    def train_batch(self, features, targets):
        self.count += 1
        return TrainingDiagnostic("fixture_native_v1", 0.0)
    def predict(self, features): return features[:]
    def snapshot_state(self): return self.count
    def restore_state(self, state): self.count = state

clock = LogicalClock()
budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda:0.0)
inbox = ExperienceInbox(Learner(), clock=clock, budget=budget, learner_version="candidate-0")
for sample in ("s1","s2"):
    inbox.record_experience(Experience(sample,"e1",1,"actor-0",[sample],"train",
        ExperiencePermissions(training=True)))
    inbox.record_label(LabelArrival("label-"+sample,sample,"e1",3,"actor-0",["target"]))
clock.advance_to(3)
inbox.drain(max_updates=1)
cursor = inbox.capture_cursor()
validate_inbox_cursor(cursor)
assert cursor.completed_updates == 1 and len(cursor.applied) == 1
cursor.experiences[0].features.append("mutation")
assert "mutation" not in inbox.capture_cursor().experiences[0].features
assert len(inbox.drain()) == 1 and budget.updates_completed == 2
assert cursor.completed_updates == 1
```

## Verification and next action

```powershell
python -m pytest -q tests/test_inbox_cursor.py tests/test_experience_inbox.py
python -m ruff check src tests scripts
python -m ruff format --check src/core/inbox_cursor.py src/app/experience_inbox.py tests/test_inbox_cursor.py
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
```

Next implement R3.5b's complete atomic owned handoff. Lease the candidate and require
a supported quiescent sharing state; bind the exact native state/history/revision
to actor generation and original clock/budget/resource ownership. Prepare native
restore off the serving path, recheck current cursors/ledgers, transfer to one new
candidate owner and retire the old owner. Reject foreign/stale/reused/corrupt/busy/
unsupported snapshots; failed preparation must leave live state intact. Preserve
spent native and admission quotas and elapsed time; do not construct a fresh budget
to resume consumed work. Stopped state must remain nonretryable. Keep pending
promotion authority separate and native settings fixed for both models. Actual
matched idle/training serving p50/p95 remains R3.5c after these correctness gates.


## Chronology follow-up

Applied receipt ticks must be nondecreasing in committed update-number order. Equal ticks are valid within one drain; individually valid backward times are refused. Standalone low-memory controls and scoped formatting/lint pass for this correction. Full current-source types/regressions/checkout acceptance remains unfinished after host memory exhaustion; see docs/development-log.md. No restore or native-call budget renewal is granted.


## Current validation accepted

R3.5b1 now passes330 current cases, both full645-file platform type targets, static/AST/guide/checkout/resource gates. See artifacts/runs/r35b1-clock-type-validation-20261007/. Original timeout and spent native allowance remain preserved. Full R3.5b owned restore remains the next implementation.


### Complete candidate checkpoint ownership

Supported same-process full candidate handoff is implemented through app/candidate_checkpoint.py; see docs/candidate-checkpoints.md and ADR-0205. Retain original cumulative budget/clocks/RSS sampler/resource gate and stable actor, restore independent native state plus full inbox/consolidation histories, retire old owner and invalidate its promotion authority. Identity tokens and preparation work are bounded. R3.5b acceptance remains pending current full validation; durable process recovery and actual live latency remain unfinished.

R3.5b current supported owned checkpoint acceptance:371 tests,647-file Windows/Linux types,full scoped/static/source/resource/guide gates pass. Evidence:artifacts/runs/r35b-owned-handoff-20261007/. Actual live R3.5c latency and durable R3.5b2 recovery remain unfinished.

# Matched local promotion guards

R3.4a measures a frozen actor and candidate on the same new-task and old-task
inner guards, then applies a declared policy. R3.4 remains incomplete: this report
does not authorize promotion or provide atomic serving transitions or rollback.
Those are R3.4b's preserved acceptance criteria.

## Modules

```text
src/core/promotion_guard.py             metadata, policy, evidence and decision
src/app/promotion_guard_evaluation.py   detached matched measurements
tests/test_promotion_guard.py           policy and metadata behavior
tests/test_promotion_guard_evaluation.py ownership, isolation and native parity
docs/adr/ADR-0201-measure-matched-promotion-guards-before-serving-swaps.md
```

Core performs no IO/model operations. App depends on core and the existing native
learner port, never infra or adapters. Existing actor/runtime/native model files
are unchanged. No dependency, environment variable, new training rule or service
is added.

## Policy and evidence

Declare `PromotionPolicy` before assessment. Utility is higher-is-better with an
explicit metric ID. Actor and candidate use one identical utility callable and
the same copied targets/features for each guard. Utility is separate from each
native training diagnostic; CPC energy and backprop loss are not compared.

The candidate must meet all configured gates:

- Minimum absolute new-task utility and minimum signed gain over the actor.
- Maximum old-task utility drop relative to the actor, on the declared old guard.
- Numerical validity of both source states and all measured predictions.
- Maximum observed candidate prediction duration over the two guard calls.
- Maximum candidate bytes under an explicitly named configured resource probe.
- Exactly one decoded action per sample and membership in the allowed-action IDs.

Thresholds are inclusive. Signed utility/gain thresholds may express a declared
tradeoff; retention/latency/byte limits must be nonnegative. No threshold is chosen
from a measured outcome. Nonfinite observations, unavailable required evidence
or any failed gate reject with explicit reasons. A numerical-state rejection does
not call the model factory or invent missing scores. Invalid callback types,
clock readings or native restore errors raise useful errors and leave the actual
actor/candidate untouched.

`PromotionGuardReport` records the complete policy, actor/candidate versions,
candidate wake/consolidation revision tuple, full snapshot digests, task/sample
IDs, declared label arrivals/training IDs, observation tick and every evidence
field/decision. Only immutable metadata and scalar outcomes are returned; no
mutable learner or source payload is exposed.

## Roles, timing and ownership

`GuardBatch` keeps opaque features/targets separate from task ID, episode/sample
keys, label-arrival tick and an existing role name. Evaluation accepts only
`inner_guard`. Train, outer-selection and final-test roles are refused before
payload copying or any factory/numerical/digest/probe callback. Labels must have
arrived, task IDs must differ, and new/old guards must be disjoint from each other
and all declared training IDs. Supply all allocated training IDs, including
pending events; completed update history alone is insufficient to declare a split.
The candidate's base actor version must match and its learner version must differ.
Revision counts are nonnegative and completed consolidations cannot exceed attempts.

Snapshots and inputs must remain quiescent while their copies are taken. Restore
independent native predictor copies, copy predictor inputs/results, and give each
utility/safety/numerical callback another detached copy. Input or callback mutation
cannot overwrite a sibling measurement or the source model. Refuse a builder
returning the same predictor for actor/candidate. Probe full model digests before
and after assessment; unexpected model mutation during prediction rejects it.

Builders and probes are trusted local extension code. Digest IDs and role names
do not certify arbitrary Python graphs, physical provenance or the global final
seal. This module cannot establish truth of caller-supplied IDs or police a
callback accessing data through unrelated closures. Existing scientific/source
authorities remain separate. Frequent promotion feedback is development data;
retain a disjoint assessment route before making independent performance claims.

## Measurements and supported limits

Timing uses a configured monotonic seconds clock. The observation tick used for
label arrival is a separate logical domain. Record the largest of the two actual
candidate prediction durations, including output copying; input detachment before
the call is outside that timing. A byte probe must return an exact nonnegative
integer and its definition is part of the policy. The native fixtures measure
serialized snapshot bytes. This is a model-state size measure, not process RSS,
allocator peaks or an allocation cap. These guards reject measured over-limit
results; they do not hard-preempt a prediction or implement live resource sharing.
Serving p50/p95 under contention and supported cursors remain R3.5.

Both native fixtures keep the existing fixed seed23/settings and common binary
accuracy, restore equal actor/candidate state, and reject the declared positive
gain requirement. That is valid negative engineering evidence. It establishes no
circadian/baseline ranking, generalization advantage or deployed latency claim.

## Small local example

This example uses an opaque two-coordinate fixture, with no native training or
external data. Real adapters must provide complete native restore and probe tests.

```python
from hashlib import sha256
from math import isfinite
import pickle
from time import monotonic
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.core.actor_ports import CandidateState, VersionedState
from src.core.promotion_guard import GuardBatch, PromotionPolicy

class FixtureLearner:
    def __init__(self, state):
        self.state = list(state)
    def train_batch(self, features, targets):
        raise RuntimeError("guard assessment never trains")
    def predict(self, features):
        return [self.state[features[0]]]
    def snapshot_state(self):
        return self.state[:]
    def restore_state(self, state):
        self.state = list(state)

evaluator = PromotionGuardEvaluator(
    build_learner=FixtureLearner,
    utility=lambda prediction, targets: 1.0 - abs(prediction[0] - targets[0]),
    actions=lambda prediction: ("allow",),
    state_valid=lambda state: all(isfinite(x) for x in state),
    prediction_valid=lambda prediction: all(isfinite(x) for x in prediction),
    resource_bytes=lambda state: len(pickle.dumps(state)),
    state_digest=lambda state: sha256(pickle.dumps(state)).hexdigest(),
    clock=monotonic,
)
policy = PromotionPolicy("one_minus_absolute_error_v1", "pickle_fixture_bytes_v1",
                         0.6, 0.1, 0.05, 1.0, 1000, ("allow",))
actor = VersionedState("actor-0", [0.5, 0.8])
candidate = CandidateState("actor-0", "candidate-0", [0.75, 0.8], 1, 0, 0)
new = GuardBatch("new", (("new", "1"),), 2, [0], [1.0], "inner_guard")
old = GuardBatch("old", (("old", "1"),), 1, [1], [1.0], "inner_guard")
report = evaluator.evaluate(policy, actor, candidate, new, old,
                            now=2, training_ids=frozenset())
assert report.decision.accepted
# Serving has not changed. This is a local guard report, not a promotion ticket.
```

## Verify and extend

```powershell
python -m pytest -q tests/test_promotion_guard.py tests/test_promotion_guard_evaluation.py
python -m ruff check src tests scripts
python -m ruff format --check src/core/promotion_guard.py src/app/promotion_guard_evaluation.py tests/test_promotion_guard.py tests/test_promotion_guard_evaluation.py
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
```

Implement another outer predictor builder/utility/action decoder and prove owned
restore, numerical checks and same-input measurements. Do not weaken data-role
or timing gates to obtain acceptance. R3.4b must bind policy/evidence to the exact
unchanged candidate and current actor generation, prepare complete serving
model/cache/state off-path, atomically commit under concurrent reads, and restore
the supported full prior bundle on rollback. Stale evidence and failed preparation
must leave serving/version unchanged. The R3.4 parent remains unchecked until
those criteria pass.

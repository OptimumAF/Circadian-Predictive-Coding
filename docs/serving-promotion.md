# Guarded serving promotion and rollback

R3.4b adds a compatible `PromotableActor` and `ServingPromotionController` to the
existing actor/shadow runtime. It uses the R3.4a matched inner guards; a standalone
guard report cannot authorize a swap. No learning equation or scientific protocol
changes. The existing `StableActor` continues to serve its fixed version.

## Structure

```text
src/core/serving_ports.py          cache/configuration/frame/snapshot/ticket/receipt records
src/app/serving_promotion.py       owned cache, guard-issued tickets, atomic swap/rollback
src/app/actor_shadow.py            optional serving actor, fixed candidate base, exclusive lease
tests/test_serving_promotion.py    refusals, cache/ownership, thread barriers, native rollback
docs/adr/ADR-0202-atomic-owned-serving-bundles.md
```

Core records contain no IO or model handle. The app composes existing native ports,
guard policy/evaluator and candidate ownership. Filesystem publication, deployment,
native training equations and final-label release remain outside these modules.

## Approval and commit

1. Construct a trusted, quiescent native source and a promotable actor. Pass that
   same actor to `ActorShadowRuntime(actor=actor)`. Candidate base version stays
   fixed even after the actor changes; this prevents silently rebasing old work.
2. Configure one immutable `PromotionPolicy`, existing `PromotionGuardEvaluator`,
   complete model-state SHA256 probe and native builder. The controller and
   evaluator must use the **same builder callable**. Every returned model must own
   its complete state and implement the same native policy and prediction behavior.
3. `prepare` holds the candidate's exclusive gate. It snapshots the actor's model
   and generation, unions declared training IDs with actual committed wake IDs,
   and evaluates identical new/old inner inputs on detached model copies. All
   configured utility/gain/retention/numerical/latency/byte/action criteria must pass.
   Non-inner, future, overlapping or stale data is refused by the existing evaluator.
4. Prepare another owned native model off the serving lock, restore the complete
   candidate state, and verify its full digest. Copy the non-algorithmic metadata,
   keep the fixed cache configuration and start an empty cache. Issue a bounded,
   identity-checked local ticket with an independent copy of the measured report.
5. `commit(runtime, ticket)` leases the candidate again. Check runtime identity,
   full report/policy/builder identity, base/candidate versions, all three revision
   counters, candidate and prepared model digests, actor digest and generation.
   Under the serving gate, prepare a receipt and slot, then publish **one pointer**.

Generation is separate from human-readable model version. It increases on every
promotion and rollback, so an old approval cannot become valid again when rollback
restores an earlier version. Tickets are single-use, local and nonpersistent;
constructing/copying a ticket or importing another report grants no authority.
Discard obsolete tickets to release their models. `max_pending` bounds retained
preparations. Nested/concurrent controller or candidate operations fail `busy`;
serving uses neither outer gate. Callers may defer and retry an unchanged ticket
after a prepublication busy/transient probe failure.

## Complete supported serving state

The slot owns current model, version, cache configuration, prediction-cache entries,
audit metadata and last cache tick. It also retains the previous complete bundle
for one-step rollback. The native state includes whatever the native snapshot
supports: full parameters, topology, aliases, chemistry/traffic/RNG/replay fields,
not just weights. Native adapter training/inference policy is configured identically
through the same trusted builder; it is not inferred from a weights hash.

`serve(features, now=logical_tick)` uses a configured full feature SHA256 identity
and a bounded cache. The identity callback must include every prediction-relevant
input and produce collision-resistant, deterministic keys. Each miss stores a
detached prediction with expiry `now + cache_ttl_ticks`; equality at expiry is a
miss. Purge expired entries and evict the oldest insertion when capacity is reached.
Promotion invalidates the cache. Rollback restores the previous entries and their
original expiry ticks; they can already be expired at the next current tick.
Ticks must be nondecreasing within the current bundle. Rollback restores its prior
tick as part of complete state; the external logical clock is not rewound.

Cache reads replace their bundle copy under the same gate. A promotion retains the
prior bundle **at actual commit time**, including cache fills since assessment.
Serving metadata is detached observable audit context only; it cannot affect model
prediction, action decoding or guard thresholds. Algorithm-affecting routing state
requires its own matched guard extension before support. Configuration and feature
identity semantics stay fixed for this actor's lifetime.

`predict`, `version` and `snapshot_state` remain compatible uncached interfaces.
`serve` returns one atomic frame with model version, generation, prediction, metadata
and cache-hit flag. `serving_snapshot` returns an owned complete observation. Separate
calls are separate observations; use a frame rather than separately reading version
and prediction when their consistency matters.

## Rollback and failures

`rollback(receipt)` accepts only this controller's latest genuine, unchanged receipt
at the current generation. Publish the retained prior bundle with a new generation,
consume its rollback capability and retain no deeper history. A later commit makes
earlier rollback receipts stale. No native restore or user callback runs during
rollback, and no callback runs after the commit publication point. Native build,
restore, snapshot, digest, metadata copy, rejected/stale tickets and busy gates all
fail before publication; actor version/model/cache/metadata remain unchanged.

Why this: restoring a live model can fail after partial mutation. Preparing an owned
model off the serving path and retaining the exact prior owned bundle makes rollback
a complete pointer transition. Current/prior models are never exposed to callers.
This uses trusted native ownership and prediction-purity contracts; arbitrary hostile
Python closures, mutation through private fields, or malicious shared object graphs
are not certified. Digest checks catch changed retained/prepared state at boundaries,
but cannot turn arbitrary in-process code into a secure capability system.

After promotion, construct a fresh candidate runtime from an independently owned
copy of the new actor's native state/policy and use its current actor version. The
old candidate inbox keeps its original base version and is refused for new approval.
It is not automatically retrained, rebased or cleared.

## Local example

This fixed two-coordinate fixture demonstrates transactions, not an experiment.

```python
from hashlib import sha256
import pickle
from src.app.actor_shadow import ActorShadowRuntime
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import PromotableActor, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.experience import LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.promotion_guard import GuardBatch, PromotionPolicy
from src.core.serving_ports import ServingConfiguration

class Learner:
    def __init__(self, state): self.state = list(state)
    def fork(self): return Learner(self.state)
    def snapshot_state(self): return self.state[:]
    def restore_state(self, state): self.state = list(state)
    def predict(self, features): return [self.state[features[0]]]
    def train_batch(self, features, targets):
        raise ValueError("this example uses detached consolidation only")

def digest(state): return sha256(pickle.dumps(state)).hexdigest()
source = Learner([0.5, 0.8])
actor = PromotableActor(source, version="actor-0", feature_digest=digest,
    configuration=ServingConfiguration(3, 2), metadata={"task": "old"})
runtime = ActorShadowRuntime(source, actor=actor, actor_version="actor-0",
    candidate_version="candidate-0", clock=LogicalClock(),
    budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0))
runtime.consolidate("fixture", lambda state: ConsolidatedState(
    [0.75, 0.8], TrainingDiagnostic("fixture_coordinate_change", 0.0)))
policy = PromotionPolicy("one_minus_absolute_error_v1", "serialized_state_bytes_v1",
    0.6, 0.1, 0.05, 0.1, 100, ("allow",))
evaluator = PromotionGuardEvaluator(build_learner=Learner,
    utility=lambda p,t: 1.0-abs(p[0]-t[0]), actions=lambda p: ("allow",),
    state_valid=lambda s: all(0.0 <= v <= 1.0 for v in s),
    prediction_valid=lambda p: all(0.0 <= v <= 1.0 for v in p),
    resource_bytes=lambda s: len(pickle.dumps(s)), state_digest=digest, clock=lambda:0.0)
controller = ServingPromotionController(actor, policy=policy, evaluator=evaluator,
    build_learner=Learner, state_digest=digest)
new = GuardBatch("new", (("new", "s1"),), 2, [0], [1.0], "inner_guard")
old = GuardBatch("old", (("old", "s1"),), 1, [1], [1.0], "inner_guard")
assert actor.serve([0], now=2).prediction == [0.5]
ticket = controller.prepare(runtime, new, old, now=2, training_ids=frozenset(),
    metadata={"task": "new"})
receipt = controller.commit(runtime, ticket)
assert actor.serve([0], now=2).prediction == [0.75]
assert controller.rollback(receipt) == 2
assert actor.serve([0], now=2).prediction == [0.5]
assert actor.serving_snapshot().metadata == {"task": "old"}
```

## Verification and extension

```powershell
python -m pytest -q tests/test_serving_promotion.py tests/test_promotion_guard.py tests/test_promotion_guard_evaluation.py
python -m ruff check src tests scripts
python -m ruff format --check src/core/serving_ports.py src/app/serving_promotion.py src/app/actor_shadow.py tests/test_serving_promotion.py
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
```

The native integration uses fixed seed23, rates0.03/0.2, width4 and inference2 for
both methods. Its predeclared permissive utility/gain/retention fixture thresholds
exercise the transaction independently of which method improves; they are not a
deployment threshold or performance claim. Separate strict fake guard controls
reject every criterion. No baseline, seed or metric is tuned to obtain an advantage.

Configured byte limits retain their explicit probe meaning (native serialized state,
not RSS). Entry quotas/TTL are enforced; arbitrary payload bytes are not a hard
allocation bound. Guard timing measures candidate guard calls, not live serving
p50/p95. R3.5 must measure actual serving latency during candidate work, prioritize
serving, defer contention and checkpoint supported cursors before claiming live
resource-sharing performance. Preserve outer/final isolation and all existing
scientific provenance/refusal gates when extending these boundaries.

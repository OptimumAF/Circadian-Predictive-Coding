# Repeated runtime fault sequences

R3.7a tests repeated supported in-process failures against the existing actor,
checkpoint, sharing, consent, byte-budget and serving boundaries. Each fake case
uses one or two finite phases of64 fault probes. No internal retry loop is added.
The separate fixed native stream uses eight cycles for each existing BP/CPC
adapter, one seed/configuration, two checkpoint handoffs and explicit purge after
each cycle. It checks availability, spent authority and observed memory.

Why this: passing one isolated refusal does not establish stability after repeated
failures or after handoff. These tests keep their initial authority and verify
that failure cannot renew attempts, work, records, copy bytes or consumed IDs.

## Files and responsibilities

```text
tests/test_runtime_failure_sequences.py  finite faults and reserved native stream
docs/runtime-failure-sequences.md        scope, usage, commands and remaining work
docs/adr/ADR-0214-bound-repeated-runtime-fault-sequences-before-crash-recovery.md
```

No production interface, dependency, environment variable, scheduler, native
equation or source changes are required by the final validation. The fixture
uses the existing inward runtime composition and trusted fake ports. The native
test keeps sampler/authority setup, cycles and guaranteed trace cleanup in one
scope so failures cannot lose the original resource owner. It is test
orchestration, not a new production runtime module.

## Covered sequences

| Fault sequence | Required outcome |
|---|---|
| Rejected consolidation transforms, then handoff | Attempt IDs remain spent; quota refuses; actor stays available |
| Rejected promotion guards | No pending/rollback growth or serving swap; tracked allocation peak below4MiB |
| Corrupt checkpoint observation | No preparation starts; original owner/state remains; explicit discard removes token |
| Repeated failed checkpoint builder | Four original attempts, then refusal; no model growth or implicit retry |
| Partial failed update with RuntimeError/ValueError/KeyboardInterrupt | Candidate stops; stopped handoff cannot retry; original actor remains available |
| Oversized labels and revoked-data retry | Denial before copying/training; deletion does not refund bytes |
| Labels mismatching known experiences and backwards cache ticks | Refusal before target copying/native prediction; serving/cache state remains stable |
| Eight native arrival/update/serve/purge cycles | Stable actor output, same clock/budget/gate, preserved retired work, bounded sampled RSS/tracked allocations/owned-array copies |

Label-first arrivals may legitimately carry another model version until an
experience establishes pair authority. The first new test incorrectly assumed
all unpaired labels must match candidate base. Existing version-rule tests exposed
that assumption. Preserve the version-neutral inbox and supported replay behavior;
test stale labels against a known experience. The rejected production restriction
and failed assumption are retained in the local evidence directory.

## Example (no native training or prediction)

```python
import numpy as np
from src.adapters.numpy_learners import BackpropLearner, ManagedNumpyBuilder
from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.backprop_mlp import BackpropMLP
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.resource_sharing import SharingLimits
from hashlib import sha256
import pickle

class CannotCopy:
    def __deepcopy__(self, memo):
        raise AssertionError("stale targets were copied")

source = BackpropLearner(BackpropMLP(2, 4, seed=23), learning_rate=0.03)
budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
runtime = ActorShadowRuntime(source, actor_version="actor-0", candidate_version="candidate-0", clock=LogicalClock(), budget=budget, max_experiences=1)
gate = ServingPriorityGate(SharingLimits(2, 1, 1), resource_available=lambda: True)
shared = ResourceSharedRuntime(runtime, gate)
runtime.record_experience(Experience("sample", "episode", 0, "actor-0", np.array([[0.3, -0.2]]), "train", ExperiencePermissions(training=True)))
for i in range(64):
    try:
        runtime.record_label(LabelArrival("label-" + str(i), "sample", "episode", 0, "another-actor", CannotCopy()))
    except ValueError as error:
        assert "version" in str(error)
    else:
        raise AssertionError("mismatched label was admitted")
assert budget.updates_completed == 0
gate.pause()
digest = lambda value: sha256(pickle.dumps(value)).hexdigest()
controller = CandidateCheckpointController(shared, build_learner=ManagedNumpyBuilder(source), state_digest=digest, policy_digest=lambda model: digest("fixed"))
token = controller.capture()
assert controller.inspect(token).inbox.labels == ()
controller.discard(token)
```

## Reproducibility and memory limits

The native successor fixes seed23,input2,width4,rate.03,CPC inference2/rate.2 and
the original two rows/targets. Sixteen updates and sixteen cache-miss prediction
calls total, zero native consolidation/sleep; no model/seed selection or sweep.
Original per-model caps: eight declarations/updates,8192 copied-array bytes,
2048 ingress bytes and four checkpoint preparations. After eight cycles,64 quota/
duplicate probes refuse without new native work. Original IDs and retired work
counts survive the two handoffs. Source models stay exact.

Windows process RSS uses the existing reader/sampler and an absolute512MiB bound;
observed growth must stay within32MiB from sampler start. Tracked native-case
allocation peak must stay below8MiB. These are prospective finite-run observations,
not a hard allocator/preemption certificate, long production deployment, RSS
attribution or a claim that Python tracing sees all native allocations. Owned
array accounting and process/trace measurements remain distinct. Sampler cleanup
is verified after the context exits.

## Commands and next extension

```powershell
python -m pytest tests/test_runtime_failure_sequences.py -q -k "not eight_native_stream_cycles"
python -m mypy --platform win32 --no-incremental
python -m mypy --platform linux --no-incremental
python -m ruff check src tests scripts
```

Exact affected regression/format commands and the single reserved native capture:
`artifacts/runs/r37a-inprocess-20261007/`. Both type targets use the Windows
interpreter. Broader runtime matrices, clean clone and global formatter debt stay
separate. Do not rerun a spent capture or increase bounds after a failure.

Full R3.7 remains unchecked. Injected exceptions are not actual process crashes.
Authentic durable recovery remains R3.5b2: specify monotonic elapsed/resource/source/
consent authority and single ownership before disk restore. Sustained deployment,
failure under process loss, durable non-resurrection and broader long-run memory
coverage remain unproven. Preserve original R3.7 criteria and all scientific/
human deferrals; this validation authorizes no algorithm expansion.

# Isolated LS task/head transfer probes

The existing in-memory probe API now accepts its original fixed transfer task
list with LS heads:

```python
from scripts.context_head_probe import build_head_cohorts
from scripts.probe_ticket_policy_learning import run_arm, TRANSFER_HEAD_TASKS

cohorts = build_head_cohorts(spec, head_task_transfer=True)
row = run_arm(
    source, source.snapshot_state(), cohorts, spec, check_boundary, memory,
    "continued", separate_context_heads=True, linear_fit=True,
    head_tasks=TRANSFER_HEAD_TASKS,
)
```

`source` is the current owned D9/K6 contextual LS learner. The supplied boundary
guards its caller/environment/benchmark/deployed state; all updates occur on the
existing disposable FreshTaskProbe fork. There is no saved-source loader or CLI
workflow added here. Prior native counters come from the actual immutable source
snapshot, including further-trained donors. Perceptron's original warm12/72 and
fresh0/0 transfer contracts stay unchanged. Default LS legacy probes also retain
their original behavior.

The outcome-free selected new task/head pairings are parcel-v1 and password-v2;
old tasks remain invoice/password-v1 and invoice/parcel-v2. Initial, after-probe
and after-return task readouts cover those exact six tasks as `probe_test`, with
no online updates. Initial ability is MEASURED, never presumed absent. These
pairings share global concepts and workflow stages; selecting them is not proof
that an arbitrary donor never saw them or that they are unseen/disjoint data.

Admission checks all original16/2/2/2 cohort groups, roles, class/feature geometry,
context order and exact supplied feature/label arrays against six authentic
sandbox traces. It never substitutes those labels or converts data roles.
Those six trace checks are diagnostic work performed once BEFORE the probe
sessions are born, not inside repeated source guards. Subsequent guards bind
original array owners/values, source/spec/RNG and selected-task report metadata.
Wrong legacy cohorts or altered labels/head bits refuse before a learner fork.

Original probe16updates/48rows and separate old-TRAIN return16updates/96rows use
the existing runner. Both HEAD_LIMITS16-update/30-second/512MiB sessions are born
once at arm entry; return time includes the preceding probe. Checkpoints0/4/8/16,
CE/top1/decision metrics, old-TRAIN monitors and LS residual diagnostics are
unchanged. Probe/return failures keep permanent charges and close later phases;
final evaluation stays unavailable. No replay, promotion or deployed updates.

## Prospective #302 software validation

Pure semantic/role/metadata refusals precede native fixtures. One complete
continued-donor arm per platform costs32 physical LS solves/144 fitted rows:
16probe3-row solves followed by16return6-row solves. Donor zero coefficients
and23/144 counters are handwritten fixture metadata, NOT observed source
training or scientific qualification. Its deployed StableActor and benchmark
sibling must stay unchanged, as must source inputs and NumPy/Python RNG.
Initial and both endpoint results are diagnostic observations, not win criteria.

Shared engineering allowance<=96LSsolves/480fitted rows, forecast64/288; no
scientific comparative cohort granted. At most3 complete cohorts/platform,
120seconds/512MiB/process, original probe/return30s budgets unchanged.
Stage/output<=8MiB. No Torch/GPU/optimizer/latent/modelrandom work or seed changes,
downloads/installs/deletion/source loader. Semantic teachers and all diagnostic,
monitor, probe/return and readout costs count. Existing default/native baseline
results are reused; only relevant pure compatibility checks run. Finite reviewer
<=4 readonly source/result calls<=20seconds, no imports/tests/models/network/writes;
lead owns all edits and shared budgets. Preserve #285's forgetting result and
#287 FAILED/SPENT. A later observation on an actual continued donor needs its own
small frozen/reviewed protocol and measured initial ability; no automatic run.

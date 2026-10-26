# Detached neural phase readouts

The existing `run_continuous_learnability` and JSON-only wrapper accept
`checkpoint_observer=None`. None preserves the previous report schema and call
shape. A supplied callback is called at the existing four diagnostic boundaries:
initial, after warm A, after new B, and after return A.

```python
from src.adapters.continuous_learnability_demo import run_continuous_learnability

def observe(model, metadata):
    # model is detached; no labels, actor or runtime authority are supplied.
    return {"phase": metadata["phase"], "updates": metadata["updates"]}

result = run_continuous_learnability(
    warm_train, new_train, learner_family="bp", head_gauss_newton=True,
    checkpoint_observer=observe,
)
print(result.report["phase_readouts"])
```

Why this: the runner previously returned only the final neural policy. Frozen
native comparisons of before/after learning need intermediate policies. This
small callback follows the existing nearest-stream observer pattern and keeps
all learning in the existing runner; it does not add another scheduler.

## Contract and isolation

The callback receives an exact native BP or PC learner fork and an owned metadata
dictionary with `checkpoint_index`, `phase`, `updates`, `training_examples`.
Forks use no initializer or RNG. Parameters, momentum, policy and any CPC wake
state are owned and must remain unchanged during the callback. Retained earlier
readouts do not acquire later training labels or updates. Native serving can
make its own frozen fork through the existing serving API.

Return an exact finite JSON dictionary at most64KiB, using native JSON values.
Returned values never select, stop, tune, promote or update stream training.
Metadata changes, model changes, callback exceptions, invalid JSON or corruption
of protected benchmark state fail the run and release no candidate. Paid return
and preceding training receipts are retained before later checks. Corrupted live
models, budgets and clocks are not restored away.

The integration guards private source/actor/candidate/readout state, fixed
runtime/inbox versions and roles, native recording sinks, original budget and
clock, paid receipts and scalar benchmark evidence. At most four scalar records
are retained. This is an ordinary Python callback API, with the same trusted
local caller boundary as existing experiment hooks.

Initially use plain TRAIN A/B/A with BP, PC, neutral CPC, active CPC, or explicit
BP head GN. Observers with replay, fresh-task probes, RGB range normalization,
stage-balanced packets or A-only control are refused before fitting/model work.
Both input episodes still require released TRAIN roles. Probe/final labels have
no role in this interface or in training decisions.

## Costs and errors

`phase_readouts` reports attempts, returns, validations, forks, bounded records
and helper work. Returned valid JSON is copied before post-callback validation.
An exception keeps the original callback error; a subsequent guard failure is
also recorded. Observer failure keeps last validated training progress and paid
receipts, while the actual corrupted state remains observable internally.

Helper work counts logical snapshot array payload copies, fork bytes and result
bytes. These are cumulative logical copies, not peak memory. Root boundary
guards, validation, Python/JSON objects and other copies are additional.
`unknown_callback_work` explicitly marks computation performed by callbacks as
opaque. Benchmark `work` remains the learner/diagnostic work; callback prediction,
simulator and other work must be separately budgeted and reported by the caller.
No scientific or native execution allowance comes from enabling the option.

## Structure and scoped checks

```text
src/adapters/continuous_phase_readouts.py       detached callback and scalar records
src/adapters/continuous_learnability_demo.py    existing phase integration/guards
tests/test_continuous_phase_readouts.py         phase/isolation/failure checks
```

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_continuous_phase_readouts.py tests/test_continuous_head_stream.py
.venv/Scripts/python.exe -m ruff check src/adapters/continuous_phase_readouts.py src/adapters/continuous_learnability_demo.py tests/test_continuous_phase_readouts.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/continuous_phase_readouts.py src/adapters/continuous_learnability_demo.py tests/test_continuous_phase_readouts.py
.venv/Scripts/python.exe -m mypy --platform linux --follow-imports silent src/adapters/continuous_phase_readouts.py src/adapters/continuous_learnability_demo.py tests/test_continuous_phase_readouts.py
```

Freeze and independently review a small protocol before extending this to paired
native control. Keep the seed, updates, metrics and comparison directions fixed,
retain every failure, and account for observations separately. Existing known
TRAIN17 success and the offline retention tradeoff remain distinct evidence;
observer software correctness alone does not prove useful adaptation.

## Accepted execution and scientific scope

33new+23affectedstream checks passed on Windows and pinned Linux; scoped statics
and independent actualreview accepted. [Software evidence/costs](../artifacts/runs/continuous-phase-readouts-20261021/RESULTS.md).

One separately frozen nativequestion used the four existing phases: Acontrol
succeeded afterAlearning, Bfailedbefore/afterBlearning, and AfailedafterBlearning.
The accepted final-A return was reused to demonstrate knownA recovery.165new
controls825knownphysics, allprimarymetrics and negativeoutcomes retained.
[Fullstudy](../artifacts/runs/gn-native-adaptation-20261021/RESULTS.md). Observer
correctness does not establish useful Btaskacquisition, reliability or CPCbenefit.

## Executed isolated GN checkpoint probe composition

The artifact-local gn-checkpoint-probes-20261021 case calls the existing
run_continuous_probe on each detached observer fork, with original reserved23/24
ContinuousProbePlan(updates=1,diagnostics=True) and TRAIN A/B score readouts.
The probe uses default SGD on its own learner, returns only bounded reports,
and never changes the source or chooses benchmark work. Direct GN probe_plan
remains refused; no production interface change is required. Source/paid-report
seals and full original benchmark comparison passed. Four positive train/test
gains coexist with mixed old-task costs, without native/generalization claims.
[Exact commands, controls, costs and evidence](../artifacts/runs/gn-checkpoint-probes-20261021/RESULTS.md).

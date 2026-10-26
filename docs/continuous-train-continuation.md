# Continue a learned continuous BP model on released TRAIN data

## Current: all148 corrective rows

Explicit mixed scopes retain50correction+18original expert+80old TRAIN rows;
head130/default8 remain unchanged. One fixed eight-update fit reaches120/11016.
279focused checks pass on each Windows/Linux, scoped statics and independent
actual review pass. The fixed physical trial is negative: Push17 and all three
Picks fail50controls, losing112 Pick17 success. No useful adaptation follows.
[Usage/tree/extension notes](push-cube-learned-continuation.md) ·
[Exact commands, costs, failures and review](../artifacts/runs/mixed-correction-fit-20261021/RESULTS.md).

## Previous increment: finite longer mixed scopes

Explicit `update_rule="bp_mixed"` now accepts native integer `updates`1..64;
default8 and common-MSE/head-GN maximum8 are unchanged. Booleans and65 refuse
before forking. Existing source/actor/seed/normalization/momentum guards, permanent
charges, exact delayed receipts and validation of preceding native evidence apply
at every update. Each invocation has its own prospective finite budget; it never
renews the earlier scope. Storage is64 scalar records and one latest native pin,
not64 retained model snapshots. No numerical kernel or dependency changed.

```python
longer = run_continuous_train_continuation(
    mixed_candidate, released_push_train,
    references=old_train_episodes, updates=64, update_rule="bp_mixed",
)
```

Use only under a separately frozen workload/protocol. An accepted98-row64-update
scope reaches112/9832 from48/3560, preserving the original source and actor. It
improves Push TRAIN fit but worsens all old group means; native usefulness remains
unmeasured.193 focused cases pass on each Windows/Linux (44legacy+16longer+
133study); exact full-tree static/test commands, costs, failures and independent
review are in [RESULTS](../artifacts/runs/mixed-longer-train-fit-20261021/RESULTS.md).
New behavior tests live in `tests/test_continuous_mixed_longer_continuation.py`.
Extend this API by preserving the existing recorder/finite-budget mechanics and
qualifying paid late failures before freezing a new numerical protocol.

## Explicit private mixed-gripper learning

The fixed Pick-only successor is `--profile bp_mixed_retention`. It pays the same
source/candidate reconstruction, matches whole source/data/continuation reports
(only two descriptive duration values differ), reuses the accepted negative Push
trace from the failed original study, and executes only Pick17/19/20. A separately
frozen protocol is required. Original failure and charges stay unchanged. One accepted successor observes
Pick17/20success16/14controls and Pick19failure50controls; Push remains negative.
The local retention question is answered negatively overall, not useful adaptation.
[Retention protocol](../artifacts/runs/mixed-pick-retention-20261021/PROTOCOL.md).

The existing fixed TRAIN Push study can select the private mixed method:

```powershell
python -m scripts.push_cube_continuation_study fresh-output.json --profile bp_mixed
```

Run only under a separately frozen local protocol. It binds the accepted PLC
control, verifies reconstructed source/data reports (only the descriptive stream
duration may differ), reuses historical pre-update Push, then runs eight mixed
updates and four new frozen scenes. Historical controls retain their original
costs; they are not new simulator calls. Omitting the flag preserves the original
head-GN study. One frozen run completed8mixedupdates but Push still failed50
controls; the enclosing study failed on list/tuple reset comparison before Pick.
The comparison is repaired and independently accepted. Retention was unobserved
in that original run; the separately budgeted Pick-only successor above now
records Pick19failure. Original failed run and its closed allowance stay immutable.
[Results and repair](../artifacts/runs/mixed-push-study-20261021/RESULTS.md).
[Prospective physical protocol](../artifacts/runs/mixed-push-study-20261021/PROTOCOL.md).

`update_rule="bp_mixed"` is an opt-in extension, now executed with17new+27existing
focused checks passing on each Windows/Linux runtime and scoped statics passing.
Independent actual source/result review accepted the scoped software behavior. It uses the existing all-layer native
BP mixed XYZ-MSE/gripper-BCE objective; head_gauss_newton remains the default.
The source may be common-MSE or alreadyMixed(axisexact3). All original source and
stable actor policy/parameters/momentum/counters remain unchanged. Only a private
copy changes objective, explicitly recorded. Frozen normalization and original
optimizer momentum carry forward; no initializer, fitting or extra prediction.

```python
mixed = run_continuous_train_continuation(
    learned_source, new_train_episode,
    references=(old_train_episode,), updates=8,
    update_rule="bp_mixed",
)
if mixed.report["complete"]:
    mixed_candidate = mixed.candidate  # detached; no promotion or deployment
```

TRAIN-only admission accepts1..148rows for explicit mixed continuation and
1..130 for head-GN, with0..8references. Why148: retain all50 released correction,
18original Push expert and80old TRAIN rows together; see
[ADR-0505](adr/ADR-0505-retain-all-released-correction-fit-rows.md). Updates are
1..64 for explicit mixed scopes,1..8 for the default head-GN path.
The mixed path additionally requires exact endpoint gripper labels(-1,+1) on
normalized_gripper axis3 before any fork. Unsupported rules and final/probe roles
are refused. An already-mixed candidate can continue in a new explicitly bounded
scope; it does not renew an earlier scope's budget. The extra immutable private
training seed gives5named owners, with bounded receipts and one latest model pin.

Mixed reports use continuous_train_mixed_gripper_continuation_v1, training_rule
bp_mixed and policy_transition metadata, not head-GN acceptance or loss_after.
Work lists SOURCE-DERIVED successful SGD counters; original native failure.work
remains separate. A BCE value must not be compared directly to an MSE value.
The controlled bias20 fixture demonstrates finite gradient/logit movement without
claiming8updates flip that endpoint or that any physical task succeeds.

```text
src/adapters/continuous_train_continuation.py   existing API/runtime integration
src/adapters/continuous_mixed_training.py      private objective copy and bounded SGD records
src/adapters/numpy_continuous_bc.py            native SGD birth/commit source guard
tests/test_continuous_mixed_continuation.py    endpoint/isolation/paid-failure checks
```

```powershell
python -m pytest -q tests/test_continuous_mixed_continuation.py tests/test_continuous_train_continuation.py
python -m ruff check src/adapters/continuous_train_continuation.py src/adapters/continuous_mixed_training.py src/adapters/numpy_continuous_bc.py tests/test_continuous_mixed_continuation.py
python -m ruff format --check src/adapters/continuous_train_continuation.py src/adapters/continuous_mixed_training.py src/adapters/numpy_continuous_bc.py tests/test_continuous_mixed_continuation.py
python -m mypy --platform linux --follow-imports silent src/adapters/continuous_train_continuation.py src/adapters/continuous_mixed_training.py src/adapters/numpy_continuous_bc.py tests/test_continuous_mixed_continuation.py
```

Why this: address an observed learning gap with already implemented BP math,
keeping the existing runtime and source/actor separation. No new dependency or
application configuration is introduced. Qualify changes with focused behavior/
isolation/work tests, then freeze a small separate protocol before native use.
[Protocol, costs, failures and evidence](../artifacts/runs/mixed-gripper-continuation-20261021/RESULTS.md) ·
[ADR-0503](adr/ADR-0503-private-mixed-gripper-continuation.md).

## Previously accepted default head-GN path

Scoped software behavior is executed and independently accepted: Windows26
original passes plus one repaired refusal case, Linux27PASS on the final tree,
with all original failures and costs preserved. This is a Python API for trusted,
already released model-space training episodes. Native task usefulness and
comparative scientific claims require a separate frozen study.

```python
from src.adapters.continuous_train_continuation import run_continuous_train_continuation

result = run_continuous_train_continuation(
    learned_source,
    new_train_episode,
    references=(old_train_episode,),
    updates=8,
)
if result.report["complete"]:
    adapted_candidate = result.candidate
else:
    # Preserve the report: incomplete work may already have spent computation.
    paid_failure = result.report
```

The exact native BP source must use the original common MSE, TRAIN role, at most
64 input and 64 hidden coordinates, and no binary-gripper override. Its existing
observation standardization stays frozen. There is no initialization or fit;
source parameters, momentum, counters and serving actor remain unchanged. The
successful candidate starts at the source's original counters plus actual new
work. It is detached from the source and private runtime.

`training` is one complete exact `ContinuousEpisode`; `references` is an exact
tuple of zero through eight such episodes. All must have TRAIN role, match the
source specification, have distinct episode IDs, and total at most 130 rows.
All are owned and admitted before forking. A raw robotics trajectory needs its
existing qualified projection before this API; no representation is inferred
from shapes, and no probe, guard or final record is converted to TRAIN.

Each of one through eight updates uses the ordered new rows followed by every
reference row, with equal row weight. References add exposure and computation;
they do not imply matched compute with a new-data-only control. For example,
10 new rows and 15 reference rows over eight updates cost 200 training-row
exposures: 80 new and 120 reference. This API does not select references, tune
the dose, promise old-task retention, or read evaluation outcomes.

The existing delayed-label runtime observes a batch at `1000 + 2*i`, polls with
no labels, then releases that update's label at `1001 + 2*i`. These are derived
exposure clocks; they do not replace or reconstruct original acquisition times.
One original update budget covers the scope. A failed native proposal or later
validation stops the scope; no fallback model, budget renewal or retry occurs.

Reports retain input identities and row groups, original and final counters,
delayed-label events, native work and receipts, selected memory costs, and paid
failure evidence. Native solve counts and row exposure are distinct from total
CPU time, allocations, solver arithmetic and process RSS. A missing native
return can leave unknown work; do not count it as zero. Storage is bounded by
the eight-update scope and one latest native model pin.

Why this: the learned stream already has a private actor/shadow runtime and a
bounded Gauss–Newton head updater. This adapter lets a caller supply an existing
learned model and newly released data without rebuilding the fixed acquisition
recipe or changing its defaults. It performs no simulator work or deployment.

## Files and verification

```text
src/adapters/continuous_train_continuation.py  owned training inputs and runtime composition
tests/test_continuous_train_continuation.py   tiny real updates, refusal and isolation checks
```

The focused correctness run itself spends model work. Its prospective limits,
exact commands, outcomes, failures and review are recorded in
[the executed evidence](../artifacts/runs/continuous-train-continuation-20261021/RESULTS.md).
No scientific or native allowance follows from a passing software test. Extend
through the existing learner/runtime contracts; keep final evaluation and private
probes outside this adapter.

```powershell
python -m pytest -q tests/test_continuous_train_continuation.py
python -m ruff check src/adapters/continuous_train_continuation.py tests/test_continuous_train_continuation.py
python -m ruff format --check src/adapters/continuous_train_continuation.py tests/test_continuous_train_continuation.py
python -m mypy --platform linux --follow-imports silent src/adapters/continuous_train_continuation.py tests/test_continuous_train_continuation.py
```

The executed serial fixture commands include their RSS/ledger wrappers and are
recorded in the evidence. Across original Windows, its narrow repair and Linux:
23GN updates,344row exposures,46solves826augmented rows, three tiny source
initializers/24source-derived normal values. No explicit prediction, SGD, fit or
simulator work. The original Windows invalid binary-policy fixture remains a
failed run. A passing continuation is not a promise of old-task retention.

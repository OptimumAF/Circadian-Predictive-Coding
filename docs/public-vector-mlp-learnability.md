# Public21 MLP learnability

The existing continuous learnability runner can consume fixed public21 views of
qualified raw RGB TRAIN episodes. The stream CLI now exposes that input choice.
It preserves the existing MLP, optimizer, delayed-label runtime and fixed
A8/B8/A8 recipe. It introduces no new learner or orchestration framework.

## Accepted offline outcome

The single fixed24-update arm completed and passed independent artifact readback.
Final physical-action half-MSE was worse than unchanged RGB-MSE in both contexts:

| Context | Public21 | RGB-MSE | Public21 minus RGB |
| --- | ---: | ---: | ---: |
| A17 |.0482477274 |.0446707855 |+.0035769418 |
| B19 |.1944966594 |.1811039013 |+.0133927581 |

B forgetting worsened; A retention and some axis errors improved. All four
endpoints and six metrics remain in the evidence. Model arithmetic fell to
4381440 dense MACs versus131586816; this is a cost/error tradeoff. Wrong-sign
gripper examples remain. These are same-PickCube reset contexts, not distinct
tasks. No native success or neural generalization is established.

61focused cases pass on each platform, including52new; statics and independent
source/gate/actual review accept the scoped behavior. The initial filepath launch
failed at import before any data/model/training. Its receipt is preserved; a
reviewed module-only correction completed within the original aggregate30s,
using the entirely unspent scientific budget. No further arm was run.

## Structure

```text
src/core/pick_cube_public_vectors.py         accepted pure owned input view
src/adapters/continuous_learnability_demo.py existing generic24-update runner
scripts/run_native_stream_demo.py           explicit CLI selection and costs
tests/test_public_vector_learnability.py    synthetic learning/routing/refusal
artifacts/runs/public-vector-mlp-learnability-20261019/
  run_case.py                               fixed offline comparison composition
  PROTOCOL.md / REVIEW.md / RESULTS.md       limits, review and observed evidence
```

Why this: A-only public21 nearest memory suffices for the known camera fixture,
while earlier neural native results remain negative. The existing runner already
supports matching dimensions; adding explicit CLI input selection makes the
next learner question runnable without changing the algorithm or inventing a
runner. Strong neural/generalization requirements remain open.

## CLI

```powershell
python -m scripts.run_native_stream_demo --scenario continuous-mlp-learnability --warm-teacher accepted-A17.json --new-teacher accepted-B19.json --visual-observation --public-vector-only
```

The input files remain qualified raw789D TRAIN17/TRAIN19 teacher receipts. Full
source validation, including dropped RGB and terminal observations, precedes
owned21D projection and fitting. Roles, identities, controls, rewards and flags
are preserved. The explicit report adds `input_projection`; `input_observation`
identifies `public_vector21` model input and `public_rgb` source. Projection
validation/returned numeric payloads are separate from training/diagnostic work.

The flag requires visual `continuous-mlp-learnability` and refuses RGB-range
normalization before file reads. The existing binary-gripper option can compose
with it through the same runner; the frozen comparison uses default MSE.
Omitting the flag preserves the existing routes, calls and report keys.

Programmatic callers can project with `public_vector_training_episode`, then
call existing `run_continuous_learnability(projected_A, projected_B)`. Its owned
result retains a detached learner only on complete success. Probe/final data
remain unavailable to fitting, tuning and promotion.

## Frozen comparison and interpretation

The stage uses the same accepted A17/B19 source records as the earlier RGB-MSE
sequence, default seed17/hidden64/lr.01/momentum.9 and fixed24 updates. It reuses
the old RGB report with zero new baseline updates. The public21 arm performs
one A-only21D normalization fit; B cannot refit statistics. This is intentionally
a new fit using the same algorithm and A rows, rather than exact restoration
of the earlier789D statistics. Descriptive rounding differences are reported.

Changing dimension also changes first-layer shape and RNG draw layout. Same
seed/default settings do not imply identical initialization or equal computation.
The arm cannot prove that pixels caused earlier failures. Signed differences
use the same physical-action half-MSE and all six acquisition/retention/recovery
metrics, retaining all four endpoints, axis errors and training losses. Complete
negative or inconclusive results are valid. Offline TRAIN fit does not establish
native task reward, generalization, visual learning or CPC advantage.

The complete final parameters and momentum are saved as eight float64 arrays in
bounded exclusive `final-state.npz`; policy/counters remain in `case.json`.
This uses the existing typed snapshot contract and avoids retraining solely to
prepare a later native check. No new archive codec or permission is introduced.
Failures after training retain the returned sequence and paid progress.

## Scope and extension

The fixed case is one-use under its own prospective protocol. A later native
validation must first bind the saved policy/counters/arrays through the existing
typed snapshot API, then freeze a small local protocol. Do not rerun training,
select an endpoint or change settings based on favorable offline losses.

[Exact commands, outcomes, costs and next action](../artifacts/runs/public-vector-mlp-learnability-20261019/RESULTS.md) ·
[Protocol](../artifacts/runs/public-vector-mlp-learnability-20261019/PROTOCOL.md) ·
[Independent review](../artifacts/runs/public-vector-mlp-learnability-20261019/REVIEW.md).

## Focused developer commands

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_public_vector_learnability.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check scripts/run_native_stream_demo.py artifacts/runs/public-vector-mlp-learnability-20261019/run_case.py tests/test_public_vector_learnability.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check scripts/run_native_stream_demo.py artifacts/runs/public-vector-mlp-learnability-20261019/run_case.py tests/test_public_vector_learnability.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux scripts/run_native_stream_demo.py artifacts/runs/public-vector-mlp-learnability-20261019/run_case.py
```

## Subsequent saved-state native evidence

[Accepted zero-retraining validation](public-vector-mlp-native.md) reproduces all42source residual scalars exactly. TRAIN17/19 each returns0/no success after50steps, matching RGB MLP; full state unchanged. This preserves the offline and native negative results.

## Fixed phase-count option

The existing runner now supports [bounded explicit phase updates](continuous-phase-updates.md), default8 unchanged. Historical24-update arms/protocols stay fixed; the option does not renew budgets or establish that more training improves native task skill.

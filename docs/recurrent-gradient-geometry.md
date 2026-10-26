# TRAIN episode gradient geometry

`diagnose_episode_gradients(snapshot, episodes)` reads two gradients of the
existing recurrent learner's mean cross-entropy objective. It reports signed
dot products, norms and cosines jointly and for each of the five parameters.
Zero-norm cosine is undefined (`null` with `zero_norm`), never a claimed alignment.

## Usage and boundaries

```python
from src.adapters.torch_recurrent_gradient_diagnostics import diagnose_episode_gradients

# Existing owned native snapshot; two complete, distinct, same-context TRAIN
# demonstrations with observation/action geometry matching its policy.
report = diagnose_episode_gradients(snapshot, (blue_episode, red_episode))
print(report["losses"], report["joint_geometry"], report["work"])
```

The caller supplies original episode provenance and accounts for repeated data
exposure. The component enforces TRAIN roles and bounded native policy geometry;
it cannot authenticate a caller's naming of a cue or an episode's origin. Probe
and final roles are refused. There is no environment, probe selection, replay,
promotion, deployed learner update or new training objective in this port.

The adapter uses the existing feature conversion and finite-checked forward,
resetting recurrent activation to zero for each complete episode. Detached
parameter clones receive `autograd.grad` of ordinary mean CE with hard class
indices. Existing gradients are never accumulated. It supports `no_grad`
callers through a scoped `enable_grad`; inference mode is refused.

The existing read-only native snapshot validator bounds nested tensors and
optimizer configuration before making an owned copy. That validated copy is
the comparison baseline immediately after validation and after the last native
callback. Unknown input, snapshot, RNG or mode changes are refused without
restoring them. This trusts the existing validator; it does not attest against
malicious replacement of that primitive before it copies.

## Output, cost and failure

Results contain JSON scalars and records only: definition/objective IDs, two
episode loss records, joint and per-parameter geometry, and work counts. No
tensors, arrays or autograd graphs escape. The pure core admits at most five
matching pairs of finite float32 matrices, at most 65,536 cells per tuple,
before scanning or copying values. Float64 accumulation preserves the sign and
range of very small or large finite float32 gradients.

Each completed pair pays one snapshot validation, two forward calls and two
backward calls, with episode rows counted separately. Snapshot validation builds
one temporary SGD configuration validator. There are **zero optimizer steps or
published updates**. Parameter copies, validation, gradient storage and compute
still cost time and memory; a caller must include them in its protocol.

Malformed pair metadata raises `ValueError` before native work. A runtime
failure raises `GradientDiagnosticFailed`; its detached `.report` retains the
attempted/completed work and preceding losses, marks unknown failed work, and
leaves all geometry `null`. An externally changed caller snapshot is not repaired.

## Files and verification

```text
src/core/gradient_geometry.py                         pure bounded reducer
src/adapters/torch_recurrent_gradient_diagnostics.py native TRAIN diagnostic
tests/test_gradient_geometry.py                      analytical math and bounds
tests/test_torch_recurrent_gradient_validation.py    pre-native role/admission
tests/test_torch_recurrent_gradient_native.py        original CE and isolation
```

From the repository root, the ordinary scoped commands are:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_gradient_geometry.py tests/test_torch_recurrent_gradient_validation.py
$gradientPaths = @('src/core/gradient_geometry.py', 'src/adapters/torch_recurrent_gradient_diagnostics.py', 'tests/test_gradient_geometry.py', 'tests/test_torch_recurrent_gradient_validation.py', 'tests/test_torch_recurrent_gradient_native.py')
.venv/Scripts/python.exe -X utf8 -B -m ruff check --no-cache @gradientPaths
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check --no-cache @gradientPaths
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @gradientPaths
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform linux @gradientPaths
```

The stage receipt names preserve original attempts; do not overwrite them to
repeat a command. Its native unit driver is a separately budgeted qualification,
not an automatically renewable experiment. See the [protocol](../artifacts/runs/r21-cue-gradient-20261016/SCOPE.md)
and [recorded outcomes](../artifacts/runs/r21-cue-gradient-20261016/RESULTS.md).
No dependencies, environment variables, default learners or CLI routes changed.

## Existing checkpoint scorer and CLI

Preview the optional composition without constructing learners or environments:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-learnability --cue-gradient-geometry --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --native-activation-health --cue-gradient-geometry --describe
```

The option runs only on recurrent BC's owned checkpoints after all six raw TRAIN
fit scores. It binds the original A/B/C_train blue/red packets, then reports
three pairs in that order. Other learners and disposable fresh-task probes keep
their original behavior. Without the flag, original schemas and calls stay the same.

The full descriptor adds 15 pair calls, 270 forward rows and 270 backward rows,
90 geometry reductions and 321,840 logical float64 feature/target input bytes.
These are repeated TRAIN exposures, with zero added SGD or probe updates.
The new mode's recurrent forward total includes these rows and optional recurrent
activation reads; backward counts remain separate. IDs distinguish the original,
history and activation-health configurations. All original seeds, metrics,
training doses, limits and comparison directions remain.

Raw reports add `cue_gradient_geometry` records and a separate nested work ledger.
Returned scalar payloads, incomplete native error reports and their paid work
survive a later refusal. Private counts are checked before accepting any public
counter change; altered source/state/packets or prior results are refused.
Nested `seconds` times diagnostic port calls; the caller's total time and sampled
RSS also cover packet validation, copying, guards and record storage. Logical
input bytes are not an allocation or peak-memory measurement. Returned geometry
contains no gradient tensor history.

[Composition protocol and observed qualification](../artifacts/runs/r21-cue-gradient-compose-20261016/RESULTS.md)
are separate from the component qualification. The small native integration case
uses one four-update checkpoint; it does not execute or qualify the complete
800-update five-learner/probe run with the new option.

Focused composition checks use ordinary repository commands:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_minigrid_gradient_route.py tests/test_minigrid_gradient_cli.py
$gradientRoutePaths = @('src/adapters/minigrid_learnability.py', 'src/adapters/minigrid_gradient_readouts.py', 'scripts/run_coupled_attribution.py', 'tests/test_minigrid_gradient_route.py', 'tests/test_minigrid_gradient_cli.py', 'tests/test_minigrid_gradient_native.py')
.venv/Scripts/python.exe -X utf8 -B -m ruff check --no-cache @gradientRoutePaths
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check --no-cache @gradientRoutePaths
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @gradientRoutePaths
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform linux @gradientRoutePaths
```

## Scientific limits and extension

Whole-episode positive cosine can hide conflict on a small set of cue decisions
or reflect shared labels. Geometry is local to one snapshot and pair; it neither
explains a failure causally nor establishes learning, retention, recovery or a
circadian advantage. Preserve all signed and undefined results.

Use the existing scorer integration for a separately frozen, small TRAIN
measurement protocol. Reuse accepted fit/probe/health and matched baseline
evidence; keep probe updates
separate and final evaluation unavailable to training and selection. See ADR-0362.

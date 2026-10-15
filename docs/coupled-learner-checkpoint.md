# Complete coupled learner checkpoints

The learner envelope saves both complete native PC/CPC state and its update
settings: learning rate, inference step cap/rate and optional guarded inference
policy. Restoring a captured learner reproduces its next deterministic update,
diagnostic and prediction. Existing `snapshot_state()` still returns the native
`CoupledSnapshot`; the separate envelope adds policy without changing that API.

## Run the fixed example

```powershell
.\.venv\Scripts\python.exe -m scripts.run_coupled_learner_checkpoint --describe
New-Item -ItemType Directory -Path .\artifacts\my-learner-checkpoints
.\.venv\Scripts\python.exe -m scripts.run_coupled_learner_checkpoint --run --directory .\artifacts\my-learner-checkpoints
```

Use a new private directory; existing output names are refused. Six fixed cases
cover PC, neutral CPC and active CPC under fixed-step and guarded inference.
Each warms twice, saves A, continues the original once to B, restores A and
continues once to the same B. The guarded case uses the existing policy with
minimum one step and gradient tolerance1e6 as a deterministic convergence
control. This is an engineering equality check, not a learning-benefit comparison.

The example uses seed23, dimensions2 → 3 → 2 → 1 and fixed train-only data.
It performs24 native updates, at most120 energy evaluations and no process
launches. Six bounded files and compact JSON retain full A/restoredA/B/resumedB
bytes, policies, diagnostics and predictions. CPU thread counts are set to one;
no environment variables, downloads or additional dependencies are required.

## API and ownership

```python
from hashlib import sha256
from src.adapters.coupled_learner_checkpoint import (
    CoupledLearnerCheckpointCodec,
    capture_coupled_learner,
    restore_coupled_learner,
)
from src.core.checkpoint_codec import CodecBinding, CodecLimits

# Independently supplied by the trusted caller; consistency labels are not authority.
binding = CodecBinding(source_sha256, policy_sha256)
limits = CodecLimits(65536, 8192, 4, 16)
codec = CoupledLearnerCheckpointCodec()
checkpoint = capture_coupled_learner(learner)
raw = codec.encode(checkpoint, binding=binding, limits=limits)
expected = sha256(raw).hexdigest()  # Retain independently of later file contents.
restored = restore_coupled_learner(
    raw, binding=binding, expected_sha256=expected, limits=limits
)
```

`learner.checkpoint_policy()` returns detached validated metadata without model
work. Capture requires an exact trusted `CoupledLearner` and a quiescent caller;
it detaches model state and policy. No constructor, update or prediction occurs
during capture/encode/decode. Capture uses the existing trusted native snapshot
copies; the codec is the admission boundary for encoded bytes and arrays. The
unchanged `BoundedCheckpointFileStore` also
accepts this codec.

Restore validates the complete bounded envelope and native state before any
model construction. It explicitly reconstructs PC or CPC, restores every native
field and uses the normal learner constructor to own a separate copy. Native
chemistry, gates, wake history, precisions and both inference modes survive.
Constructor seed0 is temporary initialization overwritten by complete state;
these native training updates have no continuing random state to restore.

Captured rate values are normalized floats. Codec records also roundtrip valid
manually constructed integer metadata exactly. Restore refuses an integer rate
whose float conversion changes its numeric value, before model construction.
Lossless integer rates use the existing constructor's float normalization; raw
numeric types for arbitrary manual records are not promised to remain identical.

## Bounds and failure behavior

The canonical JSON envelope contains fixed version/kind, independently supplied
binding, complete policy and the unchanged native checkpoint body. Exact outer
metadata overhead is subtracted from the native encoded-byte allowance before
its size preflight, finite scans or array encoding. Policy does not receive a
second byte allowance. Existing aggregate array, layer and dimension caps remain.

Decode checks exact input bytes, total size and expected outer SHA first. Exact
schema, duplicate keys, policy, binding, version and canonical form are checked
before native array decoding. The inner digest is derived only to adapt the
existing native codec API after the independently expected outer digest matched.
It supplies no additional authentication or recovery authority.

Invalid input fails without building a model. No pickle, arbitrary code selection,
live-owner reconstruction, runtime quota renewal, publication or retry is added.
These files do not capture managed inbox/lifecycle/serving authority, training
data roles, runtime budgets or optimizer state beyond the current plain SGD.
Full managed recovery and comparative scientific benefit remain open.

## Files and checks

```text
src/core/coupled_learner_state.py                policy/checkpoint records
src/adapters/coupled_learners.py                 additive policy getter
src/adapters/coupled_learner_checkpoint.py        capture, codec, restore factory
src/adapters/coupled_learner_checkpoint_example.py fixed continuation controls
scripts/run_coupled_learner_checkpoint.py        CLI
tests/test_coupled_learner_checkpoint.py         synthetic boundary controls
```

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_coupled_learner_checkpoint.py tests/test_coupled_checkpoint_codec.py --deselect tests/test_coupled_checkpoint_codec.py::test_should_continue_complete_native_state_exactly_after_restore --deselect tests/test_coupled_checkpoint_codec.py::test_should_restore_native_semantics_without_any_codec_model_construction
$learnerCheckpointPaths = @(
  'src/core/coupled_learner_state.py',
  'src/adapters/coupled_learners.py',
  'src/adapters/coupled_learner_checkpoint.py',
  'src/adapters/coupled_learner_checkpoint_example.py',
  'scripts/run_coupled_learner_checkpoint.py',
  'tests/test_coupled_learner_checkpoint.py'
)
.\.venv\Scripts\python.exe -m ruff check @learnerCheckpointPaths
.\.venv\Scripts\python.exe -m ruff format --check @learnerCheckpointPaths
.\.venv\Scripts\python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @learnerCheckpointPaths
```

Repeat the type command with `--platform linux` for import/type portability.
The two explicit deselections cover11 existing native cases; their unchanged
accepted evidence is reused. The new native CLI validates actual full-policy
continuation separately, so synthetic tests need no real native work. Source,
commands, results and independent review are in
`artifacts/runs/coupled-learner-checkpoint-20261015/`. See ADR-0288.

Recorded validation:197 tests passed,11 native cases explicitly deselected; all
actual native/model/process counters were zero in pytest. Six native continuation
cases passed once in0.8288 seconds with24 updates/96 energy evaluations and no
process launches. Scoped Ruff/format/Windows+Linux mypy and independent source/
actual review passed. Next compose this boundary with the existing candidate
checkpoint controller to retain managed cursor/quota/actor state through handoff.

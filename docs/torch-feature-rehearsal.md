# CPU Torch feature rehearsal

The owned feature port calls existing native PC/CPC and matched BP heads. It
trains two-class heads in this fixed recipe without an image encoder or dataset.
The general port now also accepts explicit categorical class counts; see
[categorical feature actions](categorical-feature-actions.md).

```powershell
.\.venv\Scripts\python.exe -m scripts.run_coupled_attribution --describe --scenario torch-feature-rehearsal
.\.venv\Scripts\python.exe -m scripts.run_coupled_attribution --scenario torch-feature-rehearsal
```

Describe remains Torch-lazy. Execution needs the repository's optional CPU Torch
environment. This fixed engineering demo uses one CPU/interop thread, no GPU,
60 seconds, 768 MiB absolute process RSS including Torch, and a 2 MiB JSON report.
This separate allowance leaves existing scientific ceilings unchanged. Do not
repeat the demo to select a favorable outcome.

## Port and state

`TorchFeaturePolicy` binds method, dimensions, seed, native learning/inference
rates, BP momentum and class count (default2). This recipe uses exact float64
NumPy matrices; features must remain finite in float32, and its targets must be
hard binary B×1 labels. Native
training uses owned CPU float32 features and long class indices. Prediction is
target-free softmax class-one probability in an owned float64 B×1 array.
PC/CPC energy and BP cross entropy retain different explicitly labeled units.

Snapshot format1 owns all native state. PC includes traffic; CPC retains native
format2 chemistry, lineage, ages, clocks and split RNG; BP includes actual SGD
state and name-bound gradients. Policy, geometry, CPU dtypes, finite values,
optimizer settings and native clocks are validated before publication. Fixed
no-sleep CPC ages preserve native float32 increment saturation. Training and
restore stage a complete independent head/optimizer before publishing one live
payload. Forks own their complete state. Snapshots carry no budget, consent,
receipt or deployment authority.

Why this: missing momentum or chemistry breaks continuation even when current
predictions agree.

## Replay and costs

The existing attribution runner and `StreamRehearsal` execute A2/B2/A2 wakes
with one-tick delayed labels. Replay is independently off/on across four methods.
On-mode uses the rotating recent/reservoir selector every two wakes, one row
per event, four unique rows/96 float64 array bytes, separate two-row/48-byte
partition quotas. Context/final labels never reach selection. Equal caps and
proposals do not imply equal occupancy, process memory or compute.

Each arm has six wake updates; on-mode also has a distinct permanent three-row
replay allowance. Replay advances native update/traffic and CPC chemistry/ages.
Existing native CPC clock names call all such calls wake work; app reports
separately identify actual wake/replay work. No native sleep method is called.

Eight arms plan 48 wakes, 12 replay calls, 108 training-row exposures and
64 target-free score forwards. All checkpoint A/B BCE and accuracy, B acquisition,
A loss and returning A recovery are retained regardless of direction. All final
scoring follows all training for that arm. Off/on work differs, so this is not a
causal replay estimate or evidence of a circadian advantage.

## Checks and extension

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_torch_feature_learner.py tests/test_torch_feature_rehearsal_pilot.py
.\.venv\Scripts\python.exe -m ruff check src/adapters/torch_feature_learner.py src/adapters/torch_feature_rehearsal_pilot.py scripts/run_coupled_attribution.py tests/test_torch_feature_learner.py tests/test_torch_feature_rehearsal_pilot.py
.\.venv\Scripts\python.exe -m ruff format --check src/adapters/torch_feature_learner.py src/adapters/torch_feature_rehearsal_pilot.py scripts/run_coupled_attribution.py tests/test_torch_feature_learner.py tests/test_torch_feature_rehearsal_pilot.py
.\.venv\Scripts\python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/torch_feature_learner.py src/adapters/torch_feature_rehearsal_pilot.py scripts/run_coupled_attribution.py tests/test_torch_feature_learner.py tests/test_torch_feature_rehearsal_pilot.py
```

The stage's bounded worker is the exact recorded serial CPU test invocation;
ordinary pytest alone does not impose that worker's limits. Linux typing uses
the same command with `--platform linux`.

Component tests compare resumed native head/optimizer/local RNG and copied
buffer rotation with uninterrupted continuation. The separate
[owner-local stream continuation](attribution-stream-continuation.md) now
reconstructs complete candidate/inbox/buffer state at two boundaries using the
same runtime and original budgets/receipts. The helper remains single-use;
no rebinding or allowance renewal occurs. Complete binary-action
[demonstration replay](trajectory-rehearsal.md) is now accepted through the same
ports, including transition metadata and whole-episode charges. Encoder changes,
managed and durable/process recovery remain separate unfinished requirements.
Extend existing ports/runner for demonstrated requirements; do not add another
orchestration framework. See [ADR-0313](adr/ADR-0313-own-torch-feature-heads-for-shared-rehearsal.md).

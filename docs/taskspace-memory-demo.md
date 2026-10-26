# Portable taskspace memory replay

This CPU example reconstructs a bounded nearest-memory learner from released
TRAIN packets and checks all70 recorded commands. It needs NumPy and the files
in this repository, without a simulator, model weights or research archives.

## Setup and run

Use Python3.11+ and a virtual environment. The minimal dependency list preserves
the project's existing NumPy dependency; the normal requirements.txt also works.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-memory-demo.txt
.venv/Scripts/python.exe -m scripts.run_taskspace_memory_demo --output replay.json
```

On Linux use `.venv/bin/python`. The output path must be fresh. Omitting
`--output` prints the JSON report; `--bundle PATH` selects a bounded compatible
TRAIN bundle. The default bundle is resolved relative to the source, not the
current working directory. Direct file execution also works from another cwd.

## Build and install a CPU wheel

The alpha distribution uses the existing NumPy range as its sole runtime
dependency. Optional simulator/research SDKs are prepared separately. It includes
the existing Python modules; this does not qualify every optional research CLI.
Use a build environment with setuptools>=77 installed, then run from the source:

```powershell
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
# In a separate environment, install NumPy and this wheel:
python -m pip install dist/circadian_predictive_coding-0.1.0.dev0-py3-none-any.whl
python -m scripts.run_taskspace_memory_demo --output replay.json
# Equivalent registered entrypoint:
circadian-memory-demo --output another-replay.json
```

For an offline installation, supply compatible cached wheels and add
`--no-index --find-links WHEEL_DIRECTORY` to the install command. The validation
protocol uses an existing standalone pip tool because the local Ubuntu Python
lacks ensurepip; the fresh runtime inherits neither system sites nor old .pth
files. No research archive or SDK is needed for the default demo.

The original `examples/taskspace_memory/episodes.json` is mapped at build time to
the private installed `src.infra.taskspace_demo_data` resource package. There is
one source dataset. Installed defaults use `importlib.resources.as_file` for the
loader's lifetime; source checkout/direct-file usage still finds the original
example. Explicit `--bundle PATH` selects exactly that path, including refusal
when it is absent or invalid. No working-directory search or silent data fallback.
Keep license/notice/provenance with the data; no simulator assets are packaged.

## Optional installed MiniGrid game control

The same local wheel offers the existing conventional FIFO36 game runner. Install
its explicit optional extra from the wheel directory, then inspect the protocol:

```console
python -m pip install "./circadian_predictive_coding-0.1.0.dev0-py3-none-any.whl[minigrid]"
python -m scripts.run_coupled_attribution --scenario minigrid-action-memory --invariant-fifo-memory --describe
```

Omit `--describe` only to run the fixed128-update/1152-TRAIN-row case and its
known-game readouts. Describe needs no optional game/model imports and performs
no collection, learning or environment work. The base install stays NumPy-only;
the extra preserves existing MiniGrid/Gymnasium/pygame-ce and transitive pins.
Fresh installed gameplay and clean dependency resolution are separate checks.
Known-game FIFO success does not establish held-out generalization or CPC benefit.
See docs/minigrid-cpu-install.md and THIRD_PARTY_NOTICES.md; no simulator assets
are bundled, and installed MiniGrid metadata/license-file disagreement remains.

## Meaning of the report

`complete` means the stream and every replay check completed. It does not mean
new simulator success. `recorded_outcomes` describes the saved sparse rewards
and endings and is checked against them. The default saved outcomes are Pick19
success22controls and Pick17/Pick20/Push17 success15/15/18. The failed198-row
Pick19 baseline remains in provenance and in the original research evidence.

The stream appends all220 original training rows in seven packets under one
six-update budget. Source and actor stay at A15; the candidate receives delayed
labels at ticks2/5/8/11/14/17. At the final detached checkpoint, four batches
query the saved trajectory inputs. Every selected index and native float32
command must match. Replay data never enters the training packets. B50 desired
corrections keep their original student inputs/history.

Reconstruction costs one nearest initializer/seven appends220row exposures.
Replay costs four predictions/70queries15400distance candidates184800feature
visits. Numeric final bank is28160bytes per owner; validation, copies, Python,
JSON and RSS are additional. No fitted statistics, neural training, teacher,
probe, final evaluation, promotion or simulator controls occur. This example
does not compare algorithms or establish generalization/circadian benefit.

## Files and checks

```text
examples/taskspace_memory/episodes.json       generated numeric TRAIN data
examples/taskspace_memory/README.md           origin, terms and limitations
requirements-memory-demo.txt                 minimal existing dependency
src/adapters/taskspace_memory_demo.py          admission and existing-stream composition
scripts/run_taskspace_memory_demo.py           CLI and exclusive output
tests/test_taskspace_memory_demo.py            admission, replay and failure behavior
```

```powershell
python -m pytest -q tests/test_taskspace_memory_demo.py
python -m ruff check src/adapters/taskspace_memory_demo.py scripts/run_taskspace_memory_demo.py tests/test_taskspace_memory_demo.py
python -m ruff format --check src/adapters/taskspace_memory_demo.py scripts/run_taskspace_memory_demo.py tests/test_taskspace_memory_demo.py
python -m mypy --follow-imports=silent src/adapters/taskspace_memory_demo.py scripts/run_taskspace_memory_demo.py tests/test_taskspace_memory_demo.py
```

Why this: the accepted useful control previously depended on ignored research
history and one machine's producer environment. A small data bundle and thin
composition make its recorded behavior inspectable and runnable. Extend through
the existing learner/stream rather than creating another orchestration layer.

## Optional native fresh run

`--native` reconstructs the same seven TRAIN groups and runs fresh Pick19,
Pick17, Pick20, Push17 scenes at the final detached checkpoint. It composes
the existing `run_sequence`, `run_frozen_rollout` and `roll_push`; saved replay
scores never select, update or promote the learner. Explicit `--output` is
required. Main, `.failure.json` and three `-pickSEED.json` destinations must all
be fresh. Each scene has the original50-control horizon; a complete failure
score is valid. Execution failures stop later work and retain paid partial data.
The main JSON is saved incrementally; corrupted prior evidence is refused and
trusted returned bytes go to the exclusive failure file when necessary.

Use a separately prepared Linux/WSL optional simulator environment, with pinned
ManiSkill3.0.1, SAPIEN3.0.3 and the compatible Python3.12/NumPy1.26.4 profile.
See [existing CPU simulator setup and limitations](maniskill-continuous.md) and
[compatible runtime details](maniskill-expert.md). Windows native execution is
not qualified because of the preserved OpenMP conflict. No SDK dependency or
hardware robot is enabled by the default command. This entrypoint does not
install dependencies or download assets.

```bash
# From the repository root, using your prepared simulator Python:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
python -m scripts.run_taskspace_memory_demo --native --output fresh-native.json
```

The shown ICD is specific to the qualified WSL/Mesa setup and must exist.
Before running, budget four scene constructions/resets, at most200controls/
1000known control physics steps and44000nearest candidates528000feature visits
for the bundled220-row bank. Simulator initialization/reset work is additional.
No neural training/teacher/probe/final or model random initialization occurs.
The report separates `native_fresh_rollout` from default `offline_recorded_replay`.
Re-running a known TRAIN reset does not establish unseen-task generalization.

The numeric bundle is provided under the repository MIT notice. Simulator code
and bundled Panda/table assets have their own terms, documented in the linked
setup guide; no assets or weights are redistributed here. Clean installation,
portable optional GPU mode, complete comparison packaging and release acceptance
remain open under R4.8. Existing native evidence remains preserved.

New source: `scripts/taskspace_memory_native.py`; focused boundary checks:

```powershell
python -m pytest -q tests/test_taskspace_memory_demo.py tests/test_taskspace_memory_native.py
python -m ruff check scripts/taskspace_memory_native.py scripts/run_taskspace_memory_demo.py tests/test_taskspace_memory_native.py
python -m ruff format --check scripts/taskspace_memory_native.py scripts/run_taskspace_memory_demo.py tests/test_taskspace_memory_native.py
python -m mypy --follow-imports=silent --platform win32 scripts/taskspace_memory_native.py scripts/run_taskspace_memory_demo.py tests/test_taskspace_memory_native.py
python -m mypy --follow-imports=silent --platform linux scripts/taskspace_memory_native.py scripts/run_taskspace_memory_demo.py tests/test_taskspace_memory_native.py
```

### Executed native qualification

The final 61 focused tests pass on Windows and Linux. One repaired public WSL
invocation completed all four known TRAIN scenes: Pick19/17/20 and Push17 in
22/15/15/18 controls, 70 total/350 known physics steps. The first wrapper invocation
failed on tuple-versus-JSON-list equality after a successful 22-control Pick19;
its result/cost remains preserved. A prospectively reviewed boundary repair and
regression preceded requalification. Cumulative92controls460knownphysics and
440TRAINrow exposures are charged across both invocations, with no seed/data/
metric changes. [Commands, costs, failure and review](../artifacts/runs/native-memory-demo-20261021/RESULTS.md).
This proves the scoped fresh-run path on the existing CPU environment, not a
clean installation, GPU compatibility, broad reliability or scientific benefit.

## Executed qualification

Final34tests pass on Windows and Linux; both producer environments and an
isolated75-file source/data copy reproduce all70commands. The first real-data
run failed before learning on a guessed Pick17 context; the existing canonical
helper fixed it. Three successful stdout reports were recovered exactly after
a recorder filename collision overwrote the output paths. No learner was rerun
for recovery. See [commands, costs and limitations](../artifacts/runs/portable-memory-demo-20261021/RESULTS.md).
This qualification does not establish fresh installation or new native success.

# Memory transfer to unrepresented TRAIN contexts

The public native memory command has an explicit development mode:

```powershell
.venv/Scripts/python.exe -m scripts.run_taskspace_memory_demo --native --development-probe --output development.json
```

For the existing WSL CPU simulator environment, explicitly select its installed
software Vulkan ICD before running the command:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
```

The initial local invocation omitted this setting and failed before any control;
that attempt and all reconstruction/setup costs are preserved in the results.

Use the separately prepared optional simulator environment. The output directory
must exist and the output/failure/scene files must be fresh. This command performs
paid native work; freeze a small protocol and work limits before research use.
The default command still reconstructs the original memory and its four scenes.

Development mode rebuilds the bundled220-row TRAIN memory, then scores frozen
PushCube resets19 and20 in that order from the detached final checkpoint. The
bank contains Push17 but no Push19/20 episodes. These are existing TRAIN reset
allocations, not reserved probe/final seeds. The question is whether a policy
trained on this bank transfers to other permitted resets; success is measured
by unchanged any_success/final_success/sparse_return and full interaction costs.
Complete failures are valid results. No extra supervision, optimizer update,
probe adaptation, teacher call, promotion or final evaluation occurs.

Why this: keep the existing stream, frozen native readout and capture guarantees
while distinguishing development evidence from the accepted fixed-scene demo.
`taskspace_memory_development_v1` / `unrepresented_train_contexts` reports include
an explicit scene manifest. Previously recorded outcomes are historical provenance,
not new measurements for these resets. Inspect this report as JSON; the fixed-demo
inspector does not yet admit this separate family. No general unseen-data,
comparative, CPC benefit or broad reliability conclusion follows from two resets.

Programmatic callers use `run_native_demo(payload, output,
development_scenes=(NativeScene("push",19), NativeScene("push",20)))`. Exact frozen
descriptors permit only existing TRAIN seeds17/19/20 and tasks pick/push; one or two
distinct scenes are allowed. Admission rejects contexts already represented in
the admitted TRAIN episodes, malformed descriptors, duplicate scenes and output
collisions before model/simulator work. Descriptor values are copied into an
immutable plan. Caller data cannot redirect later work. Runtime validates prior
paid evidence before each scene and retains original returned receipts on failure.

To add another task safely, first establish its role/reset authority and frozen
runner with focused tests. Do not bypass reserved allocations or choose seeds
from favorable observed results.

Focused checks use scalar scene doubles; existing native-boundary tests include
one seven-row nearest stream:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_taskspace_memory_development.py tests/test_taskspace_memory_native.py tests/test_taskspace_memory_packaging.py
.venv/Scripts/python.exe -m ruff check scripts/taskspace_memory_native.py scripts/run_taskspace_memory_demo.py tests/test_taskspace_memory_development.py
.venv/Scripts/python.exe -m ruff format --check scripts/taskspace_memory_native.py scripts/run_taskspace_memory_demo.py tests/test_taskspace_memory_development.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --platform win32 scripts/taskspace_memory_native.py scripts/run_taskspace_memory_demo.py tests/test_taskspace_memory_development.py
```

## Local result and limits

The declared Push19/20 case completed with success on both resets (19/23controls,
210known control physics steps total). The unchanged220-row memory transferred
to these two TRAIN contexts. An initial missing-software-driver failure remains
failed and charged; cumulative reconstruction is440row exposures across two runs.
This does not establish broad generalization or circadian benefit.
[Commands, work, failures, protocol and review](../artifacts/runs/memory-development-contexts-20261021/RESULTS.md).

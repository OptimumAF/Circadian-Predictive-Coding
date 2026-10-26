# ManiSkill continuous episode port (R4.4)

## Current frozen BC option — 2026-10-19

The existing smoke now accepts an in-memory TRAIN BC learner and executes a
frozen private copy; its ordinary CLI keeps zero actions. 358 scoped tests and
one real TRAIN-context integration pass review. The fixed256 policy still
fails the task within50steps. [Usage, state/work boundaries and negative evidence](frozen-bc-robot-rollout.md).
Earlier runtime milestones below retain their original scope and next actions.

## Current WSL CPU integration — 2026-10-19

The existing port and smoke CLI now complete a real PickCube episode in isolated
WSL Ubuntu:50transitions,42state coordinates, four zero normalized actions,
final time-limit truncation,250control physics steps, all rewards0, optimizer0.
Independent review accepts compatibility and exact seams/flags/work. This is not
an expert trajectory, successful picking, BC quality or adaptation evidence.
R4.4 remains open. Earlier Windows failures and the first Linux construction
failure are preserved below/in linked artifacts.

Linux uses SAPIEN's bundled C++ Pinocchio rather than the incompatible Windows
OpenMP stack. Real Panda FK/IK and224focused checks pass. Even render_backend=none
requires a RenderMaterial device; SAPIEN autoselects missing NVIDIA Vulkan under
this WSL host. The verified process-local software ICD fixes that dependency.
Do not set KMP_DUPLICATE_LIB_OK or modify upstream/global driver configuration.

From the repository root in WSL, after installing the pinned optional runtime:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
export LP_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MESA_SHADER_CACHE_DIR="$HOME/.cache/circadian/maniskill-wsl-cpu-20261019/shader-cache"
# Existing local isolated interpreter; use a fresh output and a separately
# budgeted protocol before executing another simulator episode.
/home/avery/.cache/circadian/maniskill-wsl-cpu-20261019/env/bin/python -X utf8 -B -m scripts.maniskill_continuous_smoke --output fresh-trace.json
```

The exact86official-wheel URL/SHA256 lock is
`artifacts/runs/maniskill-wsl-cpu-20261019/resolved-wheels.txt`. Installation is
local and ignored, not a tested clean-clone delivery. `pip.pyz --python PREFIX
install --require-hashes --no-deps -r resolved-wheels.txt` records the selected
solution; see the preserved install script/receipt for complete invocation.
The system Lavapipe manifest/library must actually exist; this observation does
not establish other machines' compatibility. No hardware GPU compute/frames.

Native successor5.928403718s; observed process peak sum567164928B, final self
507162624B. Polling cannot prove full-lifetime process-tree memory; that gate
remains UNVERIFIED. Reset physics are unknown, not zero. First native FAIL and
first probe serialization FAIL are separate permanent costs, not successful runs.
[Exact commands, scoped checks, failures and independent review](../artifacts/runs/maniskill-wsl-cpu-20261019/RESULTS.md).

**Next:** Implement a bounded training-only bridge from the bundled official Panda PickCube expert and its joint-position-to-pd_ee_delta_pos conversion into the existing ContinuousEpisode port. First reconcile the converter's extra steps/clipping and fixed-orientation controller with the unchanged50-step horizon; add focused boundary tests, then freeze/review one seed17 expert qualification before any BC training. Preserve failed expert/conversion outcomes; do not silently extend the horizon or label zero actions expert.

## Historical Windows runtime finding — 2026-10-19

A separate pinned Windows Pinocchio runtime installs and runs207boundary/guard
tests, but actual Panda Jacobian execution aborts on conflicting libomp/libiomp
OpenMP runtimes. No new robot smoke was admitted. Earlier empty-log fatal and
1DOFshape failure are retained separately; successful imports do not qualify CPU
IK compatibility. Do not set KMP_DUPLICATE_LIB_OK. The observer now includes real
interpreter/descendants, with whole-lifetime memory qualification still open.
[Full evidence and scoped commands](../artifacts/runs/maniskill-cpu-ik-20261019/RESULTS.md).
Next use the installed WSL Ubuntu CPU route with bundled SAPIEN IK and the same
task/controller, under a new frozen local install/native protocol; not yet executed.

## Historical initial port status

The adapter and CLI are implemented and tested at their data/state boundary.
The first actual native smoke **failed during startup**; no robot trajectory or
BC task result has been qualified. R4.4, robot adaptation and release remain open.
The existing categorical adapters and stream/runtime/learner APIs are unchanged.

```text
src/infra/maniskill_continuous.py     Optional real task factory and bounded port
scripts/maniskill_continuous_smoke.py One fixed engineering smoke, no learner
tests/test_maniskill_continuous.py    Explicit fake-boundary and CPU tensor tests
```

## Exact boundary

Use official ManiSkill3.0.1 `PickCube-v1`, Panda, `pd_ee_delta_pos`, flattened
`state`, sparse reward, one CPU environment, no renderer, official50-step time
limit. Why this: the roadmap permits reach/push/pick and PickCube is an actual
registered arm task; no arm `Reach-v1` task was assumed. This is a real optional
simulator integration, not a replacement dynamics model.

The four action coordinates are normalized root-frame delta x/y/z and normalized
gripper target, each[-1,1]. They are **not metre commands**. The pinned Panda
configuration maps the first three to±.1m from current end-effector position,
holds orientation, and maps the last to mimicked finger qpos[-.01,.04]m. The
negative finger bound is the official contact-force mechanism, not a new limit.
See [Panda source](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/agents/robots/panda/panda.py)
and [controller source](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/agents/controllers/pd_ee_pose.py).

Original native tuples are validated before float32 quantization. Records store
the **actual quantized command** sent to the controller. No clipping or rescaling
is introduced by this port. Full flattened state coordinates retain their native
order/scale; dicts, images, multiple environments, changing shapes and nonfinite
values are refused. Exact CPU Torch tensors or NumPyfloat32/64 rows convert to
owned native tuples. Sparse Boolean reward becomes exact numeric0/1; termination
and truncation remain independent Boolean flags, including bothtrue.

```python
from src.infra.maniskill_continuous import ManiSkillPickCubeEpisode

with ManiSkillPickCubeEpisode("rollout-17", "pick-cube", seed=17) as port:
    while True:
        transition = port.step((0.0, 0.0, 0.0, 0.0))
        if transition.terminated or transition.truncated:
            episode = port.episode()
            break
# A released, expert-labelled training episode can use the existing
# NumpyContinuousBCLearner.train_episode API with this identical action spec.
# A zero-action rollout alone is not an expert demonstration or useful BC data.
```

The official simulator constructor performs its own seed2022 reconfiguration
reset; the port then explicitly resets seed17. No autoreset/retry/reset API is
exposed. Storage stops at50 transitions. A missing terminal/time-limit flag at50
poisons the port rather than fabricating completion. A native exception or bad
returned output permanently refuses more steps and episode export. Invalid
caller actions are refused before spending a native step and can be corrected.
Complete records can still be read after close; incomplete records cannot export.

`work` records attempted calls, successful native returns (including malformed
returned output), known controller physics for successful returns, and the upper
bound of physics possibly spent in failed calls. Construction/reset physics are
**unmeasured**, never zero. Factory/reset failure can occur before a port object
is returned; the CLI preserves this with creationattempt1/worknull. These are
engineering counters, not runtime budget grants. Roles train/probe_train/probe_test
are preserved; final_test creation is refused pending a separate final protocol.
The existing training extraction/BC adapter rejects probe_test. No evaluation,
teacher, training, promotion or probe-selection authority is added here.

## Optional runtime and licensing

An isolated Python3.11 environment installed pinned ManiSkill3.0.1/SAPIEN3.0.3/
Gymnasium1.2.3 from official wheels, reusing read-only CPU Torch2.14.0+cpu and
NumPy2.4.6 from the existing3.11 snapshot. Main/global environments were not
modified. Exact66wheel URLs/hashes and package receipt are in the stage below;
the environment itself is an ignored local artifact, not a clean-clone deliverable.
Relevant installed simulator files match reviewed v3.0.1 files byte-for-byte.

The first Windows CPU attempt warns that Pinocchio is not installed and raises
`TypeError: 'NoneType' object is not callable`. Installed SAPIEN sets its optional
PinocchioModel toNone on Windows when that import fails; ManiSkill's CPU IK then
calls it. This is a source-supported explanation, not a captured traceback of
that first call. The final adapter now checks callable PinocchioModel before
scene creation and gives an actionable error; the CLI includes future tracebacks.
No native retry was made to qualify those changes.

The [official Pinocchio repository](https://github.com/stack-of-tasks/pinocchio)
recommends Conda and currently describes pip `pin` as Linux-only. Do not install
an unrelated similarly named package or silently switch controller/task/platform.
Next establish a compatible isolated Windows Pinocchio runtime (or explicitly
justify a supported Linux execution environment), then freeze a new bounded
smoke after prerequisite checks/review. No new native allowance is implied here.

ManiSkill code is Apache2.0; its published assets are CC BY-NC4.0, separately from
code. This local research attempt redistributes no asset/demonstration dataset.
Bundled Panda/table assets are used by the intended task; no external dataset or
automatic asset download is admitted. [Published package license statements](https://pypi.org/project/mani_skill/3.0.1/).

## Verification and limitations

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_maniskill_continuous.py tests/test_continuous_demonstration.py
.venv/Scripts/python.exe -m ruff check src/infra/maniskill_continuous.py tests/test_maniskill_continuous.py scripts/maniskill_continuous_smoke.py
.venv/Scripts/python.exe -m ruff format --check src/infra/maniskill_continuous.py tests/test_maniskill_continuous.py scripts/maniskill_continuous_smoke.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.11 --platform win32 src/infra/maniskill_continuous.py scripts/maniskill_continuous_smoke.py
# Same source check with --platform linux.
# Only after simulator prerequisite/protocol gates, using its isolated Python:
# python -X utf8 -B -m scripts.maniskill_continuous_smoke --output new-trace.json
```

Final200focused cases pass:90adapter +110unchanged continuous records. Tests use
an explicitly named fake boundary plus real CPU tensor conversions, with no
simulator dynamics or learner updates. Source/CLI review and scoped static checks
are separate from the failed native case. Initial30s type-check timeout is
preserved; putting the optional tensor import behind import_module removed the
unnecessary static traversal of Torch internals. No numerical behavior changed.

The native supervisor's5,599,232B reading covers the Windows launcher only;
**full native RSS gate is UNVERIFIED**. Future native admission must observe the
actual interpreter process and descendants. No native-success, strong BC,
retention/recovery, CPC advantage, full-suite or repository-readiness claim.

[Exact protocol, commands, failures and review](../artifacts/runs/maniskill-continuous-port-20261019/RESULTS.md).
Extend by satisfying this actual CPU IK dependency, then collect a small frozen
expert training protocol into these same ContinuousEpisode records and reuse the
existing NativeLearner/stream. Do not add another orchestration framework.

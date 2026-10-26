# Native memory demo: observed setup and remaining gates

The ordinary CPU replay needs only NumPy; see [installation](cpu-memory-package.md).
Native execution additionally needs a separately prepared simulator environment.
Read [component and data notices](../THIRD_PARTY_NOTICES.md) before using or
redistributing optional dependencies/assets. No SDK or asset installer runs from
the default demo.

## Observed working profile

The accepted local native memory runs used WSL Ubuntu, Python 3.12.3 and this
specific compatibility profile:

| Component | Version |
| --- | --- |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| opencv-python | 4.11.0.86 |
| contourpy | 1.3.3 |
| ManiSkill / SAPIEN | 3.0.1 / 3.0.3 |
| Gymnasium / Torch | 1.2.3 / 2.14.0+cpu |

The four compatibility packages live in
`/home/avery/.cache/circadian/maniskill-expert-numpy1-20261019/env`.
A `.pth` exposes the previous `maniskill-wsl-cpu-20261019/env` SDK site. This is
an observed reuse profile, **not a validated fresh native installation recipe**.
Do not copy those machine paths into a new host and assume equivalence.

The original 86-wheel native resolution is preserved in
`artifacts/runs/maniskill-wsl-cpu-20261019/resolved-wheels.txt`; the separate
four-wheel compatibility overlay is in
`artifacts/runs/maniskill-expert-diagnosis-20261019/numpy1-wheels.txt`.
Retain the original installation receipts and both prefixes. These dated files
document the resolution used; the combined clean installation, system-library
requirements and optional GPU path remain unqualified. Upstream
[installation documentation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
is a prerequisite reference, not evidence that this profile has been reproduced
on another host.

Metadata can be inspected without importing NumPy, Torch or the simulator:

```bash
/path/to/prepared/python -B - <<'PY'
from importlib.metadata import distribution
for name in ('numpy', 'scipy', 'opencv-python', 'contourpy',
             'mani_skill', 'sapien', 'gymnasium', 'torch'):
    item = distribution(name)
    print(name, item.version, item.locate_file(''),
          item.metadata.get('License-Expression'),
          (item.metadata.get('License') or '')[:100])
PY
```

This checks metadata only. Read shipped license files too: SAPIEN 3.0.3 metadata
says MIT while its installed/versioned license says Apache-2.0. Asset terms are
separate. [Local metadata/source evidence](../artifacts/runs/memory-native-terms-20261022/local_evidence.json)
is repository-local research evidence, excluded from the distribution archives.

## Explicit, budgeted native invocation

From the repository root, with a fresh output path and a separately budgeted
four-scene run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
/path/to/prepared/python -X utf8 -B -m scripts.run_taskspace_memory_demo \
  --native --output fresh-native.json
```

The ICD file and its Lavapipe library must exist. Even `render_backend="none"`
required a SAPIEN material device on this host; omitting the process-local ICD
previously failed during construction. Do not modify global drivers or use
`KMP_DUPLICATE_LIB_OK`. The existing Windows native OpenMP abort and original
MPlib 0.1.1/NumPy 2 fatal attempt remain preserved; the memory path does not need
the MPlib planner. No clean Windows native or GPU claim follows from CPU replay.

The default native sequence is Pick19/Pick17/Pick20/Push17, each at most 50 controls:
budget four constructions/resets, at most 200 controls/1000 known control physics
steps, one 220-row bank preparation and at most 44000 distance candidates/528000
feature visits. Setup/reset physics and process overhead are additional. The
CLI preserves incomplete/failed results and refuses replaced prior evidence.
The separate `--development-probe` mode selects the frozen Push19/20 TRAIN
contexts, at most two scenes; it is not final evaluation or probe training.

Accepted results: [four known TRAIN scenes](../artifacts/runs/native-memory-demo-20261021/RESULTS.md)
and [two changed TRAIN contexts](../artifacts/runs/memory-development-contexts-20261021/RESULTS.md).
These are bounded successes for the conventional memory learner. They establish
neither neural/CPC advantage nor general reliability, clean setup, GPU readiness
or permission to redistribute upstream assets.

Why this: record the environment that actually worked and its limitations rather
than presenting an unexecuted generic upgrade/install command as reproducible.

# Installing the CPU memory demo

The alpha wheel and source archive support the existing offline memory example.
They retain the original TRAIN data and license; optional simulator SDKs are
separate. See [setup and report meaning](taskspace-memory-demo.md).
The distribution includes [component and data notices](../THIRD_PARTY_NOTICES.md).
For optional native dependencies, see the [observed pinned setup](native-memory-setup.md);
clean native installation and individual asset redistribution terms remain open.

## Build, install, run

Use Python3.11+ and a build environment with setuptools>=77 installed:

```powershell
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
python -m pip install dist/circadian_predictive_coding-0.1.0.dev0-py3-none-any.whl
python -m scripts.run_taskspace_memory_demo --output replay.json
# Registered alternative:
circadian-memory-demo --output another-replay.json
```

Install the wheel into a separate virtual environment to keep it isolated.
For offline installation, provide compatible cached NumPy/project wheels and
use `--no-index --find-links WHEEL_DIRECTORY`. The validation used a new Linux
environment, no system sites/.pth, and an external cached pip tool because that
host's Ubuntu Python lacks ensurepip. The installed runtime needs no research
archive, checkout path, weights or SDK. A fresh Windows installation of the current
source wheel has also passed the scoped checks below.

```text
pyproject.toml                              ordinary wheel/sdist metadata
MANIFEST.in                                 source-only distribution contents
examples/taskspace_memory/__init__.py        mapped resource package
examples/taskspace_memory/episodes.json      single original numeric dataset
scripts/run_taskspace_memory_demo.py         source or installed data lookup
tests/test_taskspace_memory_packaging.py     resource and selection boundaries
```

The installed resource is `src.infra.taskspace_demo_data`. Source checkout and
direct-file commands still use the original example directory. Explicit
`--bundle PATH` takes precedence; invalid/missing input is refused. Extend data
through the existing bounded schema rather than adding a new loader.

## Executed evidence and limits

The source archive rebuilt a wheel with identical member contents. A fresh
Python3.12/NumPy1.26.4 installation reproduced all70recorded commands under-I
from outside the checkout. Every observed project/NumPy origin and data resource
was inside that environment; SDK imports0. Independent source/software/actual
review accepted this CPU child.

Linux75tests pass. Windows uses61verified unchanged tests plus14repaired resource
tests; the initial combined Windows run remains failed on an oversized automatic
pytest ID/log. Short explicit IDs repaired it without repeating model work.
[Exact commands, costs, failures, artifacts and review](../artifacts/runs/cpu-memory-package-20261021/RESULTS.md).

This is offline replay, not new simulator success, generalization or CPC benefit.
Clean committed-clone delivery, optional GPU mode, complete comparison packaging,
upstream terms and full release readiness remain separate open requirements.

## Fresh Windows validation

A new Windows Python3.14.7 environment installed the current project wheel and
cached NumPy2.4.6 without network access or other installed packages. The source
archive built a wheel from fresh extracted source, including the current memory
inspector and development-mode CLI. Under `-I` from outside the checkout, one CPU
run reproduced all70 selectors/native32 commands and the installed inspector
exported its exact source cells without further model work. Both public CLIs,
all170 project/NumPy module origins, dataset and template resolved inside the new
environment. Dependency check and registered console `--help` passed. No SDK/native
work or browser interaction was performed in this Windows case.

The local MSIX app redirected the named AppData prefix into its LocalCache path.
Strict resolved-path equality and `samefile` bind both names to the same fresh
prefix/work directory; the check does not accept an arbitrary prefix mismatch.
The runtime needs only NumPy and the project; pip was an external installation
tool, and no `.pth` or system/user site packages were admitted.

Current Windows wheel SHA256:
`3b8cde3290b805612a0f3e8c207f46f63259bf85bb14534688925e094e4ddb75`.
It is a distinct source snapshot from the earlier Linux wheel with the same alpha
version. [Exact commands, contents, costs and review](../artifacts/runs/windows-memory-package-20261022/RESULTS.md).
This closes the scoped fresh Windows CPU gate. Browser interaction, optional GPU,
upstream terms, complete comparison delivery and clean committed-clone/release
readiness remain open. Extend current report families through their existing
bounded app projectors; package templates/data through the existing resource map.

## Focused development checks

```powershell
python -m pytest -q tests/test_taskspace_memory_packaging.py
python -m ruff check scripts/run_taskspace_memory_demo.py examples/taskspace_memory/__init__.py tests/test_taskspace_memory_packaging.py
python -m ruff format --check scripts/run_taskspace_memory_demo.py examples/taskspace_memory/__init__.py tests/test_taskspace_memory_packaging.py
python -m mypy --follow-imports=silent --platform win32 scripts/run_taskspace_memory_demo.py examples/taskspace_memory/__init__.py tests/test_taskspace_memory_packaging.py
python -m mypy --follow-imports=silent --platform linux scripts/run_taskspace_memory_demo.py examples/taskspace_memory/__init__.py tests/test_taskspace_memory_packaging.py
```

The wheel also includes the [saved four-family comparison](continuous-comparison-inspector.md).
Use the existing inspector CLI with `--example-comparison`; this adds no learner
execution or SDK requirement. The original CPU memory default remains unchanged.

The [portable comparison input](portable-continuous-comparison.md) validates exact
bundled saved controls without training; fresh execution keeps its original
producer checks and needs a separately budgeted protocol.

# Installed CLM inspector package boundary

[Issue155](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/155),
R4.8-CLM.P. One wheel/source archive and one fresh cache-only installed environment
pass the scoped package/export checks. Both literal complete and missing-native
reviews export outside the checkout, with numerical/model imports blocked. This
does not establish new learning, interactive UI acceptance or repository release
readiness. [Issue154's manual UI criterion](clm-probe-inspector-20261025.md) remains
open after its browser policy rejection; no workaround was attempted here.

## Repair and inspected scope

The new CLM app/dispatcher/CLI and existing HTML template were already configured
as package modules/resources. Static MANIFEST inspection showed the new guide
missing from source-distribution docs. Before any build, scope was amended to add
one `include docs/clm-probe-inspector.md` line, preserving every prior rule. The
guide adds an explicitly illustrative missing-native JSON example usable without
the research tree. It is not experiment evidence. No learner, optimizer, loss,
metric, model, CLI flag, dependency or renderer change.

The package retains its declared NumPy dependency for other features. Exporting
these review tables does not import it. Ordinary setuptools/pip helpers adapt the
existing47build/install procedure to current package-dir mappings; no new build
framework or archive of scientific evidence is introduced.

## Executed package and install checks

One ordinary isolated staged-source build, no-index/no-deps/no-build-isolation:

| Build observation | Value |
| --- | ---: |
| Source bytes staged |16,836,928 |
| Wheel |3,755,847B /1,039members |
| Source archive |3,099,459B /1,081members |
| Wheel command seconds |22.875918500 |
| Source archive command seconds |8.069429400 |
| Whole controller seconds, including staging/inspection |39.487769000 |
| Sampled controller peak RSS |28,008,448B |
| Stage after original build |40,585,440B |

Affected CLM module/dispatcher/file adapter/CLI/template bytes match the current
checkout in the wheel and source archive. All configured wheel package-data and
both licenses match; the source archive also contains the new guide and licenses
exactly. No research/artifacts/tests/pycache are in the wheel. Source/build output
stays in the owned stage; root build directories/user changes are untouched.

Wheel SHA256:
`7eefc1ae3be40ef4bcfcd01c6f5c3172cc7c6dc3862e8c90ff8c37a695950df8`.
Source archive SHA256:
`e79c7332cfe0c8e6a866bee077a4ee3a164ff6071c4709b3d49f3b60f901fcf3`.
Files are under artifacts/runs/clm-inspector-package-20261025/{wheel,sdist}/.

The fresh LinuxPython3.12 environment is
`/home/avery/.cache/circadian/clm-inspector-package-20261025/env`, with unrelated
working directory `.../work`. It has no system site packages, editable/.pth files,
checkout/PYTHONPATH imports or research/source/example trees. Installation uses
only the existing cached pip.pyz, NumPy wheel and new candidate wheel, with
`--no-index --no-deps --no-cache-dir`. `pip check` exits0. Exact installed
distributions are `circadian-predictive-coding==0.1.0.dev0` and `numpy==1.26.4`.

| Installed observation | Value |
| --- | ---: |
| Cached install seconds |6.691747224 |
| pip check seconds |1.386739665 |
| Verification parent seconds |0.089374197 |
| Whole environment/install/check seconds |8.457175319 |
| Verification body seconds |0.027489933 |
| Verifier peak RSS |36,585,472B |
| Prefix bytes |132,075,718 |
| Working files bytes |133,755 |
| Product module origins checked inside installed prefix |14 |

Under `-I`, the public CLI runs on two **literal metadata fixtures**, not accepted
scientific outputs. Complete negative-case HTML93,879B and missing-native
HTML26,304B preserve exact source JSON, projected source cells, signed margins,
unknown fields and rendered bytes. Both existing-output attempts refuse with
argparse exit2 and leave bytes untouched; malformed final_test case role is
refused. The selected CLI/module/template and every loaded src/scripts module
resolve inside the installed prefix. A numerical/native import guard records
zero denied imports; NumPy/Torch/Transformers remain unloaded. No model creation,
training, prediction, backward, encoding, random initialization, effects, final
selection/replay or scientific input/review parsing occurs.

## Failures, warnings and evidence limits

An initial multi-file doc patch failed context validation without writes; an
exact-context retry succeeded before building. The sole build and install both
pass, with no retry/budget revision. Initial mypy finds reused archive/output
variable names with incompatible types and optional module paths (3Windows/
8Linux errors). Executed helpers are retained as build_executed.py and
verify_executed.py. Variable renames/identity typing.cast fix typing only; normalized
ASTs are exact before/after, no control/numerical change. No build/install was
repeated for those type-only changes. Final scoped Ruff/format/two Windows-target
and three Linux-target helper type checks pass, measured2.818104s including failed
statics of90s plus subsecond document/archive checks.

Setuptools retains warnings about deliberately excluded root README/bytecode
patterns and no conventional root README in the source context; project readme
is configured as docs/taskspace-memory-demo.md. The verification's import-origin
inspection before runpy produces a runpy warning; its expected no-overwrite
argparse errors are also retained. These are not model failures or hidden failed
installs. Source archive is not claimed to be a complete research archive.

Build uses existing backend dependencies (`--no-build-isolation`), not a hermetic
toolchain. Controller samples its own RSS via the existing stdlib-only helper;
build/install child peaks and backend import traces were not captured. Backend
metadata uses a static version and needs no source-product execution. Installed
source isolation and export import absence are explicitly observed; they should
not be generalized to every package entry point or full repository readiness.

Original limits: build120s/onewheel+onesdist/stage192MiB/log1MiB each/512MiBfresh
disk/1GiBsampledcontroller; install/check120s/prefix256MiB/work16MiB/1GiBverifier;
CPU1sequential/offline. Observed values fit these bounds. Final cache/stage sizes
are recorded at closeout: stage before the small final review40,607,395B and
ordinary tool caches96,519,793B fit192MiB/128MiB. Global/current environments and old package/science
artifacts are preserved; no downloads/install of new build dependencies, remote
CI, commit/push/publication or deletion. A fresh owned test prefix remains.

No unchanged154unit/model/math suites or browser/UI checks are repeated. Actual
Linux Ruff/mypy unavailable; Linux-target typing runs onWindows. No native model
experiment, scientific benefit, independent review, clean-clone/full release or
interactive UI claim. Lead source/procedure/results review is **self-review**.
All153failed-spent and154manualUI requirements and original negatives remain.

## Reproduction and next outcome

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.venv/Scripts/python.exe -B artifacts/runs/clm-inspector-package-20261025/build_package.py
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/Avery/Documents/ChatGPT/Circadian && timeout 125s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/clm-inspector-package-20261025/install_fresh.py'
.venv/Scripts/python.exe -m ruff format --check artifacts/runs/clm-inspector-package-20261025/build_package.py artifacts/runs/clm-inspector-package-20261025/install_fresh.py artifacts/runs/clm-inspector-package-20261025/verify_installed.py
.venv/Scripts/python.exe -m ruff check artifacts/runs/clm-inspector-package-20261025/build_package.py artifacts/runs/clm-inspector-package-20261025/install_fresh.py artifacts/runs/clm-inspector-package-20261025/verify_installed.py
.venv/Scripts/python.exe -m mypy --cache-dir .mypy_cache/clm-workflow-probe-20261025-win --explicit-package-bases --follow-imports=silent artifacts/runs/clm-inspector-package-20261025/build_package.py artifacts/runs/clm-inspector-package-20261025/install_fresh.py
.venv/Scripts/python.exe -m mypy --platform linux --cache-dir .mypy_cache/clm-workflow-probe-20261025-linux --explicit-package-bases --follow-imports=silent artifacts/runs/clm-inspector-package-20261025/build_package.py artifacts/runs/clm-inspector-package-20261025/install_fresh.py artifacts/runs/clm-inspector-package-20261025/verify_installed.py
```

Exclusive build/install names are already used; commands are recorded evidence,
not authorization to overwrite or repeat. Complete command arrays/paths/times are
in build.json/install.json; source origins/cases/absence of model imports in
installed-exports.json. No unexecuted tests or plan-only completion.

The package boundary is now evidenced; another inspector/packaging sequel is
not the learning critical path. The native CLM train step already computes a
pre-clipping gradient norm at clm_head_learner.py457 and discards it, returning
only objective/loss. Current fit diagnostics expose probability margins, not
gradient flow. The partial F acquisition/interference therefore lacks a direct
training-step observation; no zero-gradient/clipping cause is inferred.

**One next implementation action:** add an opt-in bounded observation of that
already-computed CLM pre-clipping norm through the existing training path,
preserving default TrainingDiagnostic/loss, optimizer/schedule, model/RNG/source/
probe/budget/consent isolation and exact updates. Use literal head fixtures first;
no extra forward/backward or pretrained rerun, no controller/tuning from the
observation. This advances R2.1 learner diagnosis using actual source evidence.
Full reliable learning/retention/return/R5.1/R5.3/G5/R7/release remain open.

# R4.8-CTX.N.I — installed native and weighted BP support

[Issue279](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/279).
Full goal and parents18/21/23/33 OPEN. Historical plans/log remain preserved per
AGENTS; GitHub holds live scope/handoff. This is installed project compatibility,
not fresh Torch SDK installation, scientific benefit or repository-wide release.

## Delivered

`pyproject.toml` adds optional `native = ["torch>=2.2,<3.0"]`, reusing the existing
requirements-resnet constraint. Base NumPy and existing extras/entrypoints remain
unchanged. `src/shared/torch_runtime.py` now explains the installed native extra
when Torch is absent, retaining the separate vision requirement. No numerical
algorithm/state/default changes. `docs/native-ticket-setup.md` is shipped both
by MANIFEST and explicitly as wheel data at
`<prefix>/share/doc/circadian-predictive-coding/native-ticket-setup.md`.

Three zero-model tests/platform PASS (Windows initial and final; Linux final),
covering optional metadata/base compatibility/shipped guide and lazy available/
missing runtime behavior. Ruff/format/default+Linux-target mypy PASS for the
runtime and new test; whitespace PASS with inherited CRLF advisories. No native
source suites were repeated: unchanged275–278math/roles/causal/state tests and
negative scientific results are reused, not recertified or altered.

## Fresh artifact and base install

New physical owner `D:/Dev/Projects/Circadian-Native-Installed-20261029/` contains
`build-source/`, `dist/`, `env/`, `work/`. Older owners/wheels/environments and
inherited dirty270–278 source are preserved; no commit/push/global change.
Evidence: `artifacts/runs/native-ticket-installed-20261029/`.

One explicit export1097files/17,652,705B in4.8449307s, capped1600files/32MiB.
Offline Windows setuptools build14.6490511s/45,879,296B sampledRSS, zero model
module imports. [Wheel](D:/Dev/Projects/Circadian-Native-Installed-20261029/dist/circadian_predictive_coding-0.1.0.dev0-py3-none-any.whl)
3,955,918B, SHA256
`be53902ba542b43e891ca8d61bb1fc82e14dcf4e378455fef64fc36c46eef6ef`.
1080members, all1072current code/resource inputs byte-exact; guide bytes exact;
tests/artifacts excluded. Initial source-only wheel check covered1053inputs;
`resource-check.json` also verifies the19mapped example resources, not a rebuild.

Fresh LinuxPython3.12.3 env initially has no system packages/PYTHONPATH/Torch.
Existing held pip.pyz and NumPy1.26.4 wheel installed offline with
`--no-index --no-deps --no-compile --no-cache-dir`, no resolver/download/global
install. Install13.614610909s/55,554,048B, pip check1.095384396s/36,564,992B EXIT0.
Twelve installed checks PASS2.045760885s/50,388,992B: imports/help/new flags,
unsupported CLI/API combinations, exclusive output, actionable missing-Torch
error, extra metadata and installed guide. Teacher/constructors/episodes/Torch
imports all zero; every project origin comes from the new env. Environment/
install/base checks together16.839165206s, within120s/512MiB sampled limits.

Then only this new env receives a plain dependency path file pointing at the
**read-only existing** SDK site:
`/home/avery/.cache/circadian/maniskill-wsl-cpu-20261019/env/lib/python3.12/site-packages`.
This reuses Torch2.14.0+cpu and its installed dependencies. No provider `.pth`
files are executed by this plain entry. New-env NumPy/project paths precede it.
No old SDK is copied, upgraded or altered. Base pip check preceded this addition;
no combined-SDK pip-check/fresh native dependency resolution is claimed. This
machine-specific reuse is not a portable fresh SDK recipe or range-wide support.

## Tiny installed engineering fixture and failure

The fixture uses existing AttributionStream and counted native factory for
two delayed ordinary TRAIN packets of6rows. All-one class weights, same seed17/
D9/H3/K6/lr.03/momentum.9; compare the resulting full candidate to two ordinary
BP steps. No static diagnostic teacher, task episode, final/probe/replay access,
publication, old scientific raw read or12update cohort. The sampler/profile
starts before optional numerical imports; counts physical optimizer work.

First attempt FAILED after2weighted SGD/12rows: the fixture supplied detached
public `stream.applied_updates` copies to a guard requiring the original receipt
object. Production runner already uses the correct inbox owner. Preserved
`native-first-work.json`:5.194216873s, boundary327,933,952B/sample327,716,864B,
7component inits315nominal slots,2forwards16snapshots1404logical elements.
No ordinary reference steps occurred; these paid updates are not erased.

Changed only fixture to the original inbox receipts; did not weaken production
identity validation. Independent source/procedure review used2readonly calls,
found no packaging/source blocker, and correctly noted meter exit-only wall/RSS
checks. Added an ordinary parent watchdog from the existing install subprocess
pattern before the second call. It enforces original cumulative90s/1GiB limits.
Review exception ENDED,0imports/models/tests/SDK/artifact reads/writes. Lead
source/procedure/results self-review complete; this is not independent numerical
replication or scientific acceptance.

Second/final engineering invocation PASS:4physical SGD24rows (2weighted+2ordinary),
10component inits450nominal initializer slots,4forwards25snapshots2430logical
elements. Candidate full state equals ordinary all-one continuation; source,
actor and Torch RNG unchanged;2matching causal records and zero score forwards.
All96loaded project modules originate inside the new installed env; actual Torch
origin is the explicit prior SDK. Body5.960103771s; parent watchdog6.508904979s,
parent sampled peak329,347,072B/no stop. The two shell-observed calls total
17.4877215s including WSL launch; child meters total11.154320644s.

Original2call budget held: **6SGD/backwards36TRAINrows**,17component inits765nominal
slots,6forwards36forward rows,41snapshots3834logical tensor elements;4weighted
calls24rows plus2ordinary reference calls12rows. Limits8steps48rows20inits900slots/
30forwards200snapshots100000elements90s1GiB unchanged, no renewal.0PC latent/
NumPy fits/physics/GPU/task episodes/probe/final/replay/scientific work. Nominal
slots/copies are not FLOPs/exhaustive allocations; RSS is sampled, not OS-hard peak.

Postflight owner120,618,497B/evidence357,541B before documentation, below512MiB.
The wheel remains the built inputs; later README/ARCH/report edits are usage
documentation only, not a claim those subsequent files are inside that wheel.

## Exact commands and remaining gates

All Windows TEMP/TMP=`D:/Dev/_Temp`; Linux TEMP/TMP/TMPDIR=`/mnt/d/Dev/_Temp`.
Source zero-model check:

```text
.venv/Scripts/python.exe -B -m pytest -q tests/test_native_install_metadata.py --basetemp D:/Dev/_Temp/native-install-meta-win-final
.venv/Scripts/python.exe -B -m ruff check src/shared/torch_runtime.py tests/test_native_install_metadata.py
.venv/Scripts/python.exe -B -m ruff format --check src/shared/torch_runtime.py tests/test_native_install_metadata.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent src/shared/torch_runtime.py tests/test_native_install_metadata.py
```

Mypy also `--platform linux`. Linux pytest used existing source-environment
interpreter with `-B -m pytest -q tests/test_native_install_metadata.py --capture=sys
--basetemp /mnt/d/Dev/_Temp/native-install-meta-linux` from physical repo.

Build from new owner/build-source:

```text
D:/Dev/Repositories/Circadian/.venv/Scripts/python.exe -B D:/Dev/Repositories/Circadian/artifacts/runs/native-ticket-installed-20261029/build.py
```

Linux commands from new owner/work (stage absolute D: WSL path):

```text
/usr/bin/python3.12 -B <stage>/install.py
<owner>/env/bin/python -I -B <stage>/native_meter.py native <stage>/native-first-work.json
/usr/bin/python3.12 -B <stage>/native_watchdog.py
```

First native command EXIT1 is retained. Watchdog second invocation EXIT0; its JSON
contains complete actual argv/cwd/environment-independent limits. Install JSON
contains all complete pip/check/gate commands. Compile checks on finite observers
ran without numerical imports. Some initial orientation path/glob errors were
replaced with direct existing-path reads; no data or experimental effect.

Skipped broad suite/CI/clean Git clone/sdist/Windows-installed execution/new SDK
resolution/GPU/simulators/family comparison/scientific replication/release. The
full Torch range, fresh native dependency setup and useful D9 neural learning
remain unqualified.278negative ready/task results remain unchanged/SPENT/rawclosed.

## One next learner-facing action

Do not start another packaging or capture sequel. Native H3 ready-action fit is
still unresolved; adding more mechanisms or tuning seeds/dose is not justified.
There is already a bounded fixed-intercept one-hot least-squares reference in
`src/core/categorical_linear_fit.py`, accepting D9/K6 without new dependencies;
the current ticket qualifier has no such reference. It can distinguish whether
a simple conventional score function fits the original TRAIN representation,
without claiming that a failed least-squares fit proves nonlearnability.

Next implementation: expose that existing disposable TRAIN-only reference through
the existing context ticket qualification path, sharing the original teacher and
frozen task readout. Start role/isolation/fit-cost tests. Freeze any one-fit12row
development case and task metrics before execution; keep native/deployed/benchmark
owners untouched, preserve old controls and all negatives. No automatic study,
hyperparameter search, final/probe labels or new orchestration framework.

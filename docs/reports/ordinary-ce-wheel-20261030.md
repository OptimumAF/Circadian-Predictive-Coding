# R4.7-CE.D / issue325 — local package delivery

The combined current source built into a local pure-Python wheel, installed into
a NEW private D target, and passed actual CLI help/refusal/output/dependency
guidance checks on Windows and Linux from empty D working directories. All
project imports came from the exact wheel or installed target. Valid-mode
forwarding is explicitly scripted; no model was constructed and no training or
adaptation experiment ran. First Linux dependency-path failure is preserved.
This qualifies project delivery with existing SDK dependencies, not fresh
dependency installation, native learning, a clean clone or release readiness.

Previous turn PROGRESS:324 delivered accepted ordinary CE initialization/probe
option,69distinct checks/platform149PASS/scoped statics/independent review.
That source/math is unchanged and reused.323 remains FAILED/SPENT, adaptation/
retention/recovery UNKNOWN; no retry, renamed study or scientific budget renewal.
Full goal/#18/#21/R2.1/R4.7/scientific comparisons/release remain open.

Physical checkout D:/Dev/Repositories/Circadian, HEADd9119e80926a139a2f38460564d7eab1b6d2a2b0
plus inherited dirty/untracked package files. No production or packaging config
changes; no commit/push/download/dependency/global install, SDK/cache setting
change or user-data deletion. Existing checkout build/egg-info/ignored/frozen
outputs preserved. Standard backend removed only its own new bdist staging tree.
All project source/copies/wheel/installed target/cwds/receipts remain on D;
TEMP/TMP/TMPDIR D:/Dev/_Temp. No new library-temp-byte/whole-process-tree claim.

Existing Windows pip26.2.1/setuptools84.0.0 were verified by metadata before
work. setuptools includes build_meta and bdist_wheel, so no build/wheel frontend
or dependency installation was needed. New stage
artifacts/runs/ordinary-ce-wheel-20261030/ contains small one-off build/smoke/
target-install helpers, frozen protocol, raw logs/JSON receipts, copied configured
inputs and wheel/installed files. No new verification/orchestration framework.

Prospective325 limits: one in-process backendbuild120s/512MiB observedRSS;
source copy24MiB/full intentional stage128MiB; smokes30s/256MiB each, originally
two platform processes; helper static90s. Model/native/optimizer/randomparameter/
forward/backward/probe/final/simulator work hardZERO. Build blocks NumPy/Torch/
torchvision imports; all smokes block Torch and constructors/run_example during
CLI admission. Existing actual API is invoked only to verify missing-Torch
guidance before construction. NumPy SDK imports for CLI definitions are allowed,
not model work. No numerical tests/baselines/324fixtures repeated.

ONE backendbuild exact command from physical checkout, parent TEMP/TMP/TMPDIR D:

```text
.venv/Scripts/python.exe -I -B artifacts/runs/ordinary-ce-wheel-20261030/build_once.py
```

stdout/stderr retained in build.log. Backend works inside new stage/source,
copying configured src/scripts Python, HTML, explicit package data and literal
MANIFEST includes/license/docs. Initial package-only size1091files17,851,022B;
full configured+manifest inputs1116files17,950,693B<24MiB. Containment/symlinks
checked before copy; no customsetup.py executed. build_meta.build_wheel runs
in-process, not unmetered compiler/model subprocesses. EXIT0/complete/budgettrue,
18.0199782s/49,782,784B observedRSS, no numeric imports. Wheel
`wheel/circadian_predictive_coding-0.1.0.dev0-py3-none-any.whl`,4,027,425B;
setuptools84 metadata Root-Is-Purelib true/Tag py3-none-any. build.json retained.

Seven targeted entries byte-match the actual current checkout in contents.json:
ordinary runner, ticket_ce_probe, ticket_native_probe, torch_categorical_ce_readout,
torch_categorical_ce_initialization, torch_categorical_readout_fit, torch_decision.
No source-model reconstruction or new hashing system; straight targeted byte
comparisons. This binds combined local source, not a clean committed revision.

Initial Windows wheel ZIP import smoke passed9metadata cases/.6940092s/
54,050,816B,0modelattempts. That is weaker than project installation, so BEFORE
further work325 was prospectively amended for ONE project-only target install
30s256MiB and ONE additional Windows installed-target smoke. First success and
costs retained; source/build/stage/static limits and modelZERO unchanged.

Exact target-install command:

```text
.venv/Scripts/python.exe -I -B artifacts/runs/ordinary-ce-wheel-20261030/install_target.py
```

It invokes existing pip in-process with
`install --no-deps --no-index --no-cache-dir --no-compile --target <newDstage/installed> <exactwheel>`.
Only this project is installed into the new target; no dependency/environment
replacement, network or cache edits. install.log/install.json: EXIT0/complete/
budgettrue,1.7744766s/64,753,664B, no NumPy/Torch imports. All7installed targeted
files exactly match wheel bytes (installed-contents.json). Existing SDKs supply
dependencies; advertised native Torch>=2.2,<3.0 range is metadata, not a tested
range of Torch builds. No new global/project-venv install performed.

Smoke command structure: `<existingPython> -I -S -B <stage/smoke.py> <exactwheel>
<explicitSDKsite> <newreceipt> [<installedtarget>]`. Each workdir starts empty,
and helpers assert emptiness. -I -S omits cwd/PYTHONPATH/user/site/.pth/editable
hooks. Only exact project wheel/installedroot and explicit existing SDK site are
prepended; all loaded src/scripts module origins are checked under that project
root. Actual full module-origin lists are saved, not replaced with scalar claims.

Windows exact executable D:/Dev/Repositories/Circadian/.venv/Scripts/python.exe;
SDKsite D:/Dev/Repositories/Circadian/.venv/Lib/site-packages. Initial command
uses windows-smoke.json/no optional installedroot/cwd stage/empty-windows;
installed command uses windows-installed-smoke.json/explicit stage/installed/
cwd stage/empty-windows-installed. Both use the same four other absolute paths
and the above flags.9cases each PASS; installed1.2161821s/52,760,576B.

Linux exact command after the documented repair:

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian/artifacts/runs/ordinary-ce-wheel-20261030/empty-linux-corrected env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -I -S -B /mnt/d/Dev/Repositories/Circadian/artifacts/runs/ordinary-ce-wheel-20261030/smoke.py /mnt/d/Dev/Repositories/Circadian/artifacts/runs/ordinary-ce-wheel-20261030/wheel/circadian_predictive_coding-0.1.0.dev0-py3-none-any.whl /home/avery/.cache/circadian/maniskill-expert-numpy1-20261019/env/lib/python3.12/site-packages /mnt/d/Dev/Repositories/Circadian/artifacts/runs/ordinary-ce-wheel-20261030/linux-corrected-smoke.json /mnt/d/Dev/Repositories/Circadian/artifacts/runs/ordinary-ce-wheel-20261030/installed
```

First Linux command instead used primary runtime SDKsite
`.../open-jev-native-smoke-20261022/env/lib/python3.12/site-packages`, initial
empty-linux cwd and linux-smoke.json. Under -S it failed before cases with
ModuleNotFoundError: NumPy, .080261551s/22,466,560B,0modelwork, budgettrue but
completefalse/EXIT1. This was not silently retried: readonly find_spec under the
existing runtime showed NumPy actually comes from the linked maniskill-expert
SDK site. BEFORE repair execution325 explicitly allowed ONE extra Linux
metadata smoke with that exact observed SDK site, new empty cwd/new receipt,
same30s256MiB and unchanged other caps. No dependency installed or .pth reenabled.
Corrected Linux9cases PASS/EXIT0,1.380354577s/55,865,344B. The same installed pure
Python project tree is used across OSs; Windows-generated console wrappers are
not claimed as tested Linux entrypoints. All source imports inside target.

Each successful smoke verifies9cases: actual help; four invalid option/family/
context/choice combinations; existing paid output preserved; actual native API
missing-Torch installation guidance; CE option forwarding; default PC option
forwarding. The last2use marked SCRIPTED run_example plus the actual bounded
JSON writer, not native training. Actual API error says install
`circadian-predictive-coding[native]`; Torch imports blocked and no constructor
hook fired. Metadata has NumPy>=1.26.4,<2.5 and Torch>=2.2,<3.0 only for native.
No final/probe labels, effects or simulator work.27passing metadata cases across
3successful processes plus1preserved failed import process; not27unqualified
passes with hidden failure. All4attempts stayed within time/RSS/model caps.

Helper scoped Ruff/format/default+Linuxmypy receipts: statics-first1.7090094s,
format/lintPASS but24type errors/platform from mixed-JSON inference and dynamic
modules; explicit annotations resolved those. statics-final1.6236899s left one
optional metadata-list error; strict non-None dependency assertion fixed it.
statics-metadata-final1.7874053s all4PASS before build. Extended smoke/installer
statics-install1.7088123s all4PASS before execution. Total6.8289169s<90; initial
failures retained. No production changes or new broad test runs.

Skipped fresh Torch/NumPy dependency installation, fresh external environment,
genuine native workflow, source12/F16/return16, baseline/model/gradient/physics
work, numerical suites, full repository/clean-clone/release acceptance. The
actual wheel is a local development artifact, not published or scientifically
qualified. All negative/SPENT evidence including323unknown source/actor proof,
318temporary gap and319static overrun remains intact.

Next concrete implementation after this delivery check: add a concise reserved-
probe outcome summary using its ALREADY returned typed episodes. Current
ticket_native_probe retains all phase episodes/work but no task-level summary
separating fresh acquisition, retained old skill and conditional recovery. Reuse
existing context/probe summary functions where applicable. Keep old4/fresh2
membership and original success/metrics/directions fixed; absent/partial phases
must remain unknown, and recovery must be conditional on an observed old skill
lost during F. Report auxiliary oldTRAIN rehearsal and private work separately.
No new predictions, labels, tuning, model work or native323retry; this makes
existing evidence readable without substituting reporting for adaptation proof.
Freeze its small metadata-fixture allowance before execution. Full goal active.

Independent final review ACCEPTED scoped325 package delivery, no source blockers.
Two readonlycommands/allEXIT0/nosessions/.1553382s+.414411s=.5697492s; no project/
numericalimports/models/tests/writes/network/children. Both installedplatform
receipts have157projectmodule origins inside their exact installedtarget. Review
directly compared six advertised targets plus nativelearner across actualsource,
stagedsource/wheel/installation; seventh advertised torch_decision is supported
by saved lead byte-comparison receipts. Do not inflate that direct review scope.
Reviewer first projected absent project_root in initialZIPreceipt as foreign
origin; corrected interpretation uses that old receipt's wheel field. This was
a review-helper projection error, not a package failure. No extra read/allowance
or test rerun. Finite exception ended; self-review remains distinct.

Retain one sourcecopy/build/wheel/installedtarget and all successes/failures/logs
within128MiB; no further package-copy chain or cleanup. Final storage/closeout
receipts follow below. Full native/scientific/release requirements remain open.

Closeout receipts:325 canonical comment6529566998, issue closed/completed with
four scoped criteria checked and prospective engineering amendments retained.
Tracker18 comment6529591866 and parent21 comment6529592421 keep fullgoal open.
Final retained3347files58,408,876B<128MiB. Scoped whitespace checks clean with
LF/CRLF advisories only. No production changes or additional model/science work.

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

## Explicit private training23 probe

The existing command also accepts `--training-probe ORIGINAL23_JSON`. Supply
your original qualified18-row `probe_train` export for reset23. It is intentionally
separate from the bundled TRAIN data: the complete original SHA256, decoded
values, teacher identity, role, context and executed history must match. An
arbitrary demonstration or reset24 file is refused; this option grants no
collection or new training-data authority.

```powershell
python -m scripts.run_taskspace_memory_demo --training-probe "D:\Data\probe_train-model.json" --output private-fit.json
# Optional existing simulator: read-only original23 before and after the append.
python -m scripts.run_taskspace_memory_demo --training-probe "D:\Data\probe_train-model.json" --native --output private-native.json
```

The CPU command needs only the usual NumPy dependency. It appends the seven
bundled TRAIN packets (220rows), then uses the existing delayed private probe
for one18-row append and before/after training-fit/conflict/provenance/cost
reports. It reads no evaluation packet and returns no adapted learner. Source
and stable actor remain unchanged. Output paths and their failure/before/after
siblings must all be fresh; late output failures preserve the paid report in
the bounded failure capture and stdout where available. No automatic retry.

The optional native mode uses the original `probe_train` role and distinct
inference ID, never ordinary TRAIN allocation or teacher supervision. It is
exclusive with `--development-probe`. At most two native endpoint returns are
retained; corruption or replacement of earlier evidence refuses further work.
The default replay and ordinary native/development commands keep their behavior.

API callers can qualify the file with
`scripts.push_cube_reserved_probe_inputs.load_reserved_training_input(path)`,
then call `src.adapters.taskspace_memory_demo.run_demo(payload, training_probe=episode)`.
The typed API owns and validates its episode; the IO loader additionally binds
it to the original fixed export bytes. Borrowed endpoint observers are synchronous
and must return `None`; they cannot retain or publish the private learner.

This reports known-training reproduction, not transfer, retention, reliable
recovery or a CPC advantage. The accepted native23 result succeeded on its
training reset; [reset24 remained unsuccessful](reports/nearest-reserved-native-study-20261101.md).
[Workflow qualification and limits](reports/taskspace-training-probe-workflow-20261101.md).

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

## Explicit bounded retained memory

The same public command accepts the existing FIFO or seeded-reservoir retention
rule. This is opt-in; the default still requires exact saved selectors and native
float32 commands. Use a fresh output path in an existing directory:

```powershell
python -m scripts.run_taskspace_memory_demo --retention recent_fifo --max-prototypes 128 --output retained.json
python -m scripts.run_taskspace_memory_demo --retention seeded_reservoir --retention-seed 17 --max-prototypes 128 --output retained-reservoir.json
```

The API equivalents are `run_demo(payload, retention=ReplayRetentionPolicy(...),
max_prototypes=128)` and `run_native_demo(payload, output, retention=...,
max_prototypes=128)`. Add `--native` in the existing prepared simulator environment
to execute four fresh known TRAIN scenes; all native requirements above apply.
Capacity is1..256. Reservoir requires an explicit unsigned64-bit seed; FIFO
refuses a seed. Rule-less capacity/seed, incompatible training/development probes,
and invalid capacities refuse before reading data or constructing a learner.

CPU mode `offline_retained_training_readout` scores all four original saved TRAIN
query batches. Its command-fidelity report describes native float32 mismatches,
absolute errors, direction differences and comparison work. A computed mismatch
is a valid scored result. Retained-bank selector indices are not comparable to
original full-bank indices (`selected_indices_exact=null`). Recorded rewards and
success fields belong to the original recordings; `native_outcomes=null` makes
clear that no new simulator outcome was measured. The strict default still stops
on any selector/command mismatch.

Native mode `retained_native_training_rollout` reports actual fresh scene outcomes;
a complete failure score is valid. Source/readout full-state pins, cumulative
exposure counts, retained search-row counts, prior paid receipts/captures and
bounded failure restoration remain enforced. JSON capture comparisons preserve
tuple/list conversion while refusing changed primitive types. Private probes,
development resets, promotion and final evaluation remain separate and unavailable.

Both modes use the existing seven-group stream and full-state retained learner.
Reports separate cumulative training exposure, current stored rows, duplicate/
discarded content, numeric/encoded-ID payload and retrieval/comparison work.
Those payloads exclude Python, validation and simulator overhead. FIFO can forget
old tasks; deterministic reservoir can retain a different subset. Neither rule
guarantees learning, retention or recovery, and this workflow proves no CPC
advantage. See [bounded-memory semantics](retained-nearest-memory.md) and the
[preserved mixed native results](reports/retained-memory-storage-study-20261101.md).

New integration checks: `python -m pytest -q tests/test_retained_memory_workflow.py`.
Run Ruff and platform type checks on that module and the four changed workflow/
validator modules. Keep new modes explicit and preserve strict default/probe routes
when extending the existing stream; do not add a second orchestration framework.

The one fixed FIFO128 public qualification completed on CPU and native. CPU
readouts matched Pick19 but mismatched13/15 Pick17,13/15 Pick20 and10/18 Push17
commands. Native Pick19 succeeded22controls; the other three failed50 each.
These negatives and full costs are preserved in the
[workflow report](reports/retained-memory-workflow-20261101.md). Finite independent
source/PRE/POST review accepted this scoped integration, with stated review
limits. Clean installation, GPU, generalization, recovery and release remain open.


## Explicit isolated bounded adaptation on CPU

The existing command can now learn the original reserved23 TRAIN packet on an
isolated child and optionally return to one original old TRAIN packet. The source
and stable actor stay unchanged. Reports contain fit/interference/return scores,
receipts and costs; they export no adapted learner and measure no simulator outcome.

```powershell
python -m scripts.run_taskspace_memory_demo --adaptation-probe artifacts/runs/reserved-robot-probe-input-native-20261020/probe_train-model.json artifacts/runs/reserved-robot-probe-input-native-20261020/probe_test-model.json --retention recent_fifo --max-prototypes 128 --return-task pick17 --output adaptation.json
```

Run from the repository root in the prepared NumPy environment with a fresh output.
Both original fixed-digest exports must exist; they are local research inputs,
not arbitrary replacement tasks or final-evaluation files. The strict role/history
reader preserves original23 TRAIN and readonly24 identities. It opens neither
simulator nor final data. No SDK is needed for this CPU mode. Installed-wheel and
clean-clone qualification of this new mode remain separate from checkout checks.

`--adaptation-probe` requires explicit retention, capacity and output. Omit
`--return-task` for acquisition-only scores. Return choices `pick17`, `pick19`,
`pick20`, `push17` select the original declared TRAIN packet, without relabeling.
This explicit mode excludes `--training-probe` and `--development-probe`;
`--native` adds the final endpoint observations described below. Existing default
and other probe commands keep their prior behavior and refusals.
The main output and `.failure.json` sibling must both be fresh before inputs load.
A late main-capture failure preserves the original paid result in bounded fallback
and stdout, marks root incomplete and exits1. Reports remain limited to512KiB.

The API is `run_demo(payload, adaptation_probe=plain_pair_plan, retention=rule,
max_prototypes=128, return_task="pick17")`. Use an exact `ContinuousProbePlan`
with one plain update, original typed18-row23/24 episodes, diagnostics/velocity
controls off and SGD default. The IO reader `load_reserved_probe_inputs(train_path,
test_path)` keeps strict fixed digests; construct the plain plan explicitly from
its returned episodes (the generic reader's diagnostics=True profile is neural).
The default reader paths are unchanged. Optional guarded endpoint/boundary
callbacks use the existing private-probe contracts; without `--native`, the CLI
provides CPU scores only.

The report's `stream` accounts for source preparation; `adaptation` holds either
one private report or `probe` and `return_phase` reports. Return-old rows are added
exposure even when the distinct replay-receipt field is0. `native_outcomes=None`
separates these scores from the previously observed known-reset robot recovery.
Unknown unreturned private work stays flagged. Earlier paid reports, original
inputs/header/checkpoint/source owners and original clocks/budgets remain guarded.
No source adoption, probe selection, promotion or deployment is provided.

Focused checks: `python -m pytest -q tests/test_retained_adaptation_workflow.py`;
scoped Ruff and win32/Linux mypy cover that file, the new
`src/adapters/taskspace_memory_adaptation.py`, memory demo, existing CLI and
reserved-input reader. Saved qualification slots have cumulative fixture budgets;
do not blindly rerun occupied/SPENT wrappers or old scientific/native cohorts.
Native success from the underlying API remains the scoped
[known-reset return result](reports/retained-memory-return-20261102.md), not a new
before/after native comparison from this CPU command. Generalization, all-task reliability,
CPC benefit and release readiness remain open.

## Native final endpoint after isolated adaptation

Add `--native` to the adaptation command in a prepared optional ManiSkill/SAPIEN
environment. The implementation reuses the existing frozen Pick/Push ports and
CPU adaptation workflow. It runs five fixed fresh scenes at the final isolated
endpoint: Pick17, original23, Pick19, Pick20 and Push17. With an explicit return,
that endpoint is `after_return`; otherwise it is `after_probe`. No updates happen
during these readonly rollouts. Source and stable actor stay unchanged, and the
command exports reports rather than a learner.

```powershell
python -m scripts.run_taskspace_memory_demo --adaptation-probe artifacts/runs/reserved-robot-probe-input-native-20261020/probe_train-model.json artifacts/runs/reserved-robot-probe-input-native-20261020/probe_test-model.json --retention recent_fifo --max-prototypes 128 --return-task pick17 --native --output native-adaptation.json
```

The API is `scripts.taskspace_memory_native_adaptation.run_native_adaptation(payload,
plain_pair_plan, output_path, retention=rule, max_prototypes=128, return_task="pick17",
input_receipts=receipts)`. All six destinations must be fresh before input/model
work: main, `.failure.json` and four named Pick captures (`-pick17`, `-probe23`,
`-pick19`, `-pick20`). Native main/fallback reports are bounded to3MiB and each
Pick capture to1MiB. A complete failed task score is valid; an execution or
integrity failure stops further scenes and preserves paid evidence in a bounded
fallback. Original report owners, raw births and capture bytes/fileidentity are
checked before more work. Capture writes must preserve the original main file.

The report explicitly lists `observed_native_phase` and
`unobserved_native_phases`. Final endpoint outcomes do not establish a native
before/after change in this run. CPU stage scores remain in `workflow`; readonly24
is never native-observed or trained. No reserved24/final/corrections/teacher
authority is added. Existing original23 inference retains its role/history guards.

The SDK setup and software rendering requirements are the same as the ordinary
[native workflow](#optional-native-fresh-run); use the repository's
existing prepared simulator environment. Native integration qualification is
separate from source/literal checks and the historical API result. Do not treat
a CPU fit or saved recording as simulator success. New boundary checks live in
`tests/test_native_adaptation_workflow.py`; earlier CPU/native/math/model cohorts
are reused rather than automatically repeated.

The fixed FIFO128/Pick17-return native command completed with Pick17, original23
and Pick19 successful, Pick20 and Push17 failed. It observed only after_return;
the failed first capture-read allowance and cumulative costs are preserved in the
[native integration report](reports/native-adaptation-workflow-20261102.md).
Scoped29 checks and independent SOURCE/PRE/POST accepted this command path.
Other retention rules/returns, generalization, reliability and release remain
separate qualifications.


## Optional recent-priority memory

Use `--retention recent_priority --retention-seed 17 --max-prototypes 128`
in the existing bounded command, including explicit CPU adaptation or its native
endpoint route. The API policy is
`src.core.nearest_retention.RecentPriorityRetentionPolicy(seed=17)`, passed through
`retention=`. A native uint64 seed is required; totalcapacity2..256. Defaults stay
unchanged; this nearest-only type is rejected by generic/shared/CPC replay.

At128, protect64mostrecent distinct content arrivals and evict the worst seeded
priority outside that window. A retained duplicate refreshes recency without
changing content bytes; old FIFO/reservoir duplicate semantics stay unchanged.
The bank includes the latest64arrivals and global best64priority IDs; their overlap
leaves history-dependent spare entries. It does not reconstruct the best nonrecent
complement from discarded history, and it stores no unbounded seen-ID ledger.
Logical rank work is65evaluations per discard at128; physical validation hashes,
copies, diagnostics and predictions cost more and must be counted separately.

[Software evidence, costs and qualification limits](reports/recent-priority-memory-20261102.md):
40new literal checks pass. Private learn-and-return behavior and mocked native CLI
dispatch pass; actual simulator/scientific benefit for this new mode is unmeasured.
This option changes neither data-role/final access nor probe adoption/promotion.
To extend it, preserve content identity, bounded state and original paid source/
probe guards; use existing ports and freeze any new comparison before execution.


The optional recent-priority mode now has one executed fixed128/64recent/seed17
[final native comparison](reports/recent-priority-comparison-20261102.md):3/5 known
tasks, equalFIFO and belowreservoir4/5. Pick20/Push17fail; old17recordedTRAINfit
returns0, newTRAINfit stays0. Allotheroutcomes/losses/costs and reviewlimitations
are retained. This measures final usefulness only, without native acquisition/
recovery curves, generalization, matchedcompute, default selection or CPC benefit.
Existing controls and earlier failed attempts remain unchanged.


## Qualified existing default-capacity reference

The same ordinary native-adaptation command with `--retention recent_fifo
--max-prototypes 256 --return-task pick17` completed with all5fixedfinal tasks
successful (Pick17/original23/Pick19/Pick20/Push17). It stores236distinct contents
at253training exposures,0capacityevictions, and leaves source/stableactorunchanged.
This uses the existingcapacitydefault, not a new algorithm or implicitdefaultchange.

[Complete results, commands, costs and review scope](reports/default-capacity-adaptation-20261102.md):
newTRAINfit improved, oldfit unchanged, returnchangednone so no recoverycredit.
Allbefore/after nativephases remainunobserved in thiscase. Capacity/storage/copy/
retrieval costs differfrom128controls; no matchedcompute/CPC/generalization/
reliability/release claim follows. Earliernegative andfailedresults remainfrozen.

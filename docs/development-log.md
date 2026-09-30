# Development log

## 2026-09-24 — Checkout recovery and Phase 0 start

### P0.6 — Initialize the living plan and log

Status: complete. Copied the supplied plan from
`C:\Users\Avery\Downloads\Circadian_Predictive_Coding_Development_Plan.md` to
`DEVELOPMENT_PLAN.md`, added its pointer to `AGENTS.md`, and created this log.
The plan's implementation checkboxes remain unchecked unless evidence is recorded here.

Changed files: `AGENTS.md`, `DEVELOPMENT_PLAN.md`, `docs/development-log.md`.
Validation: `git status --short` showed only the new plan before these edits; the
repository content was fetched from the remote without replacing user files.

### P0.1 — Re-audit the checkout

Status: complete. The workspace initially held only an empty `.git` on
`master` with no remote, commits, or working files. `git ls-remote
https://github.com/OptimumAF/Circadian-Predictive-Coding.git HEAD
refs/heads/master refs/heads/main` exited 0 and reported
`8793c49ee4f9f8b07649e8db6571ed53746a9a06` for `HEAD` and `master`.
`git remote add origin ...; git fetch origin master; git checkout -B master
origin/master` exited 0. `git rev-parse HEAD` matches the reviewed commit;
`git diff 8793c49..HEAD --stat` was empty. The fetched tree was clean before
the plan/log edits. No unrelated user changes were present.

Read `AGENTS.md`, `ARCHITECTURE.md`, four existing ADRs,
`docs/circadian-model-review-notes.md`, `requirements.txt`,
`requirements-resnet.txt`, `pyproject.toml`, CI configuration, the test
function/skip inventory, and the implementation paths mapped in
`docs/feature-inventory.md`. No prior `docs/development-log.md` existed.

Machine: Windows 10 build 26200 / PowerShell 7.6.5; Intel Core i7-12700K
(12 cores, 20 logical processors); 68,399,599,616 bytes RAM; NVIDIA RTX 3080
(10,240 MiB VRAM), driver 610.88. Default Anaconda Python is 3.9.12 with
NumPy 2.0.2, pytest 8.4.2, Torch 2.8.0+cu128, and torchvision 0.23.0+cu128;
Torch reports CUDA available. `py -3.11` is Python 3.11.0 but initially lacks
pytest, Torch, torchvision, Ruff, and mypy. The project targets Python 3.11,
so a default-Python run alone cannot establish support for the declared target.

Limitations: the reviewed commit is the remote tip, so there were no upstream
code changes to reconcile; local user changes did not exist in the empty
workspace. Runtime versions and hardware are specific to this machine.

### P0.2 — Baseline quality results on the original checkout

Status: complete. These results were captured before the P1.1 working-tree
edits. No benchmark ranking or scientific claim is inferred from the smokes.

| Environment | Command | Exit | Outcome |
|---|---|---:|---|
| Default Python 3.9.12 | `ruff check .` | 0 | clean |
| Default Python 3.9.12 | `mypy src tests scripts` | 0 | 38 files checked |
| Default Python 3.9.12 | `pytest -q -ra` | 0 | 37 passed, 0 skipped |
| Default Python 3.9.12 | `pytest -q tests/test_resnet50_variants.py tests/test_resnet50_benchmark.py -ra` | 0 | 13 passed, 0 skipped; CPU fixtures, no model-weight download |
| Python 3.11.0 base `.venv` | `.\.venv\Scripts\ruff.exe check .` | 0 | clean |
| Python 3.11.0 base `.venv` | `.\.venv\Scripts\mypy.exe src tests scripts` | 0 | 38 files checked |
| Python 3.11.0 base `.venv` | `.\.venv\Scripts\pytest.exe -q -ra` | 0 | 24 passed; 2 entire Torch modules skipped because optional Torch was absent |

Python 3.11 environment setup: `py -3.11 -m venv .venv` and
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt` exited 0;
the `.venv` directory is ignored. Resolved versions were NumPy 2.4.6,
pytest 9.1.1, Ruff 0.16.9, and mypy 1.20.2. The default Python 3.9
environment had Ruff 0.15.2 and mypy 1.19.1.

Both environments ran the following exact smoke flags, using `python` for
3.9 and `.\.venv\Scripts\python.exe` for 3.11:

```text
predictive_coding_experiment.py --samples 120 --epochs 2 --hidden-dim 8 --seed 7 --sleep-interval 1
scripts/run_continual_shift_benchmark.py --profile baseline --seeds 7 --sample-count-phase-a 120 --sample-count-phase-b 120 --phase-a-epochs 2 --phase-b-epochs 2 --hidden-dim 8 --sleep-interval-phase-a 1 --sleep-interval-phase-b 1
```

All four invocations exited 0 and the corresponding outputs matched across
Python environments. Toy final accuracies were backprop 1.000, PC 0.958,
circadian 1.000. Shift phase-B post accuracies were backprop 0.800, PC 0.800,
circadian 0.833. These are tiny functional smokes under the legacy protocol;
the models are not architecture/compute matched and no outcome is selected.
No files were written; the outputs were stdout only.

### P0.3 — Optional Torch CPU path

Status: complete. `.\.venv\Scripts\python.exe -m pip install -r
requirements-resnet.txt --index-url https://download.pytorch.org/whl/cpu
--extra-index-url https://pypi.org/simple` exited 0, installing Torch
2.14.0+cpu and torchvision 0.29.0+cpu for Python 3.11. Torch reports CUDA
false in this environment. `pytest -q tests/test_resnet50_variants.py -ra`
then passed 7 with 0 skips. The combined 13-test CPU suite was run during
concurrent P1.1 edits and had 12 passes/1 integration failure:
`SyntheticVisionDatasetConfig` had not yet gained `validation_samples`.
This was not an original-checkout failure. After dataset integration,
`.\.venv\Scripts\pytest.exe -q -ra` exited 0 with **42 passed, 0 skipped**,
covering the full NumPy and Torch benchmark/head suite on Python 3.11 CPU.
`test_resnet50_benchmark.py` uses synthetic images and `backbone_weights="none"`,
so no pretrained model weights were downloaded. NumPy-only installation
remains supported by the base requirements and its 24-pass/2-skip result.

### P0.4 — Feature inventory

Status: complete. `docs/feature-inventory.md` records tested, implemented
but unverified, incomplete, and proposed behavior for NumPy and Torch. It
lists direct tests, the differing multilayer algorithms, CLI reachability,
the batch-bounded NumPy replay store, missing Torch replay and durable
checkpoints, and existing experiment-export paths. Commands used:
`rg -n` across core/app/infra/CLI/tests, an AST comparison of config fields
to CLI mappings, and searches for serialization calls. These were read-only;
the inventory's test coverage is grounded in P0.2 outcomes. Relative-link
check passed for its 3 links. No experiment was run for this task.

### P0.5 — Historical result provenance

Status: complete. `docs/historical-benchmark-provenance.md` maps tracked
JSON, text, figures, and dashboard files to recoverable introduction commits,
embedded configurations/seeds, producer scripts, and unknowns. Commands used:
`git ls-files` for JSON/CSV/text/PNG/GIF/HTML, `git log --follow` and
`git log --diff-filter=A` for per-artifact history, `git show` for historical
script defaults, Python `json.loads` for tracked JSON, and `rg -n` for producer
references. All 20 relative links in the provenance document resolved;
`git diff --check` on both new docs exited 0. No historical artifact was
edited. Notable gaps: the chart source CSV is absent; 14-vs-20-epoch
metadata disagree across old files; some dynamics figures predate the current
hardest-case profile. The old test-informed tuning results remain historical
and are not treated as independent held-out evidence.

### P1.1 — Vision split isolation (in progress)

The first increment adds explicit train/validation/test loaders in
`src/infra/vision_datasets.py`; immutable namespaced sample IDs and SHA256
split hashes; independent synthetic validation data; and a deterministic,
unaugmented CIFAR validation view disjoint from the official test and
training indices. `src/app/resnet50_benchmark.py` now uses validation for
epoch stopping and pre/post sleep rollback, reports validation accuracy and
split hashes, and preserves backprop train/eval modes across validation.
Its main runner passes training helpers a view with no test loader, completes
all three training outcomes, then performs final test evaluation.
`src/adapters/resnet_benchmark_cli.py` and
`scripts/run_multiseed_resnet_benchmark.py` expose the validation counts;
the latter selects its descriptive winner by validation accuracy and records
per-seed split hashes. The Pareto and policy sweep scripts now use validation
accuracy for candidate ranking, Pareto dominance, balance, and efficiency
scores. They write exclusive new filenames, preserving tracked historical
JSON. `docs/adr/ADR-0005-vision-validation-split.md` records
the decision; README labels old results as historical. No old result or
figure was rewritten.

Tests added in `tests/test_vision_datasets.py` cover repeatable and disjoint
synthetic/CIFAR splits, immutable hashes, and deterministic CIFAR validation
preprocessing. `tests/test_resnet50_benchmark.py` now checks the validation
decision route, a sealed final loader until all models finish, preserved
backprop mode, and circadian state invariance when only final-test labels
change. `tests/test_tuning_selection.py` checks conflicting validation/test
rankings and exclusive output protection. `python -m pytest
tests/test_vision_datasets.py -q -ra` (default Python 3.9) exited 0 with 3
passes. After all code integration, `.\.venv\Scripts\ruff.exe check .` exited
0, `.\.venv\Scripts\mypy.exe src tests scripts` exited 0 on 40 files,
`.\.venv\Scripts\pytest.exe -q -ra` exited 0 with **52 passed, 0 skipped**, and
`git diff --check` exited 0. Both modified benchmark CLIs answered `--help`
with exit 0 and displayed the new validation flags. No new benchmark
experiment was launched.

A tiny post-change CLI smoke ran with Python 3.11 CPU Torch:

```text
.\.venv\Scripts\python.exe resnet50_benchmark.py --dataset-name synthetic --train-samples 8 --validation-samples 8 --test-samples 8 --classes 3 --image-size 32 --batch-size 4 --epochs 1 --device cpu --target-accuracy -1 --backprop-freeze-backbone --backbone-weights none --pc-hidden-dim 32 --circ-hidden-dim 32 --circ-min-hidden-dim 16 --circ-max-hidden-dim 64 --inference-batches 1 --warmup-batches 0
```

It exited 0, printed all three split hashes and validation/test metrics, and
wrote no artifact. Seed was the CLI default 7; all three models ran one epoch
with 8 training, 8 validation, and 8 test examples. This checks CLI wiring
only. The heads/backbone states and training work are not matched, so the
reported scores and timing deltas are not evidence of an algorithm ranking.

This task is not complete: compatibility single-model helpers still evaluate
test data for each tuning trial, even though the scripts select by validation;
continual-shift phase-A test labels are read before phase-B training. Split
hashes identify sample indices, not source dataset content or augmentation
draws. Exact next action: make tuning trials run without final-test access,
then isolate continual-shift phase-A reporting with model snapshots.

### P1.2 — Leakage regression tests (in progress)

The vision tests above verify stopping and rollback use validation, enforce a
sealed final loader until all three models train, and show that changing only
test labels leaves the circadian head's snapshot state and validation
decisions unchanged for a fixed small run. Replay and continual paths and the
other model states remain to be tested. No task completion is claimed.

### P1.4 — Matched backprop head groundwork (in progress)

`src/core/resnet50_variants.py` now provides `BackpropMLPHead` with the same
feature/hidden/output tensor shapes, tanh activation, and bitwise-equal
same-seed initial values as `PredictiveCodingHead`, held in independent
trainable Torch parameters. A frozen-backbone wrapper is also present. The
existing linear backprop class is unchanged. Three focused CPU tests in
`tests/test_resnet50_variants.py` verify exact initial/logit equality,
independent parameters, a successful SGD step, and wrapper dimensions.
`pytest -q tests/test_resnet50_variants.py -ra` passed 10/0 skips, and the
full 52-test Python 3.11 CPU suite passed after integration. P1.4 remains
unchecked because the matched head is not yet trained in the benchmark on
the same backbone state/features; that is the next app-level increment after
evaluation isolation.

### P1.3 — Continual protocol audit (not complete)

`src/app/continual_shift_benchmark.py` reads phase-A test labels after A
training but before B training; no current training/sleep method uses those
scores. The A/B data containers nevertheless expose test arrays throughout
orchestration, and phase B data are generated before phase A training. The
sleep budget receives the total A+B horizon, so current scheduling assumes
known task duration. A future strict-online variant must address that
assumption. The smallest correction is to preserve the old report by
snapshotting each model after A, finish B, then score A-pre from snapshots
and A-post/B-post from final models in a final-only evaluator. This is
read-only inspection, not a completed implementation or protocol decision.

### P9.1 — Guaranteed Torch CPU CI coverage (in progress)

Added a Python 3.11 `torch-cpu` job to `.github/workflows/ci.yml`. It installs
base requirements and CPU Torch/torchvision wheels, explicitly imports both
packages and asserts a CPU wheel before running the two Torch test modules.
The existing NumPy matrix is unchanged. PyYAML parse/assertion, `git diff
--check -- .github/workflows/ci.yml`, and a Linux CPython 3.11 CPU-wheel pip
dry run exited 0. Local Python 3.11 CPU Torch full suite passed 52/0.
`actionlint` was unavailable and the GitHub job has not run; keep P9.1
unchecked until the actual required CI job passes.

### Session handoff (current)

- Repository commit and working tree: `master` at
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`, tracking `origin/master`.
  The tree now has the intended modified/new plan, documentation, CI, source,
  script, and test files listed by `git status --short`; no unrelated user
  changes were present before this session. No commit or push was made.
- Completed task IDs: P0.1–P0.6.
- Commands and outcomes: `git ls-remote` (0, reviewed commit); `git fetch` and
  `git checkout` (0); `git rev-parse HEAD` (0, reviewed commit);
  `git diff 8793c49..HEAD --stat` (0, empty); initial `git status --short`
  (0, clean fetched tree); environment inspection (0); Python 3.11 base and
  optional CPU Torch installs (0); both smoke commands (0 in Python 3.9 and
  3.11); current `ruff check .` (0), `mypy src tests scripts` (0, 40 files),
  `pytest -q -ra` (0, 52 passed/0 skipped), `git diff --check` (0), and both
  benchmark `--help` invocations (0). Exact flags are listed in P0.2/P0.3
  and P1.1 above.
- Tests: original checkout Python 3.9 had 37 passed/0 skipped; Python 3.11
  base had 24 passed/2 skipped Torch modules; current 3.11 CPU Torch has
  52 passed/0 skipped, Ruff clean, mypy 40 files clean. Interim failures
  during concurrent edits were resolved; see P0.3/P1.1 entries.
- Experiments and artifacts: only tiny deterministic stdout smokes, including
  the 8/8/8 vision CLI check above; no experiment result file was written.
  The ignored `.venv` holds the local Python 3.11 base and CPU Torch installs.
- Plan changes: initialized the living plan in the repository and recorded
  checkout recovery. P1.1 implementation is staged by vision, tuning, and
  continual path; P1.4 head construction precedes app wiring. No acceptance
  criteria changed or unfinished task was marked complete.
- Blockers: P1.1 tuning trials still evaluate test per candidate; continual
  phase-A test remains exposed before phase B. P9.1 remote CI run is pending.
- Exact next action: remove per-trial final-test access from tuning wrappers,
  then snapshot phase-A continual models for final-only evaluation. After
  isolation, wire the matched MLP head into a shared-backbone benchmark track.

## 2026-09-24 — Validation-only trials and deferred continual scoring

### P1.1 — Evaluation isolation increment (still in progress)

Added `ModelDevelopmentReport` and `benchmark_validation_candidate` in
`src/app/resnet50_benchmark.py`. The candidate function accepts only a
train/validation loader view and measures accuracy, cross entropy, and
inference speed on validation. Both `scripts/run_circadian_policy_sweep.py`
and `scripts/run_pareto_hard_tuning.py` now use it, emit no per-trial test
metrics, label inference as validation, and save split hashes. Their output
filenames remain the new validation-selection names and existing historical
JSON is untouched. A frozen winner still requires a separately designed
final-test confirmation; no tuning sweep was launched.

In `src/app/continual_shift_benchmark.py`, phase-A model states are deep-copied
after A training; phase B is generated and trained next; A-pre, A-post, and
B-post test scores are computed only after all three models finish both
phases. Phase B generation now occurs after A training. The planned A+B
horizon still informs sleep scheduling, so this is not yet a strict-online
protocol. Copying three small NumPy model states adds memory proportional to
their state and replay buffers. Tests cover the retained report shape and
the evaluation timing.

### P1.2 — Leakage regression increment (still in progress)

Added a sealed-test-loader candidate test for each vision model and script
boundary tests that fail if a tuning trial reaches a final-test loader.
Extended fixed-seed test-label perturbation checks to backprop and predictive
vision model tensors, and to all three continual NumPy model states including
circadian replay/chemical state (serialized model equality). A timing test
asserts continual test scoring begins only after all three B-training paths
finish. One interim assertion assumed backprop A-post accuracy would change
under flipped labels; it was exactly 0.5 in both cases. The assertion now
checks A-pre, which changed 0.25 to 0.75; the state-invariance comparison
passed in both runs. This is a test assertion correction, not a metric change
to favor a model.

Read-only audit found `scripts/generate_hardest_mode_dynamics.py` still
computes phase-B final-test accuracy and predictions at intermediate epochs,
including during phase A. Its historical figure/output contract must be
versioned before routing in-training snapshots to validation. NumPy dataset
helpers still expose only train/test splits; P1.1 and P1.2 remain unchecked.

### Session handoff

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; the intentional working
  tree edits from the prior session and this increment are uncommitted. No
  unrelated user file or historical result was overwritten.
- Completed task IDs this session: none; P1.1 and P1.2 advanced but retain
  their original acceptance criteria and unchecked status. P0.1–P0.6 remain
  the completed tasks from the previous session.
- Commands and outcomes: `git status --short` and targeted `rg`/file reads
  exited 0; `.\.venv\Scripts\pytest.exe -q -ra
  tests/test_tuning_selection.py tests/test_resnet50_benchmark.py` exited 0;
  `.\.venv\Scripts\pytest.exe -q -ra
  tests/test_continual_shift_benchmark.py` first had the 0.5 assertion
  failure described above, then exited 0 after correction; combined target
  suites exited 0; final `.\.venv\Scripts\pytest.exe -ra` exited 0
  with **62 passed in 10.35s, 0 skipped**. `.\.venv\Scripts\ruff.exe
  check .` exited 0; `.\.venv\Scripts\mypy.exe src tests scripts`
  exited 0 for 40 files; `git diff --check` exited 0 (Git printed only
  Windows LF/CRLF conversion notices).
  After a metadata-only output refinement (`final_test_usage=none`,
  `final_test_confirmation=pending`, policy seed),
  `.\.venv\Scripts\pytest.exe -ra tests/test_tuning_selection.py`
  exited 0 with 8 passed; Ruff and mypy were rerun and exited 0.
- Skipped tests: none in the final Python 3.11 CPU Torch suite. Remote
  GitHub `torch-cpu` CI job, GPU benchmarks, and long sweeps were not run.
- Experiment artifacts: none. The only generated output was pytest's
  temporary policy JSON inside its temporary directory; no benchmark result
  file was written. The local ignored `.venv` is unchanged infrastructure.
- Plan changes: updated P1.1/P1.2 progress and this handoff; added the
  discovered historical dynamics path to their remaining work. Acceptance
  criteria were not reduced and no incomplete task was checked.
- Blockers/unfinished work: deterministic NumPy validation roles/hashes,
  early final-test use in the hardest-mode dynamics script, and direct
  replay/threshold leakage gates. P9.1 still needs a real remote CI pass.
- Exact next action: add deterministic NumPy train/validation/test roles and
  split hashes with focused tests, then route hardest-mode dynamics snapshots
  to validation under a new protocol ID while preserving historical files.

## 2026-09-24 — NumPy split roles and versioned dynamics

### P1.1, P1.2, P1.9 — Corrected hardest-mode route (in progress)

The NumPy dataset infrastructure now provides a deterministic, class-stratified
training/validation partition of the existing train/test generator. Its role
container records immutable SHA-256 content hashes for training, validation,
and final test. Validation rows come only from the former training portion;
the original test portion is unchanged. The corrected dynamics path also
hashes its actual scarce phase-B training subset. Focused dataset tests check
reproducibility, disjointness, exhaustiveness, both-class presence, immutable
hashes, and the effect of changing only final-test labels.

The hardest-mode dynamics script now defaults to protocol
validation_dynamics_v1. Intermediate accuracy, predictions, latency, probe
input, and plot bounds use phase-B validation or training data. Final-test
accuracy is computed only after the last training/sleep event. The HTML
payload names its evaluation role, protocol, split hashes, and final-test
scores. Distinct versioned default filenames and a preflight existence
check protect checked-in historical figures. Explicit
legacy_test_informed_v0 retains the old intermediate test-informed route for
reproduction, with its own distinct default filenames.

The checked-in historical GIF/HTML still represent the legacy path and were
not modified. README and historical provenance now say so.
docs/adr/ADR-0006-numpy-validation-dynamics.md records the decision and
the changed training-set size. The corrected animation is an offline
visualization: phase-B validation metrics are visible during phase A but
never passed to wake training or sleep decisions. It does not satisfy the
strict-online protocol task, and no comparative result is inferred.

Focused tests cover tiny corrected and legacy runs, rendered GIF/HTML,
final-test access only after all three model training paths complete, and
trained-state plus validation-curve invariance when final-test labels flip.
The tiny render wrote files only inside pytest's temporary directory. No
default 300-epoch animation or benchmark sweep was launched.

### Session handoff

- Repository: master at 8793c49ee4f9f8b07649e8db6571ed53746a9a06,
  with intended uncommitted source/docs/test changes. No historical
  figure/result or unrelated user file was overwritten.
- Completed task IDs this session: none. P1.1, P1.2, and P1.9 advanced but
  remain unchecked; P0.1–P0.6 are the completed prior tasks.
- Commands and outcomes: git status --short, git rev-parse HEAD, targeted
  rg/file reads, and the dynamics script --help exited 0. The targeted
  pytest run for tests/test_dataset_roles.py and
  tests/test_hardest_mode_dynamics.py exited 0 with 8 passed. Final
  .\.venv\Scripts\pytest.exe -ra exited 0 with **70 passed in 10.95s,
  0 skipped**. .\.venv\Scripts\ruff.exe check . exited 0;
  .\.venv\Scripts\mypy.exe src tests scripts exited 0 for 42 files;
  git diff --check exited 0, with only Git's Windows LF/CRLF notices.
  Interim mypy failure for a test assignment through a read-only Mapping,
  and an HTML assertion expecting a runtime JS title as literal text, were
  corrected; final checks passed.
- Skipped tests: none in the local Python 3.11 CPU Torch suite. Remote
  GitHub CI, GPU checks, and large experiments were not run.
- Experiment artifacts: no repository output. Pytest temporary GIF/HTML
  files verified the render path and were not kept as scientific results.
- Plan changes: added the completed increment and P1.9 partial
  protocol-version status without changing acceptance criteria or checking
  unfinished tasks. ADR-0006 documents the new split and offline limits.
- Blockers/unfinished work: continual-shift and toy runners retain legacy
  train/test-only containers; direct replay, sleep-trigger, and threshold
  leakage sentinels are still pending. P9.1 awaits a real remote CI run.
- Exact next action: introduce versioned role-separated NumPy inputs for
  continual-shift and toy runners while preserving historical defaults,
  then add direct replay/sleep-trigger/threshold leakage tests before
  considering P1.1 or P1.2 complete.

## 2026-09-24 — Versioned NumPy runners, guarded vision selection, and figure provenance

### P1.1/P1.2 — Corrected NumPy roles and direct leakage checks (in progress)

The toy and continual runners now default to `toy_validation_v1` and
`continual_validation_v1`, respectively. They reserve deterministic,
class-stratified validation examples from the former training portion and
report train/validation/test content hashes. Continual phase B is generated
after phase A training; validation is reserved before its scarce training
fraction is selected. Both final tests are scored after phase B training.
Explicit `toy_legacy_train_test_v0` and `continual_legacy_train_test_v0`
routes preserve the prior data allocation for reproduction. The new
allocation changes training-set size, so old and corrected scores are not
pooled. The CLI exposes protocol and validation-fraction options.

Direct role sentinels verify that NumPy wake learning, replay copies, adaptive
sleep triggering, and threshold paths use current training roles. Final-test
label perturbations leave trained states, loss histories, and validation
metrics unchanged under fixed seeds. A corrected/legacy toy and continual
CLI smoke succeeded with tiny budgets; those numbers are wiring checks, not
research comparisons. The continual output command now preflights and opens
its output exclusively, preserving existing artifacts. ADR-0007 records the
versioned split decision.

The corrected NumPy orchestration functions still bind final-test arrays in
the same scope as their training loops. The tests demonstrate no observed
dependence, but P1.1's stronger structural inaccessibility criterion needs
train-only helper boundaries; P1.2 needs a raising final-test test at those
boundaries. Both tasks remain unchecked.

### P1.3 — Guarded versus strict-online protocol definition (complete)

Vision now defaults to `vision_guard_separated_unmatched_v2`. It reserves a
separate inner guard, outer validation set, and final test; early stopping
and sleep rollback read the guard, while tuning selection reads outer
validation after candidate training. The previous corrected
`vision_validation_unmatched_v1` route remains explicit. Synthetic guard
samples use a separate seed. CIFAR guard and outer validation use disjoint
indices over one deterministic evaluation view, avoiding a third raw-image
copy. Default guard sizes match the earlier inner validation counts: 64
synthetic examples or 1,000 CIFAR examples. Both protocols are still
unmatched-head/backbone comparisons; no algorithm-attribution claim follows.

`docs/evaluation-protocols.md` specifies role access, label timing, held-out
example and memory costs, the current offline limits, and a future
strict-online continual contract. ADR-0008 records the guard/selection
decision. Four-role split tests check hashes, disjoint IDs, shared CIFAR
evaluation view, and deterministic transforms. Vision metric-call sentinels
show inner decisions on guard and final validation/test at their intended
boundaries. Changing outer validation or final-test labels leaves trained
state unchanged for backprop, PC, and circadian models. This completes P1.3
as a protocol definition; a strict-online implementation is still future
study work.

### P1.9 — Protocol identifiers and figure gate (complete)

Vision, toy, continual, and dynamics corrected outputs carry explicit
protocol IDs; their legacy routes have separate IDs. Vision tuning exports
include the protocol and hashes, use guard-separated v2 inputs, and have
new v2 default filenames. The multi-seed runner includes protocol metadata,
preflights all JSON/CSV outputs, and uses exclusive creation. Its winner
functions now receive a whitelist of validation and speed metrics without
final-test fields; the full summary still reports final-test scores
descriptively. Historical provenance distinguishes confirmed test-informed
artifacts from artifacts whose exact execution settings are unknown.

The README figure generator now requires a paired JSON manifest with a
recognized vision protocol and CIFAR-100 source. It verifies each CSV model
and metric against the manifest, rejects duplicate models, defaults to a
protocol-named output directory, and refuses any existing output before
rendering. An explicit `--legacy-unversioned` option labels a manifest-free
source `historical_unknown_v0`. It writes input hashes and protocol to a
provenance JSON beside the new figures. A tiny fixture rendered PNG, HTML,
GIF, and provenance files only in pytest's temporary directory; missing or
mismatched manifests and existing output were rejected. Checked-in figures
were not modified.

### Session handoff

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; intended source/docs/test
  changes are uncommitted. No unrelated user change or historical result was
  overwritten.
- Completed task IDs this session: **P1.3, P1.9**. P1.1, P1.2, P1.4,
  P1.5, and P9.1 remain unchecked. P0.1–P0.6 were completed earlier.
- Commands and outcomes: `git status --short`, `git rev-parse HEAD`, targeted
  `rg`/file reads, both runner `--help` checks, and
  `.\.venv\Scripts\python.exe -m compileall -q src scripts tests` exited 0.
  Toy target tests, continual target tests, vision/guard tests, tuning tests,
  and figure-provenance tests all passed after implementation. An interim
  vision target run failed six tests while mocks and expectations still had
  three roles; an interim tuning run failed two mocks lacking `guard_loader`.
  Both fixture sets were updated and rerun successfully. The final
  `.\.venv\Scripts\pytest.exe -ra` exited 0 with **89 passed in 13.59s,
  0 skipped**. `.\.venv\Scripts\ruff.exe check .` exited 0;
  `.\.venv\Scripts\mypy.exe src tests scripts` exited 0 for 44 files;
  `git diff --check` exited 0 with only Windows LF/CRLF notices.
  After a figure source-caption wording change, targeted figure/multi-seed
  tests exited 0 with 7 passed; Ruff, mypy, and `git diff --check` were
  rerun and exited 0.
- Tiny experiment commands: toy corrected and explicit legacy
  `predictive_coding_experiment.py --samples 80 --epochs 3 --hidden-dim 6
  --sleep-interval 0 --seed 7`; continual corrected and explicit legacy
  `scripts/run_continual_shift_benchmark.py --profile baseline --seeds 3
  --sample-count-phase-a 40 --sample-count-phase-b 40 --phase-a-epochs 2
  --phase-b-epochs 2 --sleep-interval-phase-a 0 --sleep-interval-phase-b 0`;
  vision corrected and explicit v1 `resnet50_benchmark.py --train-samples 8
  --validation-samples 8 --guard-samples 8 --test-samples 8 --classes 3
  --image-size 32 --batch-size 4 --epochs 1 --device cpu
  --backprop-freeze-backbone --pc-hidden-dim 32 --circ-hidden-dim 32
  --circ-min-hidden-dim 16 --circ-max-hidden-dim 64 --inference-batches 1
  --warmup-batches 0 --target-accuracy -1`. Each exited 0. The vision seed
  was the CLI default 7; all used local Python 3.11 CPU Torch/NumPy. Their
  metrics were not used to select seeds, tune baselines, or support a model
  ranking.
- Skipped tests and experiments: no tests skipped in the local suite.
  Remote Torch CI, GPU runs, CIFAR downloads/training, and large sweeps were
  not run. P9.1 still needs an observed remote CI pass.
- Experiment artifacts: no repository benchmark output or new scientific
  figure. CLI smokes printed to stdout. Pytest's tiny generated figures and
  JSON were confined to temporary directories and are not retained as
  research results.
- Plan changes: completed P1.3 as a definition with documented guard costs
  and strict-online contract; completed P1.9 after versioned report and
  figure provenance gates. P1.1/P1.2 remain open on their original strong
  access criterion; P1.4/P1.5 matched representation work remains next.
- Exact next action: extract the toy runner's training loop into a helper
  accepting only train/validation roles and add a final-test object that
  raises if accessed before that helper returns; then apply the same boundary
  to continual shift. After the NumPy gates, wire the matched MLP head and
  shared frozen backbone state into a separate vision track.

## 2026-09-24 — Final-test helper boundaries and matched fixed-feature heads

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Source, tests, plan, and
  docs are intentional uncommitted work. No unrelated user change or
  historical benchmark output was overwritten.
- Completed task IDs: **P1.1, P1.2, P1.4a, P1.4, P1.5**. P1.3 and P1.9 were
  completed in the preceding session; P0.1–P0.6 were completed earlier.
- P1.1/P1.2 evidence: extracted corrected toy training into
  `_train_toy_models`, continual phase-A/B training into
  `_train_phase_a_models`/`_train_phase_b_models`, and corrected dynamics
  training into `_train_dynamics_models`. Their training signatures receive
  only permitted train/guard/validation roles. Raising final-test sentinels
  prove the final roles remain sealed through training. Direct mechanism and
  test-label/state invariance tests from earlier sessions remain in the suite.
  Explicit `legacy_test_informed_v0` reproduction routes remain versioned.
- P1.4a/P1.4/P1.5 evidence: added `src/app/matched_head_benchmark.py`. The
  two-head gate and three-head extension build one frozen, evaluation-mode
  ResNet and cache each role's ordered features once. Backprop MLP, PC, and
  circadian heads start from equal parameter tensors and receive identical
  cached batch objects. The three-head route requires equal starting hidden
  widths, shares the existing circadian dynamics configuration, and makes
  sleep/rollback decisions on inner guard data. Final-test materialization
  waits for all included trainers. Reports record protocol, split, backbone,
  feature, initial-head, and trained-head hashes. The legacy `BackpropResNet50`
  name is retained and labeled a linear-head reference in formatted output.
  ADR-0009 and ADR-0010 document the staged decision and limits.
- Commands and outcomes: `git status --short`, `git rev-parse HEAD`, targeted
  `rg` and file reads reconciled the checkout. The new boundary tests first
  failed because the extracted helpers were absent; a continual test fixture
  briefly intercepted an internal phase-B split and was corrected. Targeted
  toy/continual/dynamics tests then passed. The two-head target had 3 passed;
  its full stage gate was `.\.venv\Scripts\pytest.exe -ra` with 95 passed,
  0 skipped. Three-head tests first failed for the missing API and then
  passed after implementation. A full suite run reached 99 passed/0 skipped;
  its concurrent mypy run found one test-only union assignment annotation,
  which was corrected. `.\.venv\Scripts\mypy.exe src tests scripts` then
  exited 0 for 46 files. Ruff and `git diff --check` exited 0; the latter
  printed only Windows LF/CRLF notices. Final rerun:
  `.\.venv\Scripts\pytest.exe -ra` exited 0 with **99 passed in 15.93s,
  0 skipped**; `.\.venv\Scripts\ruff.exe check .` exited 0;
  `.\.venv\Scripts\mypy.exe src tests scripts` exited 0 for 46 files; and
  `git diff --check` exited 0 with the same line-ending notices.
- Tiny local experiments and artifacts: the target tests run the real ResNet
  twice on CPU with seed 47, 8 synthetic examples per split, 32-pixel images,
  batch size 4, one epoch, and random frozen backbone weights. Backbone and
  feature hashes repeat, initial head hashes match, and the three-head final
  test stays sealed. A fake identity-backbone test checks shared batch-object
  identity and shows that changing outer validation and final-test labels
  leaves all trained-head hashes unchanged, including a forced sleep path.
  No repository benchmark artifact or new scientific figure was written;
  pytest temporary files are not retained as research outputs. The tiny
  checks do not support a model ranking.
- Skipped tests and experiments: no local test skipped. Remote Torch CI,
  CIFAR downloads/training, GPU runs, and large sweeps were not run. P9.1
  remains open pending an observed remote CI pass.
- Plan changes: marked P1.1/P1.2 complete after train-only boundaries and
  leakage regressions; added and completed P1.4a as a two-head dependency;
  marked P1.4/P1.5 complete only after the circadian head joined one shared
  feature bank and passed the three-head gates. P1.6–P1.8 and the Phase 1
  exit gate remain open: track metadata, independent random streams and
  model-order reproducibility, and fairness budgets are unfinished.
- Blockers: no local implementation blocker. Remote CI observation requires
  a remote run. ImageNet-pretrained and larger-data results have not been
  gathered, so the current CPU random-feature gate is a correctness check.
- Exact next action: add explicit backbone trainability, pretraining, head
  type, and parameter counts to every fixed-feature result for P1.6; then
  run a tiny reversed model-order comparison and check trained-state hashes
  and metrics within a declared tolerance for P1.7. Keep the legacy linear
  and end-to-end results in their separate protocol.

## 2026-09-24 — Separate vision tracks and matched-route order gate

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. This session continued
  the intentional uncommitted source/docs/tests work and preserved unrelated
  files and historical artifacts.
- Completed task IDs: **P1.6, P1.7a**. P1.7 and P1.8 remain unchecked; P9.1
  still awaits an observed remote Torch CI pass.
- P1.6 evidence: added track, backbone trainability/pretraining, head type,
  and total/trainable parameter fields to fixed-feature head reports and to
  image-level final/development reports. Added a separate
  `vision_end_to_end_backprop_v1` practical route with an unfrozen ResNet,
  guard-separated validation, and final-test access after training. Existing
  three-model outputs remain `unmatched_reference`. Multi-seed and tuning
  serializers retain metadata; multi-seed winner helpers require explicit
  track metadata and reject a mixture of practical and frozen-shared tracks.
  The old linear-head name and output
  protocol remain intact. ADR-0011 and README describe the track boundary.
- P1.7a evidence: fixed-feature callers can specify a validated head order.
  A tiny real CPU three-head run with forced sleep, repeated in reverse
  order, reproduced backbone, split, role-feature, initial-head, and
  trained-head hashes exactly; accuracy and cross-entropy were equal within
  declared `1e-7` absolute tolerance. A fake loader drawing from the global
  RNG first showed that unrelated backbone draws changed cached features.
  Role-specific CPU RNG scopes now make that perturbation test pass. Synthetic
  dataset/split/shuffle and head/structural-noise generators were already
  explicit; the broad P1.7 criterion still requires image-level reference,
  worker/GPU, and replay-stream audits.
- Commands and outcomes: read `AGENTS.md`, plan handoff, development-log
  tail, current code, and `git status --short`; HEAD matched the reviewed
  commit. New order tests first failed for the missing `model_order` API;
  augmentation perturbation failed before the RNG fix; metadata tests failed
  before report fields existed; the practical test first failed to import
  its new module. After implementation, targeted suites passed. An interim
  mypy run found tuple-width inference and then reused `values` names in
  exporter aggregators; both were corrected. A new export test initially
  omitted its required cross-entropy fixture value and passed after the
  fixture correction. A later missing-track rejection test also passed.
  Final `.\.venv\Scripts\pytest.exe -ra` exited 0 with
  **107 passed in 17.21s, 0 skipped**; `.\.venv\Scripts\ruff.exe check .`
  exited 0; `.\.venv\Scripts\mypy.exe src tests scripts` exited 0 for 48
  files. `git diff --check` exited 0 after the log edit, with only Windows
  LF/CRLF notices.
- Tiny local experiments and artifacts: matched order check used CPU seed
  47, eight synthetic 32-pixel examples per role, one epoch, random frozen
  backbone, and forced sleep configuration; the practical route used CPU
  seed 61, eight examples per role, one epoch, and an unfrozen random ResNet.
  Tests only; no repository benchmark output or scientific figure was
  produced. No seed or metric was selected to favor any model. Training-time
  comparisons between these tracks are not made.
- Skipped tests/experiments: no test skipped locally. No large sweep, CIFAR
  download/training, GPU experiment, or remote CI run was launched. The
  local CPU gate cannot establish worker or GPU order invariance.
- Plan changes: marked P1.6 complete against its separate-track and metadata
  criteria. Added and completed P1.7a as a staged matched-route order/RNG
  gate without weakening P1.7; the older image-level reference and other RNG
  streams remain explicit unfinished work. Phase 1 and fairness task P1.8
  remain open.
- Blockers: no local blocker. Remote CI observation requires a remote run.
- Exact next action: make `run_resnet50_benchmark` accept a validated model
  execution order, run a tiny forward/reversed image-level reference check,
  and isolate per-model loader/augmentation and initialization streams where
  the check exposes coupling. Keep that result in its unmatched protocol;
  then specify P1.8 comparison budgets before any model ranking claim.

## 2026-09-24 — Versioned seeded image-reference route

- Repository state: `master` still points to reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. The source, tests, plan,
  and docs remain intentional uncommitted work. Historical results and
  unrelated user files were not changed.
- Completed task ID: **P1.7b**. Original P1.7 remains open under its full
  random-stream acceptance criteria; P1.8 fairness budgets remain open.
- Finding and implementation: the v2 image route trains three models through
  one mutable shuffled loader, so model order can advance its generator and
  change subsequent inputs. Added opt-in
  `vision_guard_separated_seeded_unmatched_v3` instead of changing v1/v2
  behavior in place. The v3 runner validates a requested model order, forks
  each model's initialization RNG, resets the train-loader shuffle/worker
  generator and CPU augmentation RNG by epoch, and records trained-model
  hashes before final-test scoring. Validation-candidate and single-variant
  compatibility paths also use the seeded trainer when given v3. CLI,
  multi-seed export, and README figure-input validation recognize the new
  protocol; separate protocol IDs keep figure outputs apart. ADR-0012 and
  `docs/evaluation-protocols.md` record its meaning.
- Correctness evidence: a real CPU ResNet run at seed 73 with eight synthetic
  examples per role, one epoch, frozen random backbones, and forced circadian
  sleep produced identical per-model trained-state hashes when model order
  was reversed. Validation/test metrics matched within `1e-7` absolute
  tolerance. A stochastic-loader test repeated two epochs' shuffled random
  views after unrelated process RNG draws and verified the process RNG was
  restored. A sealed final-test loader remained closed until all three v3
  trainers returned. A validation-only candidate test used only the
  train/guard/outer-validation projection.
- Commands and outcomes: read current `AGENTS.md`, plan, development-log
  tail, status, vision loader, benchmark, and scripts; `git rev-parse HEAD`
  matched the reviewed commit. The new tests first failed at collection for
  the absent v3 constant; implementation made them pass. An interim mypy
  run required a test assertion narrowing optional model hashes and then
  passed. Targeted benchmark/figure/multi-seed/tuning tests passed;
  `resnet50_benchmark.py --help` listed v3. A budgeted CLI smoke with
  `--protocol-id vision_guard_separated_seeded_unmatched_v3 --train-samples 8
  --guard-samples 8 --validation-samples 8 --test-samples 8 --classes 3
  --image-size 32 --batch-size 4 --epochs 1 --seed 73 --device cpu
  --target-accuracy -1 --inference-batches 1 --warmup-batches 0
  --backprop-freeze-backbone --pc-hidden-dim 16 --pc-steps 2
  --circ-hidden-dim 16 --circ-min-hidden-dim 16 --circ-max-hidden-dim 32
  --circ-steps 2 --circ-sleep-interval 0` exited 0 and printed v3 plus
  split/model hashes. Final `.\.venv\Scripts\pytest.exe -ra` exited 0 with
  **113 passed in 23.14s, 0 skipped**. Ruff exited 0; mypy found no issues
  across 48 files. `git diff --check` exited 0 with only Windows LF/CRLF
  notices before this final documentation edit.
- Tiny experiment artifacts and interpretation: the CLI smoke printed to
  stdout only; no repository benchmark file or figure was written. Its
  random-feature metrics and speed values are descriptive, not a model
  ranking. Seed 73 was used for a correctness fixture, not selected to favor
  circadian control. No baseline tuning or metric change was used to force a
  win.
- Skipped tests and experiments: no tests skipped locally. No CIFAR
  download/training, multi-worker stochastic image check, GPU reversal,
  remote CI run, or large sweep. P9.1 still needs an observed remote CI pass.
- RNG audit: NumPy circadian initialization and structural noise use separate
  seeded generators; current replay selection is deterministic by priority
  and class balance, with no replay draw to split. Torch circadian split
  noise uses an explicit seeded generator. This is code inspection, not a
  claim of multi-worker or GPU order invariance.
- Plan changes: added/completed P1.7b as a staged, versioned image-reference
  gate. Kept P1.7 unchecked for worker/GPU and broader replay-stream
  reproducibility; P1.8 and the Phase 1 exit gate remain open. Existing v1/v2
  output semantics remain available.
- Blockers: no local implementation blocker. Remote CI observation requires
  a remote run.
- Exact next action: add a picklable multi-worker stochastic-image fixture
  for the v3 seeded loader; if local CUDA is available, run a tiny reversed
  GPU model order and record the state/metric tolerance or negative result.
  Then define/report P1.8 fixed-data, wall-time, and capacity/memory budgets
  before comparative claims.

## 2026-09-24 — Worker RNG evidence and matched-work accounting

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; source, plan, and log
  changes remain intentional uncommitted work. Historical benchmark artifacts
  and unrelated user changes were preserved.
- Completed task IDs: **P1.7b** received stronger state/worker evidence;
  **P1.8a** was added and completed as a staged work-accounting gate.
  Original P1.7 and P1.8 remain unchecked.
- Implementation: the v3 trained-model hash now includes the full circadian
  head snapshot, its structural-noise generator state, and PC traffic state,
  in addition to backbone/head tensors. This fixes the prior overclaim that
  parameter-only hashes represented model state. It is an end-model-state
  hash, not an optimizer checkpoint. A top-level picklable image fixture
  exercises shuffled Torch, NumPy, and Python stochastic draws with two
  DataLoader workers. The matched fixed-feature head reports now include
  wake batches, per-batch latent relaxation iterations, repeated guard
  examples scored, sleep calls, and replay examples. Replay is zero in this
  route. The budget-scope contract is in `docs/evaluation-protocols.md`.
- Correctness evidence: the hash test first failed when changing circadian
  chemical state left the hash unchanged; it now also detects structural RNG
  and PC traffic-state changes. The real v3 CPU reversed-order test still
  matches full state hashes and metrics within `1e-7`. The two-worker fixture
  reproduces two epochs after unrelated main-process RNG draws, while epochs
  themselves differ. A forced-sleep matched-head run with one guard batch per
  stop check reports 8 examples and 2 wake batches per head; backprop/PC/
  circadian report 0/4/4 latent iterations, 4/4/20 repeated guard exposures,
  and zero replay examples; circadian reports one sleep call. Reversing model
  order preserves those counts and the earlier state/metric checks.
- Commands and outcomes: `pytest -q tests/test_resnet50_benchmark.py -k
  trained_model_hash_covers_adaptive_state_and_structural_rng` failed as
  intended before the hash fix; the same test plus the real CPU order test
  then passed. `pytest -q tests/test_resnet50_benchmark.py -k
  seeded_epoch_loader_replays_multiworker_stochastic_images` passed on
  Windows in 18.37s. `pytest -q tests/test_matched_head_benchmark.py -k
  three_head_result_is_invariant_to_model_execution_order` exposed the absent
  work fields, then passed after implementation and guard-cap correction.
  Final `.\.venv\Scripts\pytest.exe -ra` exited 0 with **115 passed in
  41.00s, 0 skipped**. `.\.venv\Scripts\ruff.exe check .` passed; 
  `.\.venv\Scripts\mypy.exe src tests scripts` found no issues in 48 files.
  `git diff --check` exited 0 with Windows LF/CRLF notices. A scoped
  `ruff format --check` returned exit 1 because all four touched Python
  files contain broad pre-existing layout differences; bulk formatting was
  deferred to preserve reviewable, small functional diffs. `python -c
  "import torch; ..."` reported `cuda_available False` and zero CUDA devices.
- Skipped tests and experiments: no local pytest tests skipped. No CIFAR
  download or training, GPU reverse-order check, remote CI run, or large
  sweep. P9.1 still needs an observed remote CI pass.
- Experiment artifacts: tests and the earlier tiny v3 CLI smoke produced
  console output only; no new benchmark result file or figure was written.
  No seed, metric, or baseline was changed to favor circadian control.
- Plan changes: added P1.8a because exact work counts are a testable
  prerequisite for full fairness comparisons. Documented separate fixed-data,
  fixed-wall-time, and capacity/memory scopes without weakening P1.8. Its
  wall-time runner, peak-memory accounting, and matched tuning effort remain
  unfinished. P1.7 still needs torchvision/CIFAR-transform and GPU evidence;
  its existing v1/v2 reference semantics remain versioned.
- Blockers: no local CPU implementation blocker. Local CUDA is unavailable;
  remote CI observation requires a remote run.
- Exact next action: add a tiny torchvision stochastic-transform replay check
  for v3 with workers using already available data and no CIFAR download.
  Then implement a fixed-wall-time matched-head runner with a common deadline
  and explicit timing scope, followed by peak-memory and tuning accounting.

## 2026-09-24 — Torchvision worker check and matched wall-time route

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; this session worked in
  the existing intentional uncommitted tree. Historical outputs and
  unrelated user files were preserved.
- Completed task ID: **P1.8b**. Original P1.7/P1.8 stay unchecked under
  their full acceptance criteria. P1.7 gained a two-worker torchvision
  random-flip/random-crop replay check using in-memory asymmetric images;
  no CIFAR data were downloaded.
- Implementation: added the separate
  `vision_three_head_fixed_feature_wall_time_v1` route. Each matched head
  receives the same positive finite training deadline after shared feature
  and head setup; the timed scope includes per-head optimizer setup, wake,
  relaxation, guard checks, and circadian sleep/rollback. Device work is
  synchronized at deadline checks. The route forbids target-accuracy
  stopping, rejects an epoch cap reached before any head's deadline before
  opening final test, and reports completed epochs, partial wake work,
  stop reason, elapsed time, and operation-boundary overrun. The earlier
  epoch protocol and its timing scope remain versioned. Equal-initialization
  hashes remain parameter-only; trained-head hashes now also cover PC
  traffic and circadian snapshot/structural RNG state. ADR-0013 and
  `docs/evaluation-protocols.md` record the timing boundary and limitations.
- Correctness evidence: the new torchvision fixture reproduced two epochs
  of shuffled random-flip/crop views across unrelated RNG draws with two
  spawned workers. The wall-time tests first failed for the absent route and
  deadline argument, then passed after implementation. A fake clock makes
  two PC wake batches consume a 2.0 s deadline and verifies that a third
  batch and guard check do not start; zero epochs are counted complete.
  Integration tests keep final test sealed until all three deadline trainers
  return, reject zero/negative/nonfinite budgets and target stopping, and
  reject a premature epoch cap before final-test access. An adaptive-state
  hash test first failed for the absent matched state hasher, then passed;
  the reversed-order matched route still reproduces full trained hashes.
- Commands and outcomes: `.\.venv\Scripts\pytest.exe -q
  tests/test_resnet50_benchmark.py -k
  seeded_epoch_loader_replays_torchvision_transforms_with_workers` passed in
  19.16 s. Targeted `pytest -q tests/test_matched_head_benchmark.py -k
  wall_time` passed 3 tests; the full matched file passed 14 tests. An
  interim mypy check caught a test-only direct method reassignment; using
  `monkeypatch.setattr` resolved it. A local Python API smoke used seed 47,
  8 synthetic examples per role, 32-pixel images, frozen random ResNet,
  1,000-epoch safety cap, and `wall_time_budget_seconds=0.1`; it exited 0
  with all three `stop_reason=deadline` and no result artifact. Final
  `.\.venv\Scripts\pytest.exe -ra` exited 0 with **120 passed in 53.59 s,
  0 skipped**. `.\.venv\Scripts\ruff.exe check .` passed and
  `.\.venv\Scripts\mypy.exe src tests scripts` passed for 48 files.
  `git diff --check` exited 0 with Windows LF/CRLF notices.
- Skipped tests and experiments: no local pytest tests skipped. No CIFAR
  download/training, GPU reversal (local CUDA unavailable), remote CI run,
  peak-memory study, capacity sweep, or tuning search. The broad Ruff
  formatter check was not repeated; its previously recorded layout mismatch
  across touched files remains outside this small functional increment.
- Experiment artifacts and interpretation: the tiny wall-time smoke printed
  to stdout only. It yielded 126/146/54 complete epochs and 254/293/109
  wake batches for backprop/PC/circadian under the 0.1 s deadlines. These
  counts verify scheduling and workload reporting on this CPU, not a model
  ranking; there was no seed selection, baseline tuning, or metric change to
  favor circadian control.
- Plan changes: added/completed staged P1.8b because an enforced common
  deadline with explicit scope is independently testable. P1.8 remains open
  for peak host/device memory, capacity matching, equal-effort tuning, and
  repeated confirmation. P1.7 remains open for actual CIFAR and GPU order
  evidence and broader replay-stream reproducibility. No original
  acceptance criterion was weakened.
- Blockers: no local CPU implementation blocker. Local CUDA is unavailable;
  GPU evidence and remote CI require suitable external execution.
- Exact next action: add budgeted peak host-memory telemetry (and device
  memory when available) to the matched-head results, with a tiny allocation
  fixture proving the peak measure differs from `feature_bytes`. Then record
  a capacity-matched and equal-effort tuning protocol before closing P1.8.

## 2026-09-24 — Opt-in observed memory for matched heads

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. This session preserved
  historical artifacts and unrelated changes in the intentional uncommitted
  tree. The new source, tests, ADR, plan, and log remain uncommitted.
- Completed task ID: **P1.8c**. Original P1.8 and P1.7 remain unchecked.
  The Torch CPU CI workflow now includes matched-head and memory tests,
  but P9.1 remains open until an actual remote run is observed.
- Implementation and rationale: `src/shared/process_memory.py` reads current
  process RSS via the Windows working-set API or Linux procfs, and samples it
  every 5 ms plus section boundaries. Unsupported hosts return `None`.
  `measure_memory=True` gives the three-head epoch and wall-time routes
  separate `vision_three_head_fixed_feature_memory_v1` and
  `vision_three_head_fixed_feature_wall_time_memory_v2` IDs. Default runs
  retain their prior IDs without the sampling thread, since polling can
  perturb a time-budgeted run. Each enabled head report includes start and
  highest observed process RSS, sample count, and optional CUDA PyTorch
  allocated/reserved peaks. The sample window covers the trainer and outer
  validation; it excludes shared feature/head setup and final test.
  `feature_bytes` remains a separate static cache-size sum. ADR-0014,
  `docs/evaluation-protocols.md`, README, and shared-module docs explain
  why observed RSS is not an attributable or guaranteed instantaneous peak.
- Correctness evidence: the new memory test first failed collection because
  the module was absent, then passed. A held 32 MiB native allocation made
  observed RSS rise by at least 1 MiB beyond a tiny `feature_bytes` fixture;
  an unsupported-reader test kept start/peak as `None`, never zero. A fake
  CUDA allocator test verified synchronization, baseline read, peak reset,
  and peak allocated/reserved read ordering without claiming real GPU
  validation. Matched-route tests confirm default protocols leave memory
  fields empty, while both opt-in protocols report values on this host.
- Commands and outcomes: `.\.venv\Scripts\pytest.exe -q tests/test_process_memory.py`
  first exited 1 at collection for the absent module; after implementation,
  targeted `pytest -q tests/test_process_memory.py
  tests/test_matched_head_benchmark.py` exited 0 with 19 passed. A budgeted
  Python API smoke used seed 47, 8 synthetic examples per role, 32-pixel
  images, frozen random ResNet, 0.1 s per head, and `measure_memory=True`;
  it exited 0 with the memory wall-time protocol, three `deadline` statuses,
  `feature_bytes=262400`, and RSS peaks of 412332032, 411258880, and
  411549696 bytes for backprop/PC/circadian. Final
  `.\.venv\Scripts\pytest.exe -ra` exited 0 with **125 passed in
  55.61 s, 0 skipped**. `.\.venv\Scripts\ruff.exe check .` passed;
  `.\.venv\Scripts\mypy.exe src tests scripts` passed for 50 files;
  `ruff format --check src/shared/process_memory.py tests/test_process_memory.py`
  passed; `git diff --check` exited 0 with Windows LF/CRLF notices.
- Skipped tests and experiments: no local pytest tests skipped. No CIFAR
  download/training, real CUDA test (CUDA unavailable), remote CI result,
  process-isolated peak study, capacity sweep, tuning search, or large
  experiment. The broad repository formatter check was not repeated; its
  earlier unrelated layout differences remain recorded.
- Experiment artifacts and interpretation: tests and the tiny CPU smoke
  printed to stdout only; no benchmark file or figure was written. The RSS
  numbers are process observations during sequential runs, can miss brief
  allocations, and inherit shared state and allocator history. They do not
  rank model memory efficiency. No seed, metric, or baseline was selected to
  favor circadian control.
- Plan changes: added/completed staged P1.8c for opt-in observed telemetry,
  while retaining full P1.8 for process-isolated confirmation, capacity
  matching, and equal-effort tuning. Default benchmark protocols keep their
  behavior and IDs; sampling is versioned. CI gains local test coverage but
  not a claimed remote pass. No acceptance criterion was weakened.
- Blockers: no local CPU implementation blocker. Real GPU memory/order
  evidence needs a CUDA device; remote CI observation needs a remote run.
- Exact next action: add a validation-only matched-head tuning ledger that
  records every attempted configuration, seed, trial count, guard access,
  and selected candidate under equal trial budgets, with final-test access
  sealed until selection is frozen. Then define a fixed-width/parameter-
  matched capacity control and process-isolated memory confirmation.

## 2026-09-24 — Equal-trial matched-head tuning ledger

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; the intentional
  uncommitted tree, historical artifacts, and unrelated user changes were
  preserved. New module, test, ADR, plan, and log remain uncommitted.
- Completed task ID: **P1.8d**. Full P1.8 and P1.7 remain unchecked. P1.8e
  is added as the next staged capacity-control task; P9.1 still needs a
  remote CI observation.
- Implementation and rationale: `src/app/matched_head_tuning.py` adds
  `vision_matched_head_equal_trial_tuning_v1` for three shared-feature heads.
  Each has the same predeclared candidate count and seed tuple; the API
  rejects unequal or duplicate trials, fixed-setting changes, target-based
  stopping, unfrozen/mismatched backbones, and more than eight trials per
  head. Only head-specific optimization settings may vary. For each seed it
  caches train, guard, and outer-validation features once, resets candidate
  RNG, verifies equal initial tensors, and records full config, hashes,
  validation accuracy, guard/validation access, actual work, and timing.
  Mean validation accuracy selects one candidate per head, with declaration
  order breaking ties. The final-test loader is accessed only after all
  selections are fixed, once per seed, and only selected heads are scored.
  Results keep `attempts`, successful `trials`, `selections`, and final
  `confirmations` separate. A failed candidate raises
  `MatchedHeadTuningError` carrying every completed/failed attempt and
  completed trial row; selection and test do not proceed. Equal trial count
  and validation access do not equalize per-trial compute.
- Correctness evidence: the sealed-loader test checks test access only
  after selection; changing only final-test labels leaves trial rows,
  trained-state hashes, and selections unchanged. Reversing candidate order
  leaves each seeded candidate trial unchanged apart from elapsed time.
  Other tests cover equal budgets, fixed-setting rejection, duplicate
  settings, deterministic ties, and an injected failure that retains the
  attempted config/seed without opening test. A real frozen random ResNet
  CPU smoke used 8 synthetic examples per role, 32-pixel images, seed 47,
  one epoch, and two candidate learning rates per head. It returned six
  trial rows, two per head, eight guard examples per trial, and three
  selected-head confirmations. All three selected candidate `a` in this
  tiny smoke; this is only a reproducibility observation, not a model
  comparison or a tuned result for scientific interpretation.
- Commands and outcomes: `.venv\Scripts\pytest.exe -q -ra
  tests/test_matched_head_tuning.py` exited 0 with **8 passed**. The first full
  `.venv\Scripts\pytest.exe -ra` gate passed 132 tests before the failure
  ledger increment; the final gate exited 0 with **133 passed in 57.87 s,
  0 skipped**. `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for 52 files; targeted
  `ruff format --check` passed after formatting the two new Python files.
  `git diff --check` exited 0 with the existing Windows LF/CRLF notices.
  The inline `.venv\Scripts\python.exe -` real-backbone smoke exited 0
  with protocol ID, trial count, confirmation count, role keys, guard counts,
  and selected candidate IDs printed to stdout.
- Skipped tests and experiments: no local pytest tests skipped. No CIFAR
  download/training, CUDA run (unavailable locally), remote CI observation,
  large sweep, process-isolated memory comparison, or repeated-seed
  performance study. No benchmark ranking or baseline/seed/metric tuning
  was attempted to favor circadian control.
- Experiment artifacts: the smoke and tests emitted stdout only; no new
  benchmark JSON, CSV, figure, or external artifact was written. The typed
  result is machine-readable in memory; failed attempts are carried by the
  exception. Durable journaling is still needed before long searches.
- Plan changes: added and completed P1.8d as a bounded selection-isolation
  gate, added unchecked P1.8e for a fixed-width parameter-matched control,
  and updated the P1.8 handoff. ADR-0015, README, app-module docs, protocol
  table, and the Torch CPU CI test list now describe the route. Original
  P1.8 criteria remain open for capacity matching, process-isolated memory,
  and repeated confirmation; no acceptance criterion was weakened.
- Blockers: no local CPU implementation blocker. CUDA order/device memory
  evidence requires a GPU; remote CI evidence requires a remote run.
- Exact next action: implement P1.8e with a tiny fixed-width forced-sleep
  fixture, verifying equal initial/final head parameter counts, no
  splits/prunes, shared feature hashes, and explicit capacity-control
  metadata. Then add process-isolated memory confirmation and repeated
  confirmation before closing P1.8.

## 2026-09-24 — Fixed-width capacity control with guarded sleep

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. All prior intentional
  uncommitted work and unrelated user changes were preserved. The new test,
  ADR, plan, and log edits remain uncommitted.
- Completed task ID: **P1.8e**. Full P1.8 and P1.7 remain unchecked. Added
  unchecked P1.8f for process-isolated memory confirmation; P9.1 still
  awaits an observed remote CI run.
- Implementation and rationale: `src/app/matched_head_benchmark.py` now has
  `run_three_head_fixed_width_capacity_benchmark`, a separately versioned
  `vision_three_head_fixed_width_capacity_memory_v1` fixed-data/epoch route.
  It requires one equal predictive/circadian width, with circadian minimum
  and maximum both fixed at that width, target-accuracy stopping disabled,
  scheduled forced sleep within the epoch cap, an executable sleep budget,
  and guard-based rollback. Before final-test materialization it verifies
  equal initial/final head parameter counts, unchanged circadian width, no
  splits/prunes, and at least one guarded sleep attempt. The result carries
  explicit `capacity_control` metadata and always enables observed process
  RSS/CUDA allocator telemetry, separate from cached `feature_bytes`. The
  existing adaptive-width and wall-time route IDs remain unchanged.
- Correctness evidence: the new real-ResNet CPU fixture spied on the
  circadian sleep call and observed chemical-state change with the same
  parameter count before/after and empty split/prune indices. A paired
  adaptive-width invocation retained `vision_three_head_fixed_feature_v1`
  and reproduced the backbone, role-feature, and initial-head hashes.
  Invalid fixed-width/sleep/rollback configurations fail before loading
  data; an injected final-parameter mismatch fails before a sealed test
  loader can be iterated. The tiny one-epoch, eight-example-per-role smoke
  reported 32,835 head parameters for each head both initially and finally,
  one circadian sleep attempt, zero splits/prunes, guard-example counts
  8/8/24, `feature_bytes=262400`, and sequential observed RSS peaks of
  412897280/413454336/414236672 bytes. These RSS values include shared
  state and execution-order effects; they are not a head memory ranking.
- Commands and outcomes: the new `pytest -q -ra
  tests/test_matched_head_capacity.py` gate first failed with nine missing-
  API cases as intended, then passed after implementation. Final targeted
  `.venv\Scripts\pytest.exe -q -ra tests/test_matched_head_capacity.py
  tests/test_matched_head_benchmark.py` exited 0 with **29 passed**. The
  inline `.venv\Scripts\python.exe -` smoke exited 0 with the values above.
  Final `.venv\Scripts\pytest.exe -ra` exited 0 with **145 passed in
  59.54 s, 0 skipped**. `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for 53 files;
  `ruff format --check src/app/matched_head_benchmark.py
  tests/test_matched_head_capacity.py` passed after formatting the touched
  files; `git diff --check` exited 0 with existing Windows LF/CRLF notices.
- Skipped tests and experiments: no local pytest tests skipped. No CIFAR
  download/training, CUDA execution (unavailable locally), remote CI
  observation, large sweep, process-isolated memory comparison, or repeated
  seed confirmation. The first targeted formatter check found layout
  differences in the existing matched-head module; the touched module and
  new test were formatted, then the targeted check passed. A broad
  repository formatter check was not run because other prior layout
  differences are recorded in earlier sessions. No seed, baseline, or
  metric was chosen to favor circadian control.
- Experiment artifacts: the CPU smoke printed to stdout only. No benchmark
  JSON/CSV, figure, or external artifact was written. ADR-0016, the
  protocol table, README, app-module docs, and Torch CPU CI test list were
  updated. The RSS values are observed in one sequential process and can
  miss transient allocations.
- Plan changes: completed P1.8e after its parameter, sleep, guard, hash,
  memory-reporting, and test-sealing criteria passed. Added P1.8f as an
  explicit process-isolated memory gate without weakening full P1.8.
- Blockers: no local CPU implementation blocker. GPU order/device-memory
  evidence requires CUDA hardware; remote CI evidence requires a remote
  run.
- Exact next action: implement P1.8f's child-process CPU measurement
  boundary for one matched head at a time, preserving identical frozen
  feature/initial hashes and final-test sealing. Record setup/cache and
  trainer RSS scopes separately, then run one tiny three-head confirmation.

## 2026-09-24 — Process-isolated matched-head memory boundary

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Prior intentional
  uncommitted work and unrelated user changes were preserved. The new
  module, test, smoke script, ADR, plan, and log remain uncommitted.
- Completed task ID: **P1.8f**. Full P1.8 and P1.7 remain unchecked. Added
  unchecked P1.8g for predeclared repeated confirmation. P9.1 still needs
  an observed remote CI run.
- Implementation and rationale: `src/app/isolated_head_memory.py` adds
  `vision_three_head_fixed_width_process_memory_v1`. The parent validates
  the fixed-width guarded synthetic config and starts each matched head in
  a fresh spawned process with a bounded timeout. A child reconstructs the
  same seeded frozen backbone and cached train/guard/validation features,
  initializes only its assigned head, and trains through the existing
  matched-head trainer. It never reads the test loader or scores final test.
  The parent verifies matching split, feature, backbone, and initial-head
  hashes, feature bytes, and initial/final parameter counts. Setup RSS
  sampling begins after Torch import/device resolution and covers loader,
  backbone, feature, and head setup; a separate trainer RSS window covers
  wake, guard/sleep, and outer validation. Reports also carry pretraining
  RSS, cached tensor bytes, work counts, and optional CUDA allocator peaks.
  PID reports are checked against the spawned process; reused OS PIDs are
  not treated as proof of a shared process. The route is a descriptive
  memory observation, not a test-score or memory-winner protocol.
- Correctness evidence: a direct worker sentinel makes `test_loader`
  inaccessible and still returns train/guard/validation hashes. A parent
  mismatch fixture rejects a changed guard feature hash. Invalid capacity
  and timeout inputs are rejected before spawning. The real CPU test and
  repeatable `scripts/run_isolated_head_memory_smoke.py` use seed 47,
  8 synthetic examples per role, one epoch, and 16 hidden units. Three
  children produced identical backbone, feature, and initial-head hashes,
  32,835 head parameters, 196,800 cached train/guard/validation bytes, and
  guard exposures 8/8/24 with one circadian sleep attempt. The smoke
  recorded setup-start RSS around 202 MB, pretraining RSS around 411 MB,
  and trainer observed peaks around 413 MB; these are separate-process
  observations with runtime/backbone/cache costs, not head-attributable
  allocations or a ranking.
- Commands and outcomes: first `.venv\Scripts\pytest.exe -q -ra
  tests/test_isolated_head_memory.py` run reached the real spawned-process
  gate but had one case-sensitive expected-error mismatch; after fixing it,
  final targeted isolated/capacity tests passed **17**. The repeatable
  `.venv\Scripts\python.exe scripts/run_isolated_head_memory_smoke.py`
  exited 0 and printed protocol, per-head PIDs, hashes, cache bytes, setup
  and trainer RSS, and guard counts as JSON to stdout. Full
  `.venv\Scripts\pytest.exe -ra` exited 0 with **150 passed in 72.78 s,
  0 skipped**. After adding a spawned-PID identity check, targeted
  `pytest -q -ra tests/test_isolated_head_memory.py` passed **4**.
  `.venv\Scripts\ruff.exe check .` passed; `.venv\Scripts\mypy.exe src
  tests scripts` passed for 56 files; targeted `ruff format --check` passed
  for the new module, test, and script. `git diff --check` exited 0 with
  existing Windows LF/CRLF notices.
- Skipped tests and experiments: no local pytest tests skipped. No CIFAR
  download/training, real CUDA allocation run (unavailable locally),
  remote CI observation, large sweep, final-test score, or repeated-seed
  comparative study. CUDA allocator hooks retain their earlier fake-device
  coverage, but this isolated route has only real CPU evidence. No
  baseline, seed, or metric was selected to favor circadian control.
- Experiment artifacts: the smoke emitted JSON to stdout only; no
  benchmark file, figure, or external artifact was written. The returned
  typed report is machine-readable. ADR-0017, README, app-module docs,
  evaluation-protocol table, and the Torch CPU CI test list were updated.
- Plan changes: completed P1.8f after real spawned-child parity, memory
  boundary, capacity, and final-test-sealing gates passed. Added P1.8g for
  predeclared paired repeats while leaving full P1.8 open for repeated and
  larger-data/GPU confirmation. No original acceptance criterion was
  weakened.
- Blockers: no local CPU implementation blocker. Device-memory evidence
  requires CUDA hardware; remote CI evidence requires a remote run.
- Exact next action: implement P1.8g's predeclared tiny CPU confirmation
  manifest and result schema. Freeze candidates, seeds, metrics, and budget
  scopes before opening final test; run at least three paired seeds, retain
  every result including negative ones, and report dispersion and work
  without selecting a favorable seed or declaring a head-family winner.

## 2026-09-24 — Predeclared three-seed matched-head confirmation

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; the prior intentional
  uncommitted work and unrelated user changes were preserved. This session's
  module, test, script, ADR, docs, plan, and log changes are uncommitted.
- Completed task ID: **P1.8g**. Full P1.8 and P1.7 remain unchecked. P9.1
  still needs observation of a remote CI run.
- Implementation and rationale: `matched_head_tuning.py` now offers
  `confirm_test=False`, returning the same bounded validation ledger and
  selections without reading the final-test loader; the default test
  confirmation route is preserved. Trial rows now include starting parameter
  counts and structural work for capacity verification.
  `repeated_head_confirmation.py` freezes a validation-only selection into
  a digested manifest with candidate IDs/configs, disjoint confirmation
  seeds, accuracy/cross-entropy metrics, fixed-data/epoch, per-head wall-time,
  and process-isolated capacity/memory scopes. The runner keeps raw reports
  and separate accuracy/RSS descriptive summaries. It rejects incomplete
  seed/head sets, changed manifests, mismatched split/feature/backbone/
  initial-head hashes, changed fixed-width parameters, and wall-time runs
  that miss their deadlines. It does not select a winner. ADR-0018, README,
  app-module docs, evaluation protocol, and Torch CPU CI suite were updated.
- Correctness evidence: a sealed fixture proves validation selection and
  manifest creation never read test; a separate three-seed fixture proves
  all nine fixed-data training runs finish before the first test-loader
  access, requires nine confirmation rows, and rejects a missing row.
  Manifest overlap, test-informed source, and post-freeze mutation fail.
  The real smoke saved selection and manifest JSON at 22:52:04 local before
  the result at 22:52:42. It selected candidate `a` for all three heads on
  seed 47 from two candidate learning rates per head, before final test.
  Confirmation seeds 53/59/61 yielded 9 complete fixed-data attempts,
  9 trials, 9 test rows, 3 wall-time reports, and 3 isolated-memory reports
  (one process per head per seed). Per-seed split, feature, backbone, and
  initial-head hashes matched across scopes; all heads retained 32,835
  starting/final parameters and zero structural splits/prunes. Fixed-data
  work per seed was 8 examples and 2 wake batches/head, 0/4/4 relaxation
  steps, 8/8/24 guard exposures, and 0/0/1 sleep attempts. Wall-time
  deadlines all fired around 0.05 s/head; wake batches ranged 109–133
  backprop, 140–156 PC, and 56–58 circadian, with circadian guard exposures
  664–688 and 28–29 sleep attempts. These are unequal work counts under
  the common deadline, not a compute-equivalent epoch claim.
- Experimental outcome: fixed-data per-seed test accuracies were
  `[0.375, 0.125, 0.25]` for every head (mean 0.25, population SD 0.102).
  Wall-time accuracies were backprop `[0.125, 1.0, 0.375]` (mean 0.5), PC
  `[0.125, 1.0, 0.25]` (mean 0.458), and circadian
  `[0.125, 0.125, 0.25]` (mean 0.167). The circadian negative result was
  retained without changing seeds, metrics, candidates, or budgets.
  Isolated trainer observed RSS means were about 412.8/412.7/413.7 MB for
  backprop/PC/circadian. These include runtime/backbone/cache costs and
  are not exact incremental head memory; all raw per-seed values and work
  are in the result artifact. The tiny random-feature inputs do not support
  a general head-family ranking.
- Commands and outcomes: `.venv\Scripts\pytest.exe -q -ra
  tests/test_matched_head_tuning.py tests/test_repeated_head_confirmation.py`
  passed **13**. `.venv\Scripts\python.exe
  scripts/run_repeated_confirmation_smoke.py` exited 0 in about 40 s.
  `.venv\Scripts\pytest.exe -ra` passed **155 in 69.08 s, 0 skipped**.
  `.venv\Scripts\ruff.exe check .` passed; `.venv\Scripts\mypy.exe src
  tests scripts` passed for 59 source files; targeted `ruff format --check`
  passed for 5 changed Python files. `git diff --check` exited 0 with
  existing Windows LF/CRLF conversion notices.
- Skipped tests and experiments: no local pytest tests skipped. No CIFAR
  download, larger-data sweep, real CUDA run (unavailable locally), or
  remote CI observation. The wall-time and memory scopes do not reselect
  candidates from final test. No baseline, seed, or metric was tuned to
  favor circadian control.
- Experiment artifacts: ignored local files
  `artifacts/benchmark_repeated_selection_smoke.json` (72,851 bytes),
  `artifacts/benchmark_repeated_manifest_smoke.json` (19,677 bytes), and
  `artifacts/benchmark_repeated_result_smoke.json` (188,983 bytes). The
  manifest digest is
  `e313e290af6c737460e4ae5e573f8ca9ec35e3c65025ef1f5ae675ff9802da2e`.
  The script refuses to overwrite artifacts, so reruns need a deliberately
  separate output location or explicit archival of the prior evidence.
- Plan changes: P1.8g was completed after sealed selection, manifest
  freezing, all-seed evidence, raw work/dispersion reporting, and the full
  quality gate. P1.8 remains open for larger-data and real CUDA evidence;
  P1.7 remains open for broader stream replay and execution environments.
  No acceptance criterion was weakened.
- Blockers: none for the next local code audit. Real CUDA telemetry needs a
  CUDA host; actual CIFAR data are not present in this checkout, and remote
  CI requires observing an external run.
- Exact next action: inspect the NumPy and Torch structural-noise and replay
  RNG paths for P1.7, then add a small reverse-execution-order seeded fixture
  that checks replay selections, structural decisions, and resulting state
  hashes without downloading data. Keep P1.7 unchecked until its broader
  criterion is verified.

## 2026-09-24 — Toy replay and structural-noise order gate

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Prior intentional
  uncommitted changes and unrelated user work were preserved. This session
  changed the toy runner, two existing test files, README, app/evaluation
  docs, ADR-0019, plan, and log; all remain uncommitted.
- Completed task ID: **P1.7c**. P1.7 and P1.8 remain unchecked; added
  unchecked **P1.7d** for the continual phase-A/B order gate. P9.1 still
  needs a remotely observed CI run.
- Inspection and rationale: NumPy circadian initialization uses a local
  `default_rng(seed)` and noisy splitting uses a separate
  `default_rng(seed + 10001)`. Replay selection is deterministic priority
  sorting with optional class balancing, not a random draw. The Torch head
  initializes its own split generator at `seed + 9999`. The NumPy toy
  runner, however, could not reverse training order, so inspection alone
  could not verify its actual replay and split paths under reordering.
- Implementation: `ExperimentConfig.model_order` is a validated permutation
  of the three toy models. It is checked before dataset loading, defaults to
  the former order, controls each epoch's training order, and is recorded in
  `ExperimentResult` and the formatter. Existing toy protocol IDs and split
  roles remain unchanged. `tests/test_experiment_runner.py` reverses an
  80-sample, six-epoch CPU run with prioritized replay and nonzero split
  noise, captures actual replay priorities and split/prune decisions, and
  hashes serialized trained model objects including NumPy RNG state.
  The test compares all three model hashes, role hashes, losses, validation
  and test metrics, and sleep counts after unrelated global NumPy draws.
  `tests/test_resnet50_variants.py` reverses two CPU Torch heads with actual
  noisy splits amid unrelated global Torch draws and compares split indices
  plus the existing trained-head hash that includes structural RNG state.
  Invalid toy permutations fail before the generator is called. ADR-0019
  explains this local scope; README and module/protocol docs show the API.
- Commands and outcomes: the new three-test gate initially had two expected
  failures because the toy order field/constant did not yet exist; after
  implementation, those **3 passed**. `.venv\Scripts\pytest.exe -q -ra
  tests/test_experiment_runner.py tests/test_resnet50_variants.py` passed
  **19**. `.venv\Scripts\pytest.exe -ra` passed **158 in 69.32 s, 0
  skipped**. `.venv\Scripts\ruff.exe check .` passed and
  `.venv\Scripts\mypy.exe src tests scripts` passed for 59 source files.
  `.venv\Scripts\python.exe predictive_coding_experiment.py --samples 80
  --epochs 3 --hidden-dim 4 --sleep-interval 1 --replay-steps 1` exited 0
  and printed the default order with guarded split hashes. Targeted
  `ruff format --check` returned 1: these existing files contain prior
  layout/line-ending differences; an untouched core file also returns 1.
  The session did not mass-reformat unrelated lines. `git diff --check`
  exited 0 with existing LF/CRLF notices.
- Skipped tests and experiments: no pytest tests skipped. No CIFAR download,
  real GPU run (not available locally), large sweep, or final-test-driven
  seed/configuration selection. The direct Torch test is a head-only CPU
  gate, not evidence about ResNet loaders or CUDA kernels.
- Experiment artifacts: no new benchmark file was written; deterministic
  state/replay/decision comparisons live in the tests. ADR-0019 is the
  decision record. Earlier P1.8g JSON artifacts remain untouched.
- Plan changes: P1.7c was added before implementation as a local stream
  gate, then marked complete after actual reversal, full tests, lint, and
  type checks; the pre-existing formatter mismatch is recorded above.
  P1.7d is an explicit next local gate because the continual phase-A/B
  runner still has fixed model order. Full P1.7 acceptance remains unchanged
  for continual ordering, actual CIFAR/GPU, and broader stream evidence.
- Blockers: none for P1.7d. Actual CIFAR data are not in this checkout; GPU
  numerical checks require a CUDA host. A repository-wide formatter pass
  would change pre-existing layout beyond this task.
- Exact next action: inspect `_train_phase_a_models` and
  `_train_phase_b_models` in `src/app/continual_shift_benchmark.py`,
  including phase state copies and sleep/replay boundaries. Then add the
  validated model-order option and a tiny reverse-order phase-A/B fixture
  with real replay/splitting, without changing legacy/default results.

## 2026-09-24 — Continual phase order and learning-equation specification

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. All prior intentional
  uncommitted files and unrelated user changes were preserved. This session
  changed the continual runner/test, README, app/core/evaluation docs,
  ADR-0020, `docs/learning-mathematics.md`, plan, and log. No commit was made.
- Completed task IDs: **P1.7d** and **P2.1**. P1.7, P1.8, P2.2, and P9.1
  remain unchecked. P1.7's actual CIFAR/GPU and broader stream evidence,
  P1.8's larger-data/CUDA fairness evidence, and the separate strict-online
  continual label-timing issue are still open.
- Inspection and rationale: Phase A deep-copies trained models before Phase B
  data construction; corrected final-test roles are scored after Phase B.
  Both phases had fixed model order, with circadian sleep after all wake
  updates. The replay and noisy split streams are local to the circadian
  model. A validated permutation therefore provides a direct small-run
  invariance gate without changing the phase state-copy or sleep boundary.
  The subsequent core audit found that NumPy PC uses fixed feedforward
  priors in multilayer relaxation, while NumPy circadian relaxes only its
  last hidden layer; Torch PC uses CE output residuals but reports a
  squared-residual diagnostic. Those differences require explicit equations
  before numeric gradient or deeper attribution work.
- Implementation: `ContinualShiftConfig.model_order` defaults to the former
  three-model sequence, rejects invalid permutations before any data load,
  and controls one shared per-epoch helper in both phases. Each seed result
  and formatted report records the order. A tiny seed-13 corrected-protocol
  fixture (80 source examples per phase, 3+3 epochs) reverses the sequence
  with prioritized replay and real noisy splits after unrelated global
  NumPy draws. It checks six phase-role hashes, replay priorities/snapshot
  hashes, structural decisions, saved Phase A and final serialized model/RNG
  hashes, and all retention/adaptation reports for equality. A sealed-role
  test covers both orders, and an invalid-order sentinel forbids even data
  loading. Default and legacy routes retain their IDs. ADR-0020 records the
  choice and scope. `docs/learning-mathematics.md` specifies implemented
  NumPy/Torch priors, residuals, losses, latent/weight steps, normalization,
  chemical/reward gates, training diagnostics, and feedforward evaluation.
  README and core module docs link it. It explicitly leaves energy/update
  reconciliation and numeric gradient tests under P2.2/P2.3.
- Commands and outcomes: two new order tests first failed as expected before
  implementation (missing order constant/field), then passed.
  `.venv\Scripts\pytest.exe -q -ra tests/test_continual_shift_benchmark.py`
  passed **13**; `.venv\Scripts\pytest.exe -ra` passed **161 in 79.14 s,
  0 skipped**. `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for 59 source files.
  `.venv\Scripts\python.exe scripts/run_continual_shift_benchmark.py --help`
  exited 0. Two tiny one-seed CLI smokes (baseline profile, 80 examples per
  source phase, one epoch per phase, hidden width 4) exited 0 under
  `continual_validation_v1` and `continual_legacy_train_test_v0`; neither
  wrote a benchmark artifact. Both printed descriptive balanced scores of
  0.300 for backprop, 0.125 for PC, and 0.275 for circadian; no selection or
  tuning followed. An initial legacy CLI attempt used a guessed
  invalid protocol ID, exited 2, and was immediately retried with the
  documented ID. `git diff --check` exited 0 with the existing Windows
  LF/CRLF notices. Targeted `ruff format --check` exited 1 on the two
  continual Python files: its diff includes mixed line endings and
  pre-existing layout changes; an untouched core file already failed this
  check in the prior session. No broad formatting rewrite was made.
- Skipped tests and experiments: no pytest tests skipped. No CIFAR download,
  large sweep, real CUDA run (unavailable locally), remote CI observation,
  or final-test-driven seed/configuration selection. The new order fixture
  is a local CPU reproducibility check, not a circadian advantage claim.
- Experiment artifacts: no new result file. The local fixture's hashes and
  decisions are asserted in tests. ADR-0020 and the mathematical spec are
  documentation artifacts; earlier P1.8g JSON artifacts remain untouched.
- Plan changes: P1.7d was checked only after replay/structure/state/metric
  reversal, sealed and invalid-order tests, both CLI route smokes, and the
  full suite. P2.1 was checked after deriving all current variant equations
  and evaluation paths from the source. P2.2 remains unchecked for explicit
  diagnostic normalization tests and any decision to align or relabel
  reported energy. No acceptance criterion or negative result was weakened.
- Blockers: none for P2.2. Actual CIFAR files are absent in this checkout;
  GPU checks require a CUDA host. Repository-wide formatting currently has
  pre-existing layout/line-ending differences and would touch unrelated
  lines.
- Exact next action: add a tiny deterministic normalization/diagnostic
  fixture for the NumPy and Torch PC heads, then label each emitted training
  energy with its actual formula and scope without changing legacy numeric
  values. Decide explicitly whether each energy can be aligned to its
  intended update gradient; leave P2.2 unchecked until its audit and tests
  pass.

## 2026-09-24 — Training diagnostics and local gradient contracts

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; all prior intentional
  uncommitted files and unrelated user changes were preserved. No commit or
  benchmark artifact was created. Local runtime: Windows CPU, Torch
  `2.14.0+cpu`, NumPy `2.4.6`, `torch.cuda.is_available() == False`.
- Completed task IDs: **P2.2** and staged **P2.3a**. Full P2.3, P1.7,
  P1.8, and P9.1 remain unchecked. Added unchecked **P2.3b** for deeper
  circadian and Torch gate boundaries.
- Inspection and rationale: NumPy PC reports binary BCE plus half the mean
  squared residual over all hidden units; NumPy circadian uses only its
  adaptive hidden width. Torch PC/circadian reports separately averaged
  squared output and hidden residuals, although its update uses the
  cross-entropy residual. A fixed hidden error's diagnostic contribution
  falls as width rises, potentially affecting adaptive sleep. Replacing
  these values would alter historical curves and sleep decisions, so
  ADR-0021 retains the numbers and names them as training diagnostics.
  The one-hidden ungated sum-residual objective has coherent local partials;
  ordinary multilayer all-latent PC currently lacks the corresponding
  scalar objective and remains under P2.6.
- Implementation: stable metric IDs accompany NumPy PC core train results,
  toy and in-depth reports, Torch PC head/report results, multiseed and
  validation-tuning JSON/CSV rows, and the interactive dynamics payload.
  New text reports call the PC values training diagnostics; the dynamics
  chart says each model's metric is normalized separately. Existing
  `energy`, `final_energy`, and numeric series remain unchanged. Aggregators
  reject mixed metric IDs. `docs/learning-mathematics.md`,
  `docs/evaluation-protocols.md`, core module docs, and ADR-0021 record the
  formulas, width/class denominators, pre-update timing, and limits.
  `tests/test_energy_diagnostics.py` independently reconstructs two NumPy
  widths, NumPy circadian, and Torch PC/circadian values, including a Torch
  autograd counterexample to treating squared-output energy as CE.
  `tests/test_local_gradient_contracts.py` uses deterministic float64
  central differences and test-only Torch autograd for binary/multiclass
  latent and held-state parameter partials. It checks actual one-step NumPy
  and Torch PC/neutral circadian updates, two-hidden NumPy backprop,
  duplicate-batch averaging, and NumPy chemical-gate scaling. It does not
  claim a multilayer all-latent PC objective or a sleep gradient.
- Commands and outcomes: the first diagnostic tests produced **3 expected
  failures** for the missing definition field; after adding metadata,
  targeted app tests exposed **4 integration failures** from placing the
  field on `_TrainingOutcome` instead of `ModelDevelopmentReport`. That was
  corrected. `.venv\Scripts\pytest.exe -q -ra
  tests/test_energy_diagnostics.py tests/test_experiment_runner.py
  tests/test_indepth_comparison.py tests/test_hardest_mode_dynamics.py
  tests/test_tuning_selection.py tests/test_multiseed_output_protection.py`
  then passed; the targeted gradient module passed **11**. A full
  `.venv\Scripts\pytest.exe -ra` first passed **167 in 75.96 s**, then
  passed **178 in 81.77 s, 0 skipped** after the gradient fixtures.
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for 61 source files;
  `.venv\Scripts\ruff.exe format --check
  tests/test_energy_diagnostics.py tests/test_local_gradient_contracts.py`
  passed. `git diff --check` exited 0 with existing Windows LF/CRLF
  conversion notices. A tiny default toy CLI smoke (`--samples 80
  --epochs 2 --hidden-dim 4 --sleep-interval 1 --replay-steps 0`) exited 0
  and printed all three distinct IDs. Its descriptive test accuracies were
  0.938 backprop, 0.000 PC, and 0.938 circadian; no seed, metric, baseline,
  or stopping condition was adjusted in response.
- Skipped tests and experiments: no pytest tests skipped. No CIFAR download,
  larger-data run, real CUDA numerical check, large sweep, or remote CI
  observation. Repository-wide `ruff format --check` was not rerun because
  earlier sessions established unrelated mixed-line-ending/layout failures;
  both new Python files pass formatting. The current gradient fixtures do
  not verify the deeper all-latent PC formulation.
- Experiment artifacts: no new result file. Numeric and derivative evidence
  lives in tests; ADR-0021 is the decision record. Earlier P1.8g JSON
  artifacts and historical figures remain untouched.
- Plan changes: P2.2 was checked after formulas, width effects, report/export
  identifiers, mixed-ID guards, documentation, and the full quality gate.
  P3.2 now explicitly requires a width-dependent adaptive-trigger audit.
  P2.3 was split into completed one-hidden gate P2.3a and unchecked
  deeper/gated gate P2.3b; full P2.3 retains its original acceptance and
  P2.6 retains the unmatched multilayer formulation. No acceptance
  criterion or negative result was weakened.
- Blockers: none for P2.3b. Actual CIFAR data are absent in this checkout,
  and GPU checks require a CUDA host. Existing repository formatting would
  touch unrelated lines.
- Exact next action: derive a two-hidden NumPy circadian float64
  feedforward-prior objective with the adaptive state held fixed, then add
  central-difference and test-only autograd checks for the earlier-layer
  weight/bias partials and compare one actual update. Follow with a Torch
  circadian chemical/reward gate fixture. Keep P2.3/P2.3b unchecked until
  those checks and the full suite pass.

## 2026-09-24 — Deeper derivative boundaries and P2.3 completion

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Continued the intentional
  uncommitted tree without modifying unrelated user files or historical
  benchmark artifacts. Local runtime remains Windows CPU with Torch
  `2.14.0+cpu`, NumPy `2.4.6`, and no CUDA device.
- Completed task IDs: **P2.3b** and full **P2.3**. The prior P2.2/P2.3a
  entry records the first stage. P2.4, P2.6, P1.7, P1.8, and P9.1 remain
  unchecked.
- Implementation and finding: added three deterministic float64 tests to
  `tests/test_local_gradient_contracts.py`. The two-hidden NumPy circadian
  feedforward-prior fixture compares all seven state/parameter partial
  groups with central differences and independent test-only Torch autograd,
  then checks the executed one-step weight deltas. A Torch circadian
  fixture checks chemical-state update, nonunit reward scaling, gated hidden
  weights/bias, and ungated output bias against analytic local partials.
  The negative ordinary two-layer PC fixture captures the actual relaxed
  lower state: its top-down move is nonzero while the fixed-prior diagnostic
  has zero lower-state derivative. This rules out that diagnostic as the
  gradient objective for the executed lower-state move; it does not rule
  out every possible scalar formulation. No training algorithm, metric,
  baseline, seed, or result-selection rule changed. P2.6 retains the
  unmatched deeper formulation.
- Commands and outcomes: `.venv\Scripts\pytest.exe -q -ra
  tests/test_local_gradient_contracts.py` passed **14**. The final
  `.venv\Scripts\pytest.exe -ra` passed **181 in 87.40 s, 0 skipped**.
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **61 source
  files**; `.venv\Scripts\ruff.exe format --check
  tests/test_energy_diagnostics.py tests/test_local_gradient_contracts.py`
  passed. `git diff --check` exited 0 with the existing Windows LF/CRLF
  notices. Documentation and plan edits followed the full code/test gate.
- Skipped tests and experiments: no pytest tests skipped. No CIFAR
  download, larger-data run, CUDA numeric check, large sweep, or remote CI
  observation. Repository-wide `ruff format --check` was not rerun because
  earlier sessions found unrelated mixed-line-ending/layout failures; the
  two new Python files passed their formatter check.
- Experiment artifacts: none newly created. Test fixtures are the numerical
  evidence. Prior P1.8g JSON artifacts and historical figures were left
  untouched.
- Plan changes: P2.3b and P2.3 were checked only after their derivative
  boundaries and the full quality gate passed. `docs/learning-mathematics.md`
  and `docs/modules/core.md` now distinguish completed local checks from
  P2.4 stability and P2.6 deeper-formulation work. P2.3b wording now names
  the exact reported fixed-prior diagnostic disproved by the negative test;
  it avoids an unsupported claim that no alternative scalar objective can
  exist. The original P2.3 derivative criteria and P2.6 deeper-formulation
  work remain intact; no negative result was hidden.
- Blockers: none for P2.4. Actual CIFAR data are absent from this checkout;
  GPU checks require a CUDA host, and remote CI has not been observed.
- Exact next action: inspect the NumPy and Torch latent inference loops and
  their input validation, then add a deterministic small-step relaxation
  fixture for the documented local update contract, convergence or step
  limits, zero/small residuals, and nonfinite handling. Keep P2.4 unchecked
  until implementation and the full suite pass; do not assert arbitrary-step
  global monotonicity.

## 2026-09-25 — Local latent stability and finite-state boundary

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Continued the intentional
  uncommitted tree; unrelated user files, old JSON, and figures were left
  untouched. Local environment is Windows CPU, NumPy `2.4.6`, Torch
  `2.14.0+cpu`, no CUDA device.
- Completed task ID: **P2.4**. P2.5, P2.6, P2.8, P1.7, P1.8, and P9.1
  remain unchecked. P2.4 was marked in progress before the tests and checked
  only after its final quality gate.
- Changed files and behavior: `tests/test_latent_relaxation.py` adds 31
  deterministic cases. `src/core/predictive_coding.py` and
  `src/core/circadian_predictive_coding.py` reject nonfinite input/targets
  and controls, reject invalid step counts, and detect nonfinite prior
  linears, logits, and latent states during relaxation. Circadian also
  checks earlier feedforward prior linears before `tanh` can hide overflow.
  `src/core/resnet50_variants.py` gives the inherited Torch PC/circadian
  heads finite controls/features and a combined post-relaxation prior/state/
  logit check before parameter updates. Valid finite training equations,
  diagnostic values, seeds, and result selection were not changed.
- Numerical evidence: the one-hidden binary and multiclass fixtures use
  `α=0.2` below the conservative fixed-weight bound
  `1/(1+||V||₂²)`. Independent test-only autograd trajectories match actual
  NumPy/Torch one- and 60-step latent states; every tested local-objective
  step decreases and the final gradient norm is below `10⁻⁴` of its
  initial value. A two-latent NumPy PC fixture matches the simultaneous
  fixed-prior recurrence and reduces its update-residual norm below
  `10⁻⁴` without asserting a scalar-energy theorem. Zero output drive
  leaves the state at its prior; `10⁻⁸` drive produces bounded displacement.
  Nonfinite input/rate, zero step count, overflowing prior, and divergent
  latent cases raise before weight assignment. ADR-0022 and
  `docs/learning-mathematics.md` state the narrow stability contract;
  `docs/modules/core.md` points to it.
- Commands and outcomes: initial targeted
  `.venv\Scripts\pytest.exe -q -ra tests/test_latent_relaxation.py` had
  four test-fixture shape failures and eight expected finite-guard failures;
  after the shape fix and guards, the final targeted run passed **31**.
  `.venv\Scripts\pytest.exe -ra` passed **212 in 126.20 s, 0 skipped**.
  `.venv\Scripts\ruff.exe check .` passed; `.venv\Scripts\mypy.exe src
  tests scripts` passed for **62 source files**; `.venv\Scripts\ruff.exe
  format --check tests/test_latent_relaxation.py` passed. A tiny default
  toy CLI run (`--samples 80 --epochs 2 --hidden-dim 4 --sleep-interval 1
  --replay-steps 0`) exited 0 and retained descriptive test accuracies
  0.938 backprop, 0.000 PC, and 0.938 circadian. No response tuning was
  done. `git diff --check` exited 0 with existing Windows LF/CRLF notices.
- Skipped tests and experiments: no pytest tests skipped. No CIFAR download,
  large sweep, GPU numerical run, or remote CI. A source-file
  `ruff format --diff` still shows thousands of pre-existing mixed-line-
  ending/layout changes; the new test file is formatted, and unrelated
  source formatting was not rewritten.
- Experiment artifacts: no new result file. Test fixtures and ADR-0022 are
  the evidence; historical P1.8g artifacts remain untouched.
- Plan changes: checked P2.4 with evidence and moved the current task to
  P2.5. The one-hidden objective descent claim is explicitly restricted to
  fixed weights/prior and a stable small step. The unmatched deeper PC
  objective stays under P2.6; broader shape, label, and post-update
  numerical checks stay under P2.8. Torch finite checks add host/device
  synchronization cost, so P1.8 timing requires target-hardware reruns.
- Blockers: none for P2.5. Actual CIFAR files are absent locally; GPU and
  remote CI evidence require their respective environments.
- Exact next action: inspect one-hidden NumPy and Torch PC/circadian
  parameter initialization and update paths, define a neutral circadian
  configuration with sleep/replay/structure disabled, then add paired
  one-step and repeated-step state/parameter tests under identical inputs
  and initialization. Keep P2.5 unchecked until the matched control and
  full suite pass; do not compare their distinct training diagnostics as
  one objective.

## 2026-09-25 — Named shallow no-circadian parity control

- Repository state: `master` still points to reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Continued the existing
  intentional uncommitted tree. Unrelated user changes, historical figures,
  and earlier benchmark JSON were preserved.
- Completed task ID: **P2.5**. P2.6, P2.7, P1.7, P1.8, and P9.1 remain
  unchecked. No claim was made for a shared deeper NumPy algorithm.
- Inspection and rationale: one-hidden NumPy ordinary PC and circadian PC
  share binary priors, relaxation, local parameter updates, and parameter
  initialization for one seed. The Torch PC/circadian heads share a
  multiclass one-hidden architecture and initialization. A fixture-local
  neutral configuration passed the first parity test, but would not give
  future ablations a named, reproducible control. The plan was amended
  before adding public presets. The same-initialized two-hidden NumPy paths
  diverge in their earlier weight update, as predicted by P2.3; this
  negative boundary remains under P2.6.
- Changed files and behavior: added opt-in
  `CircadianConfig.matched_pc_control()` in
  `src/core/circadian_predictive_coding.py` and
  `CircadianHeadConfig.matched_pc_control()` in
  `src/core/resnet50_variants.py`. These set unit plasticity/reward, zero
  split/prune budgets, and no adaptive sleep/homeostasis; NumPy also has
  zero replay steps and memory. Default configs are unchanged. The new
  `tests/test_neutral_circadian_control.py` checks exact same-seed initial
  tensors and four successive wake parameter/prediction/traffic/diagnostic
  states for each backend. Chemistry rises but plasticity remains one;
  forced sleep changes no width or weight, and NumPy replay memory remains
  empty. The NumPy one-hidden diagnostic IDs remain distinct even when
  the numeric values agree. A two-hidden NumPy negative test records
  earlier-weight divergence. ADR-0023, learning-mathematics, evaluation-
  protocol, and core-module docs state the architecture and scope.
- Commands and outcomes: `.venv\Scripts\pytest.exe -q -ra
  tests/test_neutral_circadian_control.py` passed **3**; the combined
  targeted command with `tests/test_latent_relaxation.py` passed **34**.
  `.venv\Scripts\pytest.exe -ra` passed **215 in 71.16 s, 0 skipped**.
  `.venv\Scripts\ruff.exe check .` passed; `.venv\Scripts\mypy.exe src
  tests scripts` passed for **63 source files**; `.venv\Scripts\ruff.exe
  format --check tests/test_latent_relaxation.py
  tests/test_neutral_circadian_control.py` passed. `git diff --check`
  exited 0 with the existing LF/CRLF notices. No post-gate code changed.
- Skipped tests and experiments: no pytest tests skipped. No CIFAR
  download, large sweep, CUDA parity run, or remote CI observation.
  Repository-wide source formatting was not applied because existing
  mixed-line-ending/layout changes would rewrite unrelated lines.
- Experiment artifacts: no new result file. The parity fixtures and
  ADR-0023 are evidence; older P1.8g outputs were not overwritten.
- Plan changes: P2.5 now requires the named executable presets as well as
  its original exact shallow parity gate. It was checked after presets,
  negative deeper test, docs, and full quality gate. P2.6 remains intact;
  the control does not convert a deeper NumPy comparison into causal
  circadian attribution.
- Blockers: none for inspecting P2.6. Actual CIFAR data are absent in the
  checkout; GPU and remote CI evidence require their environments.
- Exact next action: inspect toy, continual, and figure/report routes for
  deeper PC-versus-circadian attribution. Identify which routes already
  use one hidden layer and which allow unmatched multilayer paths; then
  version a shared deeper formulation or explicitly restrict causal
  attribution to the verified shallow control while preserving legacy
  behavior. Add a regression gate before checking P2.6.

## 2026-09-25 — Versioned NumPy comparison scope

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Preserved the existing
  intentional uncommitted source/docs/tests, unrelated user changes,
  historical figures, and benchmark outputs. Local Windows environment has
  NumPy 2.4.6 and Torch 2.14.0+cpu; no CUDA device.
- Completed task ID: **P2.6** via its shallow-only attribution option.
  P2.6a, P2.7, P1.7, and P1.8 remain unchecked. No new deeper learning
  rule or causal benchmark result was claimed.
- Inspection and decision: toy, in-depth, and continual configs accept
  multiple hidden layers; hardest-mode dynamics defaults to three. NumPy
  ordinary PC relaxes all hidden latents and uses local earlier-weight
  updates; circadian PC relaxes its final adaptive latent and propagates a
  prior gradient through earlier feedforward layers. Their data-role
  protocol IDs did not identify this difference. Even shallow toy and
  continual reports use different model seeds/controls. Chose explicit
  descriptive scope reporting over changing historical numerical rules.
- Changed files and behavior: added `src/app/comparison_scope.py` with
  stable Backprop/PC/circadian algorithm IDs and shallow, deeper-unmatched,
  and unknown-architecture scope IDs; all set
  `causal_attribution_supported=False`. Wired it into
  `src/app/experiment_runner.py`, `src/app/indepth_comparison.py`,
  `src/app/continual_shift_benchmark.py`, and
  `scripts/generate_hardest_mode_dynamics.py`. Text reports and interactive
  payloads carry scope and algorithm IDs; new GIFs show the descriptive
  status and store the full scope in GIF comments. The normalized GIF
  training-series axis says metric, matching the existing per-model
  diagnostic IDs. The `objective_series` JSON key and all data-role
  protocol IDs remain for compatibility. Added
  `tests/test_numpy_comparison_scope.py`, ADR-0024, protocol/math/module
  docs, and README text.
- Numerical preservation check: the same bounded inline Python runs were
  executed before and after the reporting change. Toy config used 80
  samples, two epochs, widths `(4,4)`, seed 13, sleep interval zero; test
  accuracy remained `[0.0625, 0.8125, 0.6875]`, and last training metrics
  remained `[1.0143538970265285, 0.5101876839968554,
  0.5209464833572991]`. Continual used 80 examples per phase, two epochs
  per phase, widths `(4,4)`, seed 13, 50% phase-B training fraction, and
  sleep intervals zero; balanced scores remained `[0.15, 0.675, 0.6]`.
  Dynamics used 40 examples per phase, the same widths/epochs/fraction,
  snapshot interval one, 8-point grid, one latency repeat, and zero sleep
  intervals; final test accuracies remained Backprop `0.8`, PC `0.9`,
  circadian `0.4`, with identical snapshot accuracies. All three runs'
  split hashes matched pre/post; toy train/validation/test SHA-256 hashes
  were `2bb7909c9c02da0beda51d9dc8cf6d18421cbb63c87efea60abd266ff1054bb4`,
  `676ca2ffab0779c93d64e8aa1bc32708babdac1c506a570faf1f53157c9be790`,
  and `f7ff383ff5f569db6b8fb574d5f219a05b294228eac4900f6af64036edd40453`.
  This checks preservation on small fixtures, not all possible runs.
- Commands and outcomes: the initial focused test collected red because
  `src.app.comparison_scope` did not yet exist. After implementation,
  `.venv\Scripts\pytest.exe -q -ra tests/test_numpy_comparison_scope.py
  tests/test_experiment_runner.py tests/test_indepth_comparison.py
  tests/test_continual_shift_benchmark.py
  tests/test_hardest_mode_dynamics.py` passed **33**. After adding the
  algorithm IDs to GIF metadata, `.venv\Scripts\pytest.exe -q -ra
  tests/test_numpy_comparison_scope.py
  tests/test_hardest_mode_dynamics.py` passed **11**.
  `.venv\Scripts\pytest.exe -ra` passed **220 in 72.37 s, 0 skipped**
  after the final code change. `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **65 files**;
  `.venv\Scripts\ruff.exe format --check src/app/comparison_scope.py
  tests/test_numpy_comparison_scope.py` passed. `git diff --check` exited 0
  with existing Windows LF/CRLF notices.
- Skipped tests and experiments: no pytest tests skipped. No CIFAR
  download, GPU parity run, large sweep, remote CI, or repository-wide
  reformat. Existing unrelated source formatting was left intact.
- Experiment artifacts: no new result file; the bounded checks printed
  values only. The new regression fixture and ADR-0024 are retained
  evidence. Historical figures and prior benchmark artifacts are untouched.
- Plan changes: checked P2.6 only after its scope/report gate and full
  quality checks. Added P2.6a as an explicit deferred, unchecked shared
  deeper objective/gradient/neutral-control/fairness task. Updated the
  P2.7 handoff; no P1.7/P1.8 criterion was weakened.
- Blockers: none for P2.7. Actual CIFAR data are absent locally; CUDA
  and remote CI evidence require those environments.
- Exact next action: inspect NumPy binary and Torch multiclass one-hidden
  forward/update paths to define the smallest shared objective, output
  encoding, activation, and float64 fixture. Implement a deterministic
  cross-backend forward and one-step update test with explicit tolerances;
  report incompatible behavior instead of claiming parity for unequal
  tasks. Keep P2.7 unchecked until the matched fixture and full gate pass.

## 2026-09-25 — Bounded cross-backend parity and negative output-step result

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; preserved the existing
  intentional uncommitted tree and unrelated user changes. Local hardware
  is Windows CPU, NumPy 2.4.6, Torch 2.14.0+cpu, no CUDA device.
- Completed task IDs: **P2.7, P2.7a, P2.7b**. P2.6a, P2.8, P1.7, and P1.8
  remain unchecked. No production learning rule or benchmark result changed.
- Inspection and decision: NumPy has a binary sigmoid head; Torch has free
  multiclass softmax columns. With one tanh hidden layer and Torch logits
  `[-z/2,z/2]`, the positive probability, represented BCE/CE, and initial
  latent drive match. At equal scalar learning rate, the two Torch output
  columns change the margin by twice the NumPy scalar output step. This is
  a valid negative full-update result, so rates were not retuned. NumPy
  chooses both sleep indices before mutation; Torch chooses pruning after
  splitting. Tests therefore separate split-only and prune-only cases.
- Changed files: added `tests/test_backend_parity_boundaries.py` with five
  production-path float64 CPU fixtures, and
  `docs/adr/ADR-0025-cross-backend-output-gauge.md`. Updated
  `docs/evaluation-protocols.md`, `docs/learning-mathematics.md`,
  `docs/modules/core.md`, README, and the living plan.
- Numerical fixture and outcome: fixed 2×2 feature/label batch, width 3,
  copied input weights/biases, binary output weight `V`, Torch output
  columns `[-V/2,V/2]`, seed 13, learning rate 0.03, three relaxation steps,
  latent rate 0.2. Ordinary PC and neutral circadian matched initial
  probabilities and supervised loss, post-step hidden weights/biases and
  hidden traffic; circadian chemistry, plasticity, and importance matched.
  Output margin delta was exactly twice the NumPy scalar-output delta
  within `atol=1e-15, rtol=0`, and the difference was nonzero. A separate
  seed-19 NumPy/Torch backprop MLP fixture used zero-momentum SGD: initial
  probability/loss and hidden step matched, and the same margin relation
  held. No later trajectory parity is claimed.
- Structural fixture and outcome: fresh width-4 mapped networks, seed 17,
  chemicals `[0.95,0.85,0.08,0.12]`, distinct importance and weight
  scores, static split/prune thresholds 0.8/0.2, one structural change,
  no noise, immediate NumPy pruning and no replay. Normalized scores and
  selected indices matched: split index 0 and prune index 3 in separate
  runs. Both events matched widths, mapped parameters, chemistry, and
  feedforward probabilities; the split preserved probability. Combined
  split/prune ordering and stochastic noise were explicitly excluded.
- Commands and outcomes: `.venv\Scripts\pytest.exe -q -ra
  tests/test_backend_parity_boundaries.py` passed **5** after the backprop
  fixture and warning fix. The combined targeted command with
  `tests/test_neutral_circadian_control.py`,
  `tests/test_local_gradient_contracts.py`, and
  `tests/test_resnet50_variants.py` passed **33**.
  `.venv\Scripts\pytest.exe -ra` passed **225 in 71.45 s, 0 skipped**
  after the final test code. `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **66 files**;
  `.venv\Scripts\ruff.exe format --check
  tests/test_backend_parity_boundaries.py` passed after formatting.
  An intermediate mypy run found branch-variable type inference errors,
  and an intermediate format check identified one layout edit; both were
  fixed before the final gates. `git diff --check` exited 0 with the
  existing Windows LF/CRLF notices.
- Skipped tests and experiments: no pytest tests skipped. No CIFAR
  download, GPU run, large sweep, remote CI, default float32 parity,
  combined split/prune parity, stochastic split parity, or NumPy replay
  comparison. These limits are stated in ADR-0025.
- Experiment artifacts: no benchmark/result file. Deterministic test
  fixtures and ADR-0025 are retained evidence; historical artifacts were
  untouched.
- Plan changes: split P2.7 into numerical P2.7a and structural P2.7b
  before implementation, then added a matched backprop MLP fixture during
  inspection. Checked the parent and both subtasks only after all
  fixtures, docs, and full quality gate passed. P2.6a and P1.7/P1.8 were
  not weakened.
- Blockers: none for P2.8. Actual CIFAR files are absent locally;
  GPU/remote CI evidence requires their environments.
- Exact next action: audit NumPy and Torch training entry points for
  malformed shapes, empty batches, target ranges, and post-update
  nonfinite results. Start with a deterministic failing case for the
  first uncovered boundary, add the smallest validation guard, and rerun
  the focused and full gates. Keep P2.8 unchecked until its complete
  numerical-boundary acceptance is satisfied.

## 2026-09-25 — NumPy binary training-input boundary

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Preserved all earlier
  intentional/unrelated working-tree changes. Local Windows CPU with
  NumPy 2.4.6 and Torch 2.14.0+cpu; no CUDA device.
- Completed task ID: **P2.8a**. Parent P2.8, P2.8b, P2.8c, P2.6a, P1.7,
  and P1.8 remain unchecked.
- Inspection and negative fixture: NumPy backprop checked only
  `learning_rate <= 0`, so NaN passed. A zero-row batch reached mean/divide
  operations, emitted runtime warnings and a nonfinite metric without an
  exception. Ordinary PC and circadian PC checked finite values but not
  input/target ranks, row counts, feature width, or target interval; bad
  target shapes could broadcast or fail after adaptive state changed.
  `.venv\Scripts\pytest.exe -q -x
  tests/test_numpy_training_inputs.py` first failed on the empty backprop
  case: expected `ValueError` was not raised.
- Changed files and behavior: added pure
  `src/core/training_validation.py` with positive-finite weight-rate and
  real, nonempty `(B,D)`/same-row `(B,1)` binary-batch contracts. It checks
  expected input width, finite values, and targets in `[0,1]` before any
  trainer mutation. `src/core/backprop_mlp.py`,
  `src/core/predictive_coding.py`, and
  `src/core/circadian_predictive_coding.py` call it at training entry;
  existing latent-step/rate guards remain. Added
  `tests/test_numpy_training_inputs.py`, ADR-0026, and core/math/README
  documentation. Invalid inputs raise informative `ValueError`; finite
  soft labels inside `[0,1]` remain accepted. The training arithmetic for
  valid batches was not changed.
- Commands and outcomes: `.venv\Scripts\pytest.exe -q -ra
  tests/test_numpy_training_inputs.py` passed **48** after implementation.
  The combined targeted command with `tests/test_experiment_runner.py`,
  `tests/test_continual_shift_benchmark.py`,
  `tests/test_latent_relaxation.py`, and
  `tests/test_local_gradient_contracts.py` passed **114**.
  `.venv\Scripts\pytest.exe -ra` passed **273 in 68.48 s, 0 skipped**.
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **68 files**;
  `.venv\Scripts\ruff.exe format --check
  src/core/training_validation.py tests/test_numpy_training_inputs.py`
  passed after one layout-only format edit. `git diff --check` exited 0
  with existing Windows LF/CRLF notices.
- Rejection state checks: all three trainers preserve weights, hidden
  traffic and counters on every invalid call. Circadian also preserves
  chemical/importance, nonzero cooldown fixtures, age, epoch count, and
  replay memory. Valid soft-label calls still produce a finite metric and
  change weights.
- Skipped tests and experiments: no pytest tests skipped. No CIFAR
  download, CUDA run, large sweep, remote CI, Torch input-contract change,
  or post-update/saturation experiment.
- Experiment artifacts: no result file. The deterministic regression
  fixture and ADR-0026 are the retained evidence; historical artifacts
  were untouched.
- Plan changes: split P2.8 into NumPy input P2.8a, Torch input P2.8b,
  and post-update/saturation/topology P2.8c before implementation. Checked
  only P2.8a after the focused/full quality gates. Parent P2.8 retains
  its original full acceptance criteria.
- Blockers: none for P2.8b. Actual CIFAR files are absent locally;
  GPU and remote CI evidence require their environments.
- Exact next action: inspect Torch PC and circadian head training entry
  points; write a deterministic failing test for an empty batch,
  malformed class-index shape, or out-of-range label. Add a pre-mutation
  validator and check that rejected calls preserve parameters, traffic,
  and circadian adaptive state. Keep P2.8b, P2.8c, and parent P2.8 open
  until their own acceptance gates pass.

## 2026-09-25 — Torch head training-input boundary

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; preserved existing
  intentional and unrelated working-tree changes. Local Windows CPU has
  Torch 2.14.0+cpu and NumPy 2.4.6; no CUDA device.
- Completed task ID: **P2.8b**. Parent P2.8, P2.8c, P2.6a, P1.7, and
  P1.8 remain unchecked.
- Inspection and negative fixture: the ordinary Torch PC head accepted a
  zero-row feature/target batch without raising. Both head paths relied
  on matrix multiplication and `one_hot` for malformed shapes/labels;
  circadian cooldowns decayed before those late failures. The first
  `.venv\Scripts\pytest.exe -q -x
  tests/test_torch_head_training_inputs.py` failed at the empty ordinary
  head case because `ValueError` was not raised.
- Changed files and behavior: expanded the common private entry check in
  `src/core/resnet50_variants.py` and passed it both features and targets
  from ordinary and circadian `train_step`. It checks nonempty 2D feature
  tensors, configured width, floating dtype matching head weights, model
  device, finite values, same-row 1D `torch.int64` targets on the feature
  device, class indices in `[0,C)`, and existing positive finite step/rate
  controls. Validation runs before circadian cooldown decay and any other
  state change. Invalid inputs raise informative `ValueError`; valid
  multiclass training arithmetic and protocol IDs are unchanged. A valid
  batch still uses one host synchronization for combined finite-feature
  and label-range reductions, plus an extra reduction kernel; P1.8 must
  time that cost on target hardware. Added
  `tests/test_torch_head_training_inputs.py`, ADR-0027, and core/math/README
  documentation.
- Rejection-state evidence: 18 invalid input/rate cases across both heads
  plus one valid multiclass case each passed (**38** tests). Rejected
  calls preserve all weights and biases, traffic, nonzero cooldowns,
  chemistry, importance, age, reward/history counters, and split RNG
  state. Valid cases produce finite diagnostics and update weights.
- Commands and outcomes: `.venv\Scripts\pytest.exe -q -ra
  tests/test_torch_head_training_inputs.py` passed **38**. The focused
  command with `tests/test_latent_relaxation.py`,
  `tests/test_neutral_circadian_control.py`,
  `tests/test_matched_head_benchmark.py`,
  `tests/test_energy_diagnostics.py`, and
  `tests/test_resnet50_variants.py` passed. `.venv\Scripts\pytest.exe
  -ra` passed **311 in 86.15 s, 0 skipped**.
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **69 files**;
  `.venv\Scripts\ruff.exe format --check
  tests/test_torch_head_training_inputs.py` passed after formatting the
  new file. `git diff --check` exited 0 with existing Windows LF/CRLF
  notices.
- Skipped tests and experiments: no pytest tests skipped. No CIFAR
  download, GPU run, large sweep, remote CI, or target-hardware timing.
  P2.8c post-update/saturation/topology cases were not attempted.
- Experiment artifacts: no benchmark/result file; the deterministic
  regression test and ADR-0027 are retained evidence. Historical outputs
  were untouched.
- Plan changes: checked only P2.8b after focused, full, lint, type, and
  format gates. Parent P2.8 and P2.8c retain post-update finite, topology,
  and saturation acceptance; P1.7/P1.8 remain unchanged.
- Blockers: none for P2.8c. Actual CIFAR files are absent locally; GPU
  and remote CI evidence require their environments.
- Exact next action: audit NumPy/Torch update paths for finite inputs and
  controls that produce nonfinite parameters or metrics. Add a failing
  deterministic fixture and a pre-commit finite check or rollback; then
  test extreme finite logits and topology widths after structural events.
  Keep P2.8c and parent P2.8 unchecked until those gates and the full
  suite pass.

## 2026-09-25 — Post-update, saturation, topology, and configuration gates

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. All prior intentional
  and unrelated working-tree changes were preserved; no commit or remote
  mutation was made. Local validation used Windows CPU, NumPy 2.4.6, and
  Torch 2.14.0+cpu.
- Completed task IDs: **P2.8c1, P2.8c2, P2.8c3, P2.8c, P2.8d, P2.8e,
  P2.8**. P3.1 is next. P1.7/P1.8 remain open for target-environment
  evidence, and P2.6a remains deferred.
- Initial failing evidence: a finite `1e200` input and rate committed an
  infinite NumPy backprop hidden weight without raising. A corresponding
  Torch head case did the same. A NumPy PC fixture produced an infinite
  squared-residual diagnostic after parameter mutation. NumPy circadian
  could advance chemistry and gradual-prune state before a later hidden
  candidate overflowed; Torch circadian could decay cooldowns and advance
  chemistry first. The positive `1e300` binary BCE test initially failed
  an over-strict symmetric tail expectation by about `5e-9` because
  `1-1e-8` rounds in binary64. A deliberately corrupt post-split output
  row produced late NumPy/Torch matrix errors. `input_dim=NaN` initially
  raised NumPy's allocation `TypeError`. The exhaustive config fixture
  initially failed the field-name/finite contract against old range checks.
- Changed behavior: ordinary NumPy trainers check stored parameters and
  finite forward intermediates, then stage diagnostic, gradients,
  candidate parameters, and traffic before commit. NumPy circadian stages
  candidate weights/diagnostic/adaptive values and restores provisional
  chemistry, reward, importance, and active gradual-prune arrays/topology
  after numerical rejection. Both Torch PC heads check all candidate
  weights, traffic, and diagnostic before assignment; circadian restores
  provisional adaptive state and decays cooldowns only after the check.
  Existing valid update arithmetic, output metric IDs, and first-layer
  aliases remain. New training-entry topology checks reject misaligned
  parameter or per-neuron widths before adaptive mutation. Both circadian
  config constructors reject NaN/±inf in every numeric field by name;
  model constructors reject nonfinite, fractional, boolean, zero, and
  negative dimensions before allocation while accepting NumPy integers.
- Saturation result: all three NumPy binary trainers retain finite clipped
  BCE near 18.42 for wrong `±1e300` logits; the positive tail has the
  documented floating-point rounding difference. The matched-head
  multiclass held-out CE path reports finite `2e300` for logits
  `[1e300,-1e300,0]`; Torch PC's distinct squared training diagnostic is
  `1/3`. This is a comparison boundary, not a new or selected metric.
  Accepted one-neuron split and prune events still train at aligned new
  widths in both backends. No arbitrary-step stability claim is made.
- Torch timing boundary: the candidate reductions feed the existing
  diagnostic `.item()` host read, so they add kernels without adding a
  valid-path post-update host synchronization. Input and relaxed-state
  checks have their own reads; reward modulation may add another. These
  operations belong in P1.8 target-hardware training time. Active NumPy
  gradual pruning now copies its affected arrays for numeric rollback;
  that work likewise belongs in measured wake time.
- Commands and outcomes: the initial `pytest -q -x` runs for
  `tests/test_post_update_finite_numpy.py`,
  `tests/test_post_update_finite_torch.py`,
  `tests/test_post_update_finite_circadian.py`,
  `tests/test_circadian_finite_config.py`, and
  `tests/test_constructor_dimensions.py` each exposed the gaps above.
  Focused regression/latent/input suites passed after fixes (ordinary
  NumPy **101**, circadian NumPy **54**, Torch **78**, and the later
  saturation/config/dimension suites). Sequential full runs of
  `.venv\Scripts\pytest.exe -ra` passed **319**, **331**, **345**, **353**,
  and finally **417 in 73.56 s, 0 skipped**. Final
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **76 files**;
  `.venv\Scripts\ruff.exe format --check` on the eight new/previously
  formatted files passed; `git diff --check` exited 0 with existing
  Windows LF/CRLF notices. Existing source files were not wholesale
  reformatted because they already differ from Ruff format outside this
  increment.
- Experiment artifacts: no benchmark result file or sweep. The retained
  deterministic fixtures are
  `tests/test_post_update_finite_numpy.py`,
  `tests/test_post_update_finite_torch.py`,
  `tests/test_post_update_finite_circadian.py`,
  `tests/test_saturation_topology_boundaries.py`,
  `tests/test_circadian_finite_config.py`, and
  `tests/test_constructor_dimensions.py`; ADR-0028–0032 record decisions.
  Historical outputs and baseline metrics were untouched.
- Plan changes: split P2.8c into ordinary NumPy commits, adaptive/Torch
  commits, and saturation/topology gates. After satisfying those, an audit
  found the original finite-configuration criterion still open, so P2.8d
  and P2.8e retained config-field and constructor-dimension work.
  Checked each task only after its own regression and full quality gate;
  P2.8 parent was checked only after all five subgates passed. No
  acceptance criterion was weakened.
- Skipped tests and blockers: no pytest test skipped. No CUDA run, actual
  CIFAR download, larger-data trial, remote CI observation, or large
  hyperparameter sweep. Actual CIFAR files and CUDA hardware are absent
  locally, leaving P1.7/P1.8 target-environment evidence open. No local
  blocker exists for P3.1.
- Exact next action: inspect NumPy and Torch `sleep_event` early returns
  when split/prune budgets are zero; write a deterministic failing case
  showing whether chemical reset, replay, or homeostasis is suppressed.
  Introduce independent sleep-component switches and a distinct fully
  disabled mode under P3.1, preserve default/legacy behavior explicitly,
  and run focused then full quality gates before checking P3.1.

## 2026-09-25 — P3.1 independent sleep components

- Repository state: `master` at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. The working tree retains
  the earlier Phase 0–2 work and the new P3.1 files; no commit, reset, or
  unrelated-file cleanup was performed. Before implementation, read
  `AGENTS.md`, the full `DEVELOPMENT_PLAN.md`, this log, and inspected the
  actual NumPy/Torch sleep paths and experiment consumers.
- Completed task IDs: **P3.1a, P3.1b, P3.1c, P3.1**. NumPy now has explicit
  `legacy` (default), `components`, and `disabled` sleep modes. The component
  route independently gates chemical reset, replay, homeostasis, split,
  and prune and runs nonstructural consolidation at zero split/prune budget.
  Torch has the corresponding four available switches; it has no replay
  implementation. Disabled returns without changing the model or sleep
  clock even when forced. Both results expose backward-compatible
  `performed=False/True`; it is true after an executed no-topology event.
  Legacy budget-gated behavior and the existing matched-PC preset stay
  intact. Changing the matched preset to disabled initially broke two
  structural parity fixtures derived from it; restoring its legacy mode
  resolved both failures without reducing their assertions.
- Consumer and output evidence: toy and continual runners count performed
  component events including no-topology consolidation but retain legacy
  topology-change counts. Toy result/report, continual report, and vision
  report name the sleep mode alongside unchanged protocol IDs. The vision
  benchmark builder and CLI pass mode/switches to the head; invalid mode,
  nonboolean switch, or a false component switch in legacy mode is rejected before
  benchmark setup. A no-topology vision event still incurs pre/post guard
  evaluation when rollback checking is on. The relevant tests pass.
- Commands and outcomes: initial `pytest -q -x` runs on
  `tests/test_numpy_sleep_components.py` and
  `tests/test_torch_sleep_components.py` failed on the missing mode config;
  the event-accounting test initially exposed a zero toy count for a
  consolidation-only event. After implementation,
  `.venv\Scripts\pytest.exe -q -ra tests/test_numpy_sleep_components.py
  tests/test_torch_sleep_components.py tests/test_sleep_event_accounting.py`
  passed **32**. `.venv\Scripts\pytest.exe -ra` passed **449 in 87.24 s,
  0 skipped**. `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **79 files**;
  `.venv\Scripts\ruff.exe format --check` on the three new tests passed;
  `git diff --check` exited 0 with existing Windows LF/CRLF notices.
  Existing source files were not wholesale reformatted.
- Experiment artifacts: no sweep, dataset download, benchmark ranking, or
  changed metric. The bounded deterministic artifacts are
  `tests/test_numpy_sleep_components.py`,
  `tests/test_torch_sleep_components.py`,
  `tests/test_sleep_event_accounting.py`, and
  `docs/adr/ADR-0033-independent-sleep-components.md`. README, core module,
  feature inventory, and evaluation protocol docs record the route and
  limits. Negative and legacy results were preserved.
- Plan changes: split broad P3.1 into NumPy, Torch, and consumer/report
  increments because the two backends have different replay capability
  and existing reports use topology-change counts. Checked each subtask
  and parent only after the focused and full gates passed. P3.2 trigger
  clocks, P3.3–P3.10 state transactions and telemetry, and the Phase 3 exit
  gate remain unchecked; no acceptance criterion was weakened.
- Skipped tests and blockers: pytest skipped none. Actual CIFAR files and
  CUDA hardware remain absent, so P1.7/P1.8 target-environment and
  larger-data evidence remain open. P2.6a is still deferred. No local
  blocker exists for P3.2.
- Exact next action: inventory every NumPy/Torch `sleep_event` caller and
  `_resolve_sleep_budgets` clock input, then write a deterministic trigger
  matrix covering periodic attempts, adaptive triggers, forced calls,
  disabled mode, warmup, and a no-topology performed event. Specify which
  clocks count wake batches/examples/epochs, replay updates, and sleep
  events before changing trigger semantics; retain a legacy route where
  clock changes would alter historical results.

## 2026-09-25 — P3.2a runner sleep attempts

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; the shared Phase 0–3
  working tree and unrelated edits were preserved.
- Completed task ID: **P3.2a**. `src/app/sleep_schedule.py` now makes a
  pure decision from completed runner epochs, interval, adaptive readiness,
  periodic force, and mode. An interval means **attempt**, not guaranteed
  execution; periodic force bypasses adaptive readiness, while an unforced
  attempt still requires it. Disabled mode schedules no call or vision
  guard attempt. Toy and continual component runs can now attempt adaptive
  sleep between intervals; their legacy route stays interval-only. Both
  vision tracks use the same decision. The fixed-width capacity route
  rejects disabled or structurally switched-off sleep before data load.
- Commands and outcomes: `pytest -q -ra tests/test_sleep_event_accounting.py`
  initially failed the two component-mode, interval-zero adaptive cases.
  `pytest -q -ra tests/test_matched_head_capacity.py::test_capacity_route_rejects_invalid_control_before_loading_data`
  initially failed two new cases because the invalid runs reached data
  loading. After fixes, focused schedule/accounting/capacity tests passed
  **41**; the disabled matched-head integration case passed separately.
  `.venv\Scripts\pytest.exe -ra` passed **469 in 75.06 s, 0 skipped**.
  `.venv\Scripts\ruff.exe check .` passed; `.venv\Scripts\mypy.exe src
  tests scripts` passed for **81 files**; new-file Ruff format check and
  `git diff --check` passed (existing Windows LF/CRLF notices only).
- Experiment artifacts: deterministic `tests/test_sleep_schedule.py`,
  expanded `tests/test_sleep_event_accounting.py`, matched-head and
  capacity-route fixtures, and ADR-0034. No dataset download, large
  sweep, candidate selection, or change to a performance metric.
- Plan changes: split P3.2 into scheduling (a), typed clocks (b), and
  width-sensitive adaptive history (c). This follows the distinct caller,
  core-counter, and metric-history boundaries found in the audit. P3.2
  parent and b/c stay unchecked; the original acceptance remains intact.
- Skipped tests and blockers: pytest skipped none. Actual CIFAR/CUDA and
  larger-data evidence remain open under P1.7/P1.8, but do not block
  local P3.2 work.
- Exact next action: add a typed completed-epoch progress input and
  distinguish successful wake batches/examples, NumPy replay updates,
  performed sleep events, and runner attempts. Test warmup and
  min-between-sleep clocks on both backends while preserving legacy
  `current_step`/`total_steps` behavior and snapshot restoration.

## 2026-09-25 — P3.2b typed sleep clocks

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; all earlier local
  work remains in the shared tree. No commit, reset, dataset download,
  or unrelated-file cleanup occurred.
- Completed task ID: **P3.2b**. `SleepEpochProgress` validates completed
  runner epochs and the run cap, and all four app sleep routes pass it to
  core. Existing direct `current_step`/`total_steps` calls retain their
  numeric behavior; passing both forms raises early. Core
  `get_sleep_clocks()` separates successful wake batches/examples, wake
  batches since an executed sleep, NumPy replay updates, and performed
  sleep events. Runner attempts remain the P3.2a scheduling decision.
  NumPy replay does not advance wake counts; Torch reports no replay.
  Torch `snapshot_state`/`restore_state` now include the new counters so
  guard rollback restores them. Full NumPy state snapshots remain P3.3.
- Commands and outcomes: initial `pytest -q -x tests/test_sleep_clocks.py`
  failed at collection because the typed clock module did not exist.
  After implementation, the first focused run found four test fixtures
  with initial width 3 below each backend's default minimum width;
  setting the fixture minimum to 3 fixed setup without changing model
  validation. `tests/test_sleep_clocks.py` then passed **13** cases, and
  focused schedule/accounting/component tests passed. Final
  `.venv\Scripts\pytest.exe -ra` passed **482 in 67.04 s, 0 skipped**.
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **83 files**;
  `ruff format --check` on the two new files and `git diff --check`
  passed (existing Windows LF/CRLF notices only).
- Experiment artifacts: `src/core/sleep_clocks.py`,
  `tests/test_sleep_clocks.py`, and ADR-0035. No sweep, benchmark ranking,
  candidate/seed selection, or change to training/evaluation metrics.
  The existing diagnostic and historical protocol IDs are unchanged.
- Plan changes: P3.2b is complete for typed progress and observable core
  counters. Its new state is covered by Torch in-memory snapshot/restore;
  P3.3 still requires full NumPy and checkpoint state. P3.2 parent stays
  unchecked for the width-sensitive adaptive-history audit in P3.2c.
- Skipped tests and blockers: pytest skipped none. Actual CIFAR/CUDA and
  larger-data evidence remain open under P1.7/P1.8 but do not block the
  local history audit.
- Exact next action: build a deterministic split/prune fixture showing
  whether a width change contaminates the hidden-width-normalized
  diagnostic plateau window or adaptive budget. Define and test a
  component-mode history boundary while preserving legacy history and
  metric IDs; then run the full gate before checking P3.2c/parent.

## 2026-09-25 — P3.2c width-sensitive history and P3.2 closeout

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; all prior local and
  unrelated user changes remain. No commit, reset, external write, or
  dataset download occurred.
- Completed task IDs: **P3.2c, P3.2**. A fixed output prediction and
  summed hidden residual produce a 0.002 diagnostic difference solely
  from changing hidden width four to five. Component-mode NumPy/Torch
  sleep now clears adaptive history after an actual width change; NumPy
  also clears it when gradual pruning finalizes on a successful wake
  update or external proposals change width. An unchanged-width event
  retains history. When component-mode adaptive history is incomplete,
  budget scaling uses the configured minimum, including at startup;
  adaptive triggering waits for a full current-width window. Legacy
  preserves its historical mixed-width history and 1.0 fallback.
  Reported training diagnostics and metric IDs are unchanged; no
  baseline, seed, or metric was tuned to favor circadian results.
- Commands and outcomes: first
  `.venv\Scripts\pytest.exe -q -ra tests/test_sleep_history_boundaries.py`
  failed component split and gradual-prune cases on retained old
  history; an added external-proposal case also failed before the
  boundary was applied. After implementation, eight focused history
  cases passed, as did 65 broader targeted sleep/core cases. Final
  `.venv\Scripts\pytest.exe -ra` passed **490 in 73.32 s, 0 skipped**.
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **84 files**;
  the new-file Ruff format check and `git diff --check` passed (existing
  Windows LF/CRLF notices only).
- Experiment artifacts: deterministic
  `tests/test_sleep_history_boundaries.py` and ADR-0036; no sweep,
  benchmark ranking, selection, or artifact file with performance claims.
  Core/README/evaluation/math/inventory docs identify the opt-in route,
  legacy difference, and width-only diagnostic result.
- Plan changes: P3.2c and parent P3.2 are checked after all original
  trigger, clock, and width-sensitive criteria passed. P3.3–P3.10 and
  the Phase 3 exit gate remain unchecked. Nonstructural replay and
  homeostasis can also alter diagnostic values without width change;
  their side-effect policy remains P4.3, and rejected-event retry policy
  remains P3.8. This limitation is explicit and does not change metrics.
- Skipped tests and blockers: pytest skipped none. Actual CIFAR/CUDA and
  larger-data evidence remain open under P1.7/P1.8. No local blocker
  exists for P3.3.
- Exact next action: inventory every mutable NumPy circadian field and
  Torch head/classifier field, including P3.2 counters and adaptive
  history; add a failing deep-copy/restore fixture covering weights,
  topology, chemistry, replay snapshots/priorities, RNG, and clocks.
  Implement full-state snapshots in small backend increments and keep
  rollback/checkpoint acceptance under P3.7/P3.9.

## 2026-09-25 — P3.3a NumPy in-memory full-state snapshot

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; existing local and
  unrelated user changes were preserved. No commit, reset, external write,
  dataset download, or benchmark sweep occurred.
- Completed task ID: **P3.3a**. `CircadianNetworkSnapshot` stores a
  versioned, configuration-bound deep copy of all NumPy core-owned state.
  Restore checks compatibility and topology on a staged copy before
  replacing live state. The tests cover pre-hidden and adaptive tensors,
  replay batch contents/priorities, counters, RNG, gradual prune metadata,
  deep-copy isolation in both directions, changed-width restoration, the
  next noisy split, and unchanged state after rejected restores.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_numpy_full_snapshot.py`
  failed on the missing `snapshot_state` API as expected. Focused snapshot,
  clock, and NumPy sleep-component tests passed **26**. Final
  `.venv\Scripts\pytest.exe -ra` passed **493 in 76.39 s, 0 skipped**;
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **85 files**;
  `.venv\Scripts\ruff.exe format --check tests/test_numpy_full_snapshot.py`
  and `git diff --check` passed (existing Windows LF/CRLF notices only).
- Experiment artifacts: deterministic `tests/test_numpy_full_snapshot.py`
  and ADR-0037; no performance experiment or ranking artifact. README and
  core module documentation describe the in-memory boundary.
- Plan changes: split original P3.3 into NumPy core (a), Torch head (b),
  and whole-classifier/training-state (c) gates. The original fields,
  optimizer/scheduler/scaler condition, and deep-copy criterion remain in
  the unchecked parent. This split follows the distinct state owners and
  existing head-only guard cost found in code inspection.
- Skipped tests and blockers: pytest skipped none. Actual CIFAR/CUDA and
  larger-data evidence remain open under P1.7/P1.8, without blocking local
  snapshot work. Atomic guard rollback and durable checkpoint/resume remain
  separate P3.7/P3.9 work.
- Exact next action: add a failing Torch head snapshot fixture for detached
  tensors and list state, format/device/config/topology rejection before
  mutation, and next noisy split after restoring its model-owned generator;
  then implement P3.3b without changing head-only guard or hash call sites.

## 2026-09-25 — P3.3b Torch head in-memory full-state snapshot

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; all prior local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task ID: **P3.3b**. The existing flat Torch head snapshot now
  includes the copied model-owned split-generator state plus a version,
  static dimensions, device, and configuration. Restore stages all tensor,
  counter, list, reward, topology, and generator values before updating
  live state. Changed-width split and prune restore, snapshot mutation
  after restore, and the next noisy split are verified. The ResNet
  classifier's guard `snapshot_state()` remains head-only; trained-state
  hash call sites still work with the expanded flat dictionary.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_torch_full_snapshot.py`
  failed on missing `split_generator_state` as expected. Focused head,
  sleep-component, and clock tests passed **42**. Matched-head and ResNet
  benchmark integration tests passed. Final `.venv\Scripts\pytest.exe -ra`
  passed **501 in 119.35 s, 0 skipped**;
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **86 files**;
  `.venv\Scripts\ruff.exe format --check tests/test_torch_full_snapshot.py`
  and `git diff --check` passed (existing Windows LF/CRLF notices only).
- Experiment artifacts: deterministic `tests/test_torch_full_snapshot.py`
  and ADR-0038. No comparative experiment, ranking, or metric change.
  README, core module documentation, and feature inventory describe the
  boundary.
- Plan changes: P3.3b is checked after all acceptance criteria passed;
  P3.3c and parent remain unchecked. Code inspection of
  `_train_circadian` and `CircadianPredictiveCodingResNet50Classifier`
  found no circadian optimizer, scheduler, or scaler. Its direct head
  updates leave backbone parameters unchanged, while backbone buffers and
  submodule modes still belong in a separate full-classifier snapshot.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 still need
  actual CIFAR/CUDA and larger-data evidence. P3.7/P3.9 still own atomic
  guard acceptance and durable checkpoint/resume.
- Exact next action: write a synthetic-backbone test for a detached
  full-classifier snapshot restoring parameters, buffers, submodule modes,
  and head continuation; keep the existing head-only guard API unchanged.

## 2026-09-25 — P3.3c whole-classifier state and P3.3 closeout

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; all prior local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task IDs: **P3.3c, P3.3**. The classifier now exposes a
  separate versioned `snapshot_full_state()`/`restore_full_state()` API.
  It copies backbone parameters/buffers, module structure and train/eval
  modes, parameter gradient flags, and the full adaptive head. Restore
  checks compatibility and stages a detached head/backbone copy before
  applying state. The existing `snapshot_state()`/`restore_state()` guard
  remains head-only. The circadian image and fixed-feature training routes
  update their heads directly; neither owns an optimizer, scheduler, or
  scaler. External data-loader/process RNG and app counters are outside
  this in-memory model boundary and remain P3.9 resume work.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_torch_classifier_full_snapshot.py`
  failed on missing `snapshot_full_state` as expected. The focused
  classifier/head/variant cases then passed **24**. Final
  `.venv\Scripts\pytest.exe -ra` passed **506 in 79.94 s, 0 skipped**.
  A bounded real CPU ResNet-50 Python smoke instantiated random weights,
  changed `conv1.weight` and `bn1.running_mean`, restored both, and
  reported **318** state tensors; exit 0. `.venv\Scripts\ruff.exe check .`
  passed; `.venv\Scripts\mypy.exe src tests scripts` passed for **87
  files**; `.venv\Scripts\ruff.exe format --check
  tests/test_torch_classifier_full_snapshot.py` and `git diff --check`
  passed (existing Windows LF/CRLF notices only). The new test file was
  formatted once after the initial format check requested it.
- Experiment artifacts: deterministic
  `tests/test_torch_classifier_full_snapshot.py` and ADR-0039. No
  performance result, ranking, or large experiment. README, core module
  documentation, and feature inventory reflect the two Torch APIs.
- Plan changes: P3.3c and parent P3.3 are checked after NumPy, Torch head,
  whole-classifier, and full quality gates passed. The original state
  fields and conditional optimizer criterion were retained; the latter
  is not applicable to current circadian training. P3.7/P3.9 still own
  atomic guard acceptance and file-backed checkpoint/resume, including
  caller-owned loader/RNG and app progress.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 still need
  actual CIFAR/CUDA and larger-data evidence. No local P3.4 blocker.
- Exact next action: inspect NumPy and Torch split/prune selectors and
  mutation order, then add failing focused cases for proposal caps,
  minimum/maximum width, cooldown/age, overlap, and bad indices. Split
  P3.4 by backend or invariant if needed before implementing validation;
  retain stable neuron identity/lineage acceptance.

## 2026-09-25 — P3.4a NumPy external proposal preflight

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; existing local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task ID: **P3.4a**. Direct NumPy external proposals and
  policy requests that reach structural selection now validate typed
  hidden-layer requests, unique valid indices, split/prune and fraction
  budgets, intermediate max and pending-prune-aware min widths, and
  cooldown/age/mark eligibility before tensor or local RNG mutation.
  Explicit prune requests take precedence over an overlapping split
  source; growth chooses the next eligible source. Valid policy behavior
  and split-then-prune tensor order are preserved. An event with zero
  structural budgets still skips policy invocation, preserving its prior
  no-structure behavior.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_numpy_proposal_preflight.py`
  failed because invalid index 99 was silently ignored. Focused
  preflight, existing circadian policy, and width-history tests passed
  **32**. The expanded preflight file passed **13** cases. Final
  `.venv\Scripts\pytest.exe -ra` passed **519 in 74.61 s, 0 skipped**;
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **88 files**;
  `.venv\Scripts\ruff.exe format --check
  tests/test_numpy_proposal_preflight.py` and `git diff --check` passed
  (existing Windows LF/CRLF notices only). The new test file was
  formatted once after its initial format check.
- Experiment artifacts: deterministic
  `tests/test_numpy_proposal_preflight.py` and ADR-0040. No performance
  result, seed selection, or metric change. README/core docs and feature
  inventory describe the boundary.
- Plan changes: P3.4 was split into NumPy external/policy preflight (a),
  built-in NumPy/Torch combined proposal validation (b), and stable
  identity/lineage (c). The audit found distinct gaps: external requests
  silently ignored or overran limits, while Torch selects prune after
  split and NumPy selects both on original indices. P3.4a is checked;
  P3.4b/c and parent stay unchecked with their original acceptance.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 still need
  actual CIFAR/CUDA and larger-data evidence. No local P3.4b blocker.
- Exact next action: write a deterministic built-in overlap fixture on
  each backend, including max-width and min-width edges. Inspect Torch's
  post-split prune candidates versus NumPy's pre-split indices, define an
  explicit per-backend proposal/identity rule, and stage the combined
  proposal before any tensor or local RNG mutation. P3.4c still owns
  persistent neuron IDs and parent lineage.

## 2026-09-25 — P3.4b1 built-in NumPy structural preflight

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; prior local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task ID: **P3.4b1**. Built-in NumPy sleep now selects prune
  candidates first on the original width, then excludes them from split
  sources. The complete original-width proposal is checked before any
  split noise or tensor mutation for index type/range/uniqueness,
  configured and resolved caps/fraction, max/min width, thresholds,
  cooldowns, age, and pending gradual-prune marks. Pending marks reserve
  minimum-width capacity. Ordinary disjoint selection keeps its prior
  indices; equal-threshold overlap now gives prune priority.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_numpy_builtin_proposal_preflight.py`
  failed because both selected index 3. Focused built-in/external policy,
  circadian, and NumPy sleep-component tests passed **44** after the first
  fix; the expanded built-in file passed **14** cases. Final
  `.venv\Scripts\pytest.exe -ra` passed **533 in 70.27 s, 0 skipped**;
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **89 files**;
  `.venv\Scripts\ruff.exe format --check
  tests/test_numpy_builtin_proposal_preflight.py` and `git diff --check`
  passed (existing Windows LF/CRLF notices only).
- Experiment artifacts: deterministic
  `tests/test_numpy_builtin_proposal_preflight.py` and ADR-0041. No
  performance experiment, ranking, seed selection, or metric change.
  README and core module documentation record overlap and pending-prune
  behavior.
- Plan changes: split P3.4b into b1 NumPy original-width selection and
  b2 Torch detached post-split selection. NumPy overlap and pending
  capacity were invalid cases; Torch's existing post-split rule remains
  an explicit acceptance criterion in b2. P3.4b parent and P3.4c lineage
  remain unchecked.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 actual
  CIFAR/CUDA and larger-data evidence remain open. No local b2 blocker.
- Exact next action: inject an invalid Torch post-split prune index and
  verify the original head and split-generator state remain unchanged.
  Then stage a detached head from its P3.3b snapshot, simulate noisy split
  and post-split prune, validate both stages and final bounds before live
  mutation, and test overlap/child/min/max/cooldown/age continuation.

## 2026-09-25 — P3.4b2 Torch preflight and P3.4b closeout

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; prior local and
  unrelated user changes remain. No commit, reset, dataset download, or
  ranking sweep.
- Completed task IDs: **P3.4b2, P3.4b**. Torch sleep validates original
  split indices, then simulates the noisy split and post-split prune on
  cloned head tensors and a cloned model-owned generator. It validates
  resulting indices, budgets/fraction, widths, thresholds, age and
  cooldown, and candidate topology before live mutation. Parent or new
  child may still be pruned after split; that historical rule is explicit.
  One-action routes validate directly without cloning the candidate.
  NumPy P3.4b1 and Torch P3.4b2 together close the built-in preflight
  task. Stable neuron identity/lineage remains P3.4c.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_torch_builtin_proposal_preflight.py`
  failed with `IndexError` after an invalid prune index reached an already
  mutated live head. Focused Torch preflight/snapshot/component/variant
  tests passed **43** after the first implementation; the expanded
  preflight file passed **18** cases. Final
  `.venv\Scripts\pytest.exe -ra` passed **551 in 77.83 s, 0 skipped**;
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **90 files**;
  `.venv\Scripts\ruff.exe format --check
  tests/test_torch_builtin_proposal_preflight.py` and `git diff --check`
  passed (existing Windows LF/CRLF notices only). The new test file was
  formatted once after an initial format check.
- Bounded cost observation: a local Torch **2.14.0+cpu** fixture with
  2048 input features, 256 hidden units, 10 classes, one split and one
  prune, three warmups, and 20 planning calls observed **1.663 ms
  median** and **1.889 ms 95th sample**. The 12 cloned head tensors
  totaled **2,116,648 bytes**; the live split generator did not advance.
  This is planning cost on one CPU, not end-to-end or CUDA evidence.
- Experiment artifacts: deterministic
  `tests/test_torch_builtin_proposal_preflight.py` and ADR-0042; no
  comparative result, metric change, or performance ranking. README,
  core docs, and feature inventory record the backend behavior.
- Plan changes: P3.4b2 and parent P3.4b are checked after the cross-backend
  full gate. Torch keeps post-split prune order; NumPy keeps original-width
  selection with explicit prune priority. P3.4 parent stays unchecked for
  P3.4c stable IDs/lineage, and P3.7/P3.9 retain atomic rollback/resume.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 still need
  actual CIFAR/CUDA and larger-data evidence. No local P3.4c blocker.
- Exact next action: inventory every adaptive-width mutation and snapshot
  field on NumPy/Torch, write failing repeated split/prune/restore fixtures
  for persistent unique neuron IDs and parent lineage after index shifts,
  then add aligned model-owned ID state and Torch snapshot/restore fields.

## 2026-09-25 — P3.4c1 NumPy neuron lineage

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; prior local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task ID: **P3.4c1**. NumPy's adaptive layer now owns active
  int64 neuron IDs, birth-parent IDs, and a monotonic next ID. Children
  receive new IDs on every split; immediate and finalized gradual pruning
  mask lineage with the other adaptive tensors. A surviving child keeps
  its parent's ID after parent removal. External proposals and built-in
  sleep share these mutation paths. The read-only `NeuronLineageSnapshot`
  exposes active IDs and optional parents. Version-2 full-state snapshots
  preserve lineage, and topology validation rejects malformed IDs.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_numpy_neuron_lineage.py`
  failed on the missing API. Focused lineage/snapshot/proposal tests passed
  **33** after implementation; the final lineage file passed **3** cases.
  `.venv\Scripts\pytest.exe -ra` passed **554 in 78.50 s, 0 skipped**;
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **91 files**;
  `.venv\Scripts\ruff.exe format --check tests/test_numpy_neuron_lineage.py`
  and `git diff --check` passed (existing Windows LF/CRLF notices only).
- Experiment artifacts: deterministic `tests/test_numpy_neuron_lineage.py`
  and ADR-0043. No performance experiment, ranking, seed selection, or
  metric change. README and core module documentation describe the rule.
- Plan changes: P3.4c1 is complete after its full gate. Torch P3.4c2 is
  in progress; P3.4c and P3.4 stay unchecked until both backends pass.
  Removed-unit history remains P3.6 telemetry, and atomic rollback and
  durable resume remain P3.7/P3.9.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 still need
  actual CIFAR/CUDA and larger-data evidence. No local P3.4c2 blocker.
- Exact next action: write failing Torch lineage fixtures for repeated
  split/prune, parent and child post-split removal, rejection without ID
  allocation, and restore continuation; add aligned ID tensors/counter to
  detached planning and versioned head snapshots, then run full gates.

## 2026-09-25 — P3.4c2 Torch neuron lineage

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; prior local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task ID: **P3.4c2**. The circadian Torch head now owns two
  aligned int64 tensors for active neuron IDs and birth-parent IDs, plus
  a monotonic next ID. Split allocates children with source-ID parent
  references; prune masks IDs with other adaptive tensors. Detached
  post-split proposal planning clones lineage tensors and copies the
  counter by value. Rejected planning leaves live IDs and RNG untouched.
  A shared immutable `get_neuron_lineage()` result reports active IDs.
  Head snapshot format 2 includes lineage, restore validates it, and the
  existing trained-state hash includes the new snapshot fields.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_torch_neuron_lineage.py`
  failed on the missing API. The final new file passed **9** cases;
  focused Torch preflight/variant/lineage tests passed **38**. Final
  `.venv\Scripts\pytest.exe -ra` passed **563 in 79.13 s, 0 skipped**;
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **92 files**;
  `.venv\Scripts\ruff.exe format --check tests/test_torch_neuron_lineage.py`
  and `git diff --check` passed (existing Windows LF/CRLF notices only).
  A direct format check of the existing CRLF source file was noisy because
  Ruff proposed line-ending changes throughout it; the new test file was
  clean, and no broad source reformat was applied.
- Experiment artifacts: deterministic `tests/test_torch_neuron_lineage.py`
  and ADR-0044. No performance experiment, ranking, seed selection, or
  metric change. README, core docs, and feature inventory describe the
  stable-ID rule. Existing sleep-result indices remain positional.
- Plan changes: P3.4c2 is complete. Added P3.4c3 because the P3.4c
  changed-width telemetry criterion is not met by current positional
  `SleepEventResult` alone: a removed neuron's ID is unavailable from a
  post-event active snapshot. P3.4c/P3.4 remain unchecked until read-only
  pre/post event lineage is tested. P3.6 retains explicit proposal versus
  scheduling versus actual removal status; P3.10 retains richer logs.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 still need
  actual CIFAR/CUDA and larger-data evidence. No local P3.4c3 blocker.
- Exact next action: add failing NumPy/Torch sleep-result tests for
  immutable pre/post IDs around width-changing, net-zero-width, gradual
  schedule, and skipped events; add optional lineage fields without
  changing positional indices or learning metrics, then run full gates.

## 2026-09-25 — P3.4c3 event lineage and P3.4 closeout

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; prior local and
  unrelated user changes remain. No commit, reset, dataset download,
  performance ranking, or sweep.
- Completed task IDs: **P3.4c3, P3.4c, P3.4**. Both backends now place
  immutable `lineage_before` and `lineage_after` on executed sleep results,
  preserving removed-unit identity across index shifts and net-zero-width
  split/prune events. Skipped events keep optional fields empty; existing
  positional indices and learning metrics are unchanged. A gradual prune
  request that is only scheduled leaves active IDs unchanged. Together
  with P3.4a/b and P3.4c1/c2, this closes proposal validation and
  model-owned lineage. P3.6 still owns explicit proposed/scheduled/removed
  status, and P3.10 richer trigger/guard/duration telemetry.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_sleep_event_lineage.py`
  failed on the missing event field. New file passed **6** cases after
  implementation; focused sleep-accounting and both-backend lineage
  tests passed **35**. An initial mypy run found 9 optional-field accesses
  in the new tests; explicit presence assertions resolved them. Final
  `.venv\Scripts\pytest.exe -ra` passed **569 in 89.19 s, 0 skipped**;
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **93 files**;
  `.venv\Scripts\ruff.exe format --check tests/test_sleep_event_lineage.py`
  and `git diff --check` passed (existing Windows LF/CRLF notices only).
- Experiment artifacts: deterministic `tests/test_sleep_event_lineage.py`
  and ADR-0045. No comparative experiment, selected seed, or metric change.
  README, core docs, and feature inventory describe the event boundary.
- Plan changes: P3.4c3, parent P3.4c, and P3.4 checked only after the
  cross-backend full gate. P3.5 isolated function-preserving split checks
  are next; no P3.6/P3.10 acceptance was narrowed or transferred.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 still need
  actual CIFAR/CUDA and larger-data evidence. No local P3.5 blocker.
- Exact next action: write focused NumPy/Torch tests that disable split
  noise, replay, pruning, and homeostasis, compare predictions before
  and after single and repeated splits within dtype tolerance, and check
  parent/child outgoing-weight conservation. Test noisy splits separately
  and record any negative result without tuning seeds or metrics.

## 2026-09-25 — P3.5 isolated split conservation

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; prior local and
  unrelated user changes remain. No commit, reset, dataset download,
  ranking, or sweep.
- Completed task ID: **P3.5**. With replay, pruning, homeostasis,
  chemical reset, and split noise off, NumPy and Torch preserve predictions
  across one split and a second split of the new child. Incoming columns
  and hidden bias duplicate; parent-plus-child outgoing rows conserve.
  Separate seeded nonzero-noise cases confirm an actual perturbation,
  outgoing conservation, and isolated prediction agreement within dtype
  tolerance. A full sleep event with other components enabled is outside
  this preservation assertion. No algorithm, learning metric, or baseline
  setting changed.
- Commands and outcomes:
  `.venv\Scripts\pytest.exe -q -x tests/test_split_function_preservation.py`
  passed **4** cases on its first run. A later mypy pass caught one test
  variable reused across backend result types, and a format check caught
  one long assertion; both were fixed. Final focused file passed **4**;
  `.venv\Scripts\pytest.exe -ra` passed **573 in 74.35 s, 0 skipped**;
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **94 files**;
  `.venv\Scripts\ruff.exe format --check
  tests/test_split_function_preservation.py` and `git diff --check` passed
  (existing Windows LF/CRLF notices only). The checks use `1e-12`
  absolute tolerance for NumPy float64 predictions and `2e-6` for Torch
  float32 logits, with zero relative tolerance.
- Experiment artifacts: deterministic
  `tests/test_split_function_preservation.py` and ADR-0046. Seeds were
  fixed for branch coverage, not selected by predictive performance.
  README, core docs, and feature inventory describe the isolated scope.
- Plan changes: P3.5 checked after the full gate. P3.6 pruning metadata
  alignment and explicit proposed/scheduled/actually removed telemetry
  remain next; P3.7–P3.10 transaction/resume/telemetry tasks stay open.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 still need
  actual CIFAR/CUDA and larger-data evidence. No local P3.6 blocker.
- Exact next action: inventory immediate/gradual prune mutation and all
  aligned per-neuron arrays on NumPy/Torch; add tests for minimum-width
  protection, pending masks, counters, downstream dimensions, and
  proposed versus scheduled versus actually removed status. Keep pending
  gradual marks distinct from completed removals before changing
  telemetry, then run full quality gates.

## 2026-09-25 — P3.6a pruning alignment gate

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; prior local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task ID: **P3.6a**. NumPy immediate and gradual pruning and
  Torch immediate pruning retain exactly the surviving values across
  every adaptive vector, incoming/outgoing weights, bias, stable IDs,
  and downstream prediction dimensions. NumPy pending marks keep the
  unit active, reserve minimum-width capacity, decay by wake updates,
  and finalize at the configured TTL. Sleep and wake clocks report
  successful work independently of net width.
- Commands and outcomes:
  `.venv\Scripts\pytest.exe -q -x tests/test_prune_metadata_alignment.py`
  passed **4** new cases; `.venv\Scripts\pytest.exe -ra` passed
  **577 in 96.13 s, 0 skipped**; `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **95 files**;
  `.venv\Scripts\ruff.exe format --check
  tests/test_prune_metadata_alignment.py` and `git diff --check` passed
  (existing Windows LF/CRLF notices only).
- Experiment artifacts: deterministic `tests/test_prune_metadata_alignment.py`
  and ADR-0047. No ranking experiment, selected seed, or metric change;
  core docs and feature inventory record the invariant.
- Plan changes: split P3.6 into a alignment/capacity validation and b
  typed outcome reporting because NumPy's delayed removal and Torch's
  immediate removal have distinct timelines. P3.6a is complete; b and
  parent remain unchecked. Legacy positional prune fields and report
  counts remain reproducible.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 still need
  actual CIFAR/CUDA and larger-data evidence. No local P3.6b blocker.
- Exact next action: add failing tests for stable IDs proposed, scheduled,
  and actually removed during NumPy/Torch sleep, NumPy external proposals,
  and delayed wake finalization. Add an immutable outcome contract and
  pending-ID view, preserve positional fields/counts, and verify snapshot
  continuation and rejected-proposal isolation.

## 2026-09-25 — P3.6b typed outcomes and P3.6 closeout

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`; prior local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task IDs: **P3.6b, P3.6**. A frozen `PruneOutcome` now names
  validated proposed, delayed scheduled, and actually removed stable
  neuron IDs. Executed NumPy/Torch sleep results expose it; skipped
  events leave it empty. NumPy external proposal application returns
  one, and a `CircadianTrainResult` subtype reports delayed removals
  finalized during wake. `get_pending_prune_ids()` derives active marks;
  Torch returns empty because it removes immediately. Torch maps
  post-split child positions to IDs before removal. NumPy event outcomes
  also capture an older pending unit removed during sleep replay even
  when that sleep has no new prune request. Malformed pending mask/TTL
  state now rejects before training or restore. No learning metric or
  split/prune selection changed.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_prune_outcomes.py` failed
  on the missing event field. Expanded file passed **12** cases. A
  malformed-mask case initially raised `IndexError`; the getter now
  returns a useful `ValueError`. A corrupt snapshot with a marked unit
  and zero TTL initially restored; topology validation now rejects it
  without changing live state. Focused prune/preflight/snapshot checks
  passed **32** before the added replay boundary case. Final
  `.venv\Scripts\pytest.exe -ra` passed **589 in 86.95 s, 0 skipped**;
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **96 files**;
  `.venv\Scripts\ruff.exe format --check tests/test_prune_outcomes.py`
  and `git diff --check` passed (existing Windows LF/CRLF notices only).
  The final dataclass default and local formatting edits were followed
  by a passing 12-case focused run and static checks.
- Experiment artifacts: deterministic `tests/test_prune_outcomes.py`
  and ADR-0048. No performance experiment, ranking, selected seed, or
  metric change. README, core/app module docs, and feature inventory
  state the new typed contract and legacy report meaning.
- Plan changes: P3.6b and parent P3.6 checked after the cross-backend
  gate. Existing `pruned_indices` and app `total_prunes` remain counts of
  selected requests, including gradual marks; they are not silently
  redefined as completed removals. Broader app status aggregates and
  trigger/guard/duration telemetry remain P3.10. P3.7 rollback is next.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 still need
  actual CIFAR/CUDA and larger-data evidence. No local P3.7 blocker.
- Exact next action: trace every toy, continual, matched-head, and
  vision guard rollback boundary against the P3.3 full-state APIs.
  Write failing rejected/nonfinite/structural-event fixtures comparing
  all algorithmic fields and the next seeded update/draw to an untouched
  control. Keep attempt and rollback telemetry outside restored learning
  state; implement atomic restoration only after defining that boundary.

## 2026-09-25 — P3.7a–b atomic sleep and guard rollback

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Prior local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task IDs: **P3.7a, P3.7b, P3.7**. NumPy/Torch executed sleep
  now snapshots after eligibility checks and restores on mutation errors,
  nonfinite post-state, or invalid topology. NumPy replay and RNG, Torch
  split RNG, chemistry, counters, and lineage are included. Skipped
  events retain their early return. Matched-head and frozen-backbone
  vision guards restore the head on finite score rejection, scoring
  exceptions, or nonfinite scores/deltas. Invalid nonfinite tolerance
  rejects before training. Operational attempt/rollback counts remain
  local to runners; accepted event and legacy report meanings remain.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_atomic_sleep_core.py`
  failed because post-replay failure advanced NumPy RNG; after the core
  transaction, six cases passed. The initial guard run failed on the
  missing separately testable vision boundary; after extraction and
  guard recovery, 17 guard cases passed. Focused combined suite passed
  **23**. Final `.venv\Scripts\pytest.exe -ra` passed **612 in 75.98 s,
  0 skipped**; `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **98 files**;
  `.venv\Scripts\ruff.exe format --check
  tests/test_guarded_sleep_atomicity.py tests/test_atomic_sleep_core.py`
  and `git -c core.safecrlf=false diff --check` passed.
- Experiment artifacts: deterministic
  `tests/test_atomic_sleep_core.py`,
  `tests/test_guarded_sleep_atomicity.py`, ADR-0049, and ADR-0050.
  Rejected events match untouched controls on next seeded split and wake
  update. Core/app module docs and the feature inventory reflect the
  new contract. No ranking experiment, baseline tuning, selected seed,
  or metric change.
- Plan changes: split P3.7 into core transaction P3.7a and runner guard
  P3.7b after inspection showed distinct failure boundaries. Both
  children and parent are checked only after their acceptance tests and
  full gate. P3.8 still owns retry/cooldown after a valid rejection;
  P3.9 owns durable resume. No acceptance criterion was weakened.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 still need
  actual CIFAR/CUDA and larger-data evidence. No local P3.8 blocker.
- Exact next action: inspect matched-head and vision periodic/adaptive
  attempt scheduling after a rejected guarded sleep. Write a small
  deterministic repeated-rejection fixture, then define and test an
  explicit operational retry/cooldown policy whose counters remain
  outside restored model state and reproduce across a seeded resume.

## 2026-09-25 — P3.8a–b rollback retry gate

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Prior local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task IDs: **P3.8a, P3.8b, P3.8**. A runner-owned
  `SleepRollbackCooldown` blocks another due sleep until the configured
  completed-epoch pause has passed and a successful new wake batch has
  occurred. The default resolves to zero in `legacy`/`disabled` and one
  in corrected `components`; explicit nonnegative overrides include
  zero. Both guarded runners apply it to periodic and adaptive due
  attempts before guard scoring. Accepted sleep never arms the gate.
  Rejection and suppression counts plus a versioned operational state
  snapshot remain outside the head rollback snapshot. Matched-head,
  vision final-test, and vision validation reports expose the resolved
  cooldown, actual attempts, and suppressed due attempts; CLI config
  exposes an override.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_sleep_retry_policy.py`
  failed on missing policy imports; the runner fixture initially failed
  on the absent config field. The pure file passed **12** cases and the
  runner file **20**. Focused retry/guard/scheduling/accounting tests
  passed **77**. Final `.venv\Scripts\pytest.exe -ra` passed **644 in
  82.77 s, 0 skipped**; `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **100 files**;
  `.venv\Scripts\ruff.exe format --check
  tests/test_sleep_retry_policy.py tests/test_sleep_retry_runners.py`
  and `git -c core.safecrlf=false diff --check` passed. A final
  test-only assertion for exact retry epochs (1 and 3) was followed by
  **20** passing runner cases, mypy, and format checks; no source change
  followed the full suite.
- Experiment artifacts: deterministic
  `tests/test_sleep_retry_policy.py`,
  `tests/test_sleep_retry_runners.py`, and ADR-0051. In a four-epoch
  empty-wake fixture, legacy and explicit-zero settings attempt four
  identical rejected head states; component mode attempts once and
  suppresses three due events. With one wake batch per epoch, component
  mode retries at epochs 1 and 3, and two seeded runs finish with equal
  full head state. No ranking sweep, selected seed, baseline tuning, or
  metric change. README, app module docs, and inventory record behavior.
- Plan changes: split P3.8 into independently testable operational state
  and runner integration. Checked both children and parent only after
  focused, public-report, and full quality gates. The mode-resolved
  default preserves reviewed legacy scheduling; component-mode policy
  is fixed rather than chosen from outcome data. P3.9 retains durable
  combined model/runner/checkpoint resume.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 still need
  actual CIFAR/CUDA and larger-data evidence. No local P3.9 blocker.
- Exact next action: inventory NumPy/Torch runner-owned state and artifact
  writers. Add a failing small uninterrupted-versus-resumed fixture
  before sleep, after accepted sleep, and after rejected sleep; then
  specify and implement a compatible combined checkpoint including
  model state, P3.8 retry state, epoch/data-order progress, and caller
  RNG, with explicit incompatible-config rejection.

## 2026-09-25 — P3.9a combined in-memory checkpoint

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Extensive prior local
  and unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task IDs: **P3.9a**. `circadian_checkpoint.py` captures a
  versioned detached NumPy network or CPU Torch head snapshot together
  with protocol/config/data identity, typed completed-epoch and sleep
  stage, optional P3.8 retry state, Python/NumPy process RNG, and Torch
  CPU process RNG for a Torch head. Restore checks identity, retry and
  process RNG state, and a detached candidate model before changing
  live state. The runner must supply and later verify its data digest.
- Commands and outcomes: the initial
  `.venv\Scripts\pytest.exe -q -x tests/test_combined_circadian_checkpoint.py`
  failed on the missing module. After implementation, the focused file
  passed **32** cases; combined checkpoint/full-snapshot/retry tests
  passed **43** before the extra rejection cases. Final
  `.venv\Scripts\pytest.exe -ra` passed **676 in 100.39 s, 0 skipped**;
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **102 files**;
  `.venv\Scripts\ruff.exe format --check
  src/app/circadian_checkpoint.py tests/test_combined_circadian_checkpoint.py`
  and `git -c core.safecrlf=false diff --check` passed. The latter
  checks tracked edits; the new Python files passed the explicit Ruff
  format check.
- Experiment artifacts: deterministic
  `tests/test_combined_circadian_checkpoint.py` and ADR-0052. The three
  interruption positions are pre-sleep, accepted sleep, and rejected
  sleep; resumed fresh models match uninterrupted future wake/split
  events, retry suppression, complete model state, and process RNG
  draws on both backends. Thirteen invalid variants per backend reject
  without changing destination model or retry state. No ranking
  experiment, selected seed, baseline tuning, or metric change.
- Plan changes: P3.9 was split into in-memory primitive P3.9a,
  file-backed fixed-feature runner P3.9b, and remaining NumPy/image
  routes P3.9c after inspecting external state. P3.9a is checked only
  after its acceptance gate. P3.9b/c and parent stay unchecked; their
  acceptance criteria are preserved. Fixed-feature runner inspection
  confirmed materialized train/guard/validation batches, sequential
  baseline trainers, and a circadian epoch loop with local counters.
- Skipped tests and blockers: pytest skipped none. P1.7/P1.8 still need
  actual CIFAR/CUDA and larger-data evidence. CPU-only P3.9a does not
  claim CUDA RNG, classifier, image loader, file durability, or runner
  counter coverage. No local blocker for P3.9b.
- Exact next action: write a failing trusted-file round-trip and
  incompatible-file fixture for the fixed-feature circadian runner.
  Add feature-bank identity, next batch/pre-post-sleep stage, report
  counters, and P3.9a state to a local checkpoint; then compare
  uninterrupted versus resumed accepted and rejected guarded runs
  without touching final test during training.

## 2026-09-25 — P3.9b1 fixed-feature CPU file resume

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Prior local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task IDs: **P3.9b1**. The ordinary CPU, fixed-epoch
  three-head route can save and resume its circadian head using a
  `FixedFeatureCheckpointStore` app port and a trusted local file adapter.
  The actual guarded loop saves after successful wake batches and on
  both sides of sleep. Its file binds the complete runner config,
  initial head, split and materialized train/guard/validation hashes,
  batch/sleep cursor, report counters, P3.8 retry state, model state,
  and caller Python/NumPy/Torch CPU RNG. Restore checks file integrity,
  identity, counters, and the detached model candidate before live
  mutation. The public route retains baseline training order and opens
  final test only after all heads finish.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_fixed_feature_checkpoint_resume.py`
  failed on the missing file-store module, then on the missing runner
  checkpoint arguments. After implementation and fixture adjustment
  for dual-chemical recomputation, the focused file passed **15**
  cases. Existing matched-head, retry, and combined-checkpoint tests
  passed. Final `.venv\Scripts\pytest.exe -ra` passed **691 in
  75.05 s, 0 skipped**. A test-only assertion for the next Python,
  NumPy, and Torch CPU draws was added afterward and the 15 focused
  cases passed again; no source change followed the full suite.
  `.venv\Scripts\ruff.exe check .` passed;
  `.venv\Scripts\mypy.exe src tests scripts` passed for **105 files**;
  `.venv\Scripts\ruff.exe format --check` on the five new/changed
  Python files and `git -c core.safecrlf=false diff --check` passed.
  `.venv\Scripts\python.exe -c "import torch; print('torch',
  torch.__version__, 'cuda_available', torch.cuda.is_available())"`
  reported `torch 2.14.0+cpu cuda_available False`.
- Experiment artifacts: `tests/test_fixed_feature_checkpoint_resume.py`,
  `src/app/fixed_feature_checkpoint.py`,
  `src/infra/circadian_checkpoint_files.py`, and ADR-0053. Pytest
  temporary checkpoint files exercised checksum and atomic replacement;
  no research dataset or ranking artifact was produced. A real
  `CircadianPredictiveCodingHead` split or rejected guard rollback
  matched uninterrupted future state, report counters, retry behavior,
  and process draws after file resume. A public-route test confirmed
  equal trained-head hashes and sealed final-test access. No baseline
  tuning, selected seed, metric change, or large sweep.
- Plan changes: split P3.9b into the CPU fixed-epoch route P3.9b1 and
  remaining device/deadline/memory/capacity audit P3.9b2 after code
  inspection showed different external state and timing semantics.
  Checked b1 only after its acceptance gate. P3.9b2, P3.9b, P3.9c,
  and parent P3.9 remain unchecked with their original acceptance
  intact. README, app/infra module docs, and feature inventory record
  the scoped capability and its timing limitation.
- Skipped tests and blockers: pytest skipped none. The host has a
  CPU-only Torch build, so CUDA continuation cannot be verified here.
  P1.7/P1.8 still need actual CIFAR/CUDA and larger-data evidence.
  Checkpointed or resumed `train_seconds` includes persistence work and
  is excluded from equal-time head comparisons. Other fixed-feature
  modes reject checkpoint requests or have no checkpoint entry point;
  P3.9b2 retains that work.
- Exact next action: inspect wall-time deadline and memory/capacity
  report state under interruption. Define a deterministic resume
  contract for their clocks and peaks before adding a failing CPU
  fixture; preserve explicit errors for unsupported modes and keep
  CUDA evidence unchecked until a CUDA host is available. Then advance
  P3.9c's NumPy toy/continual and whole-image loader/classifier state.

## 2026-09-25 — P3.9b2a CPU wall-time checkpoint deadline

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Prior local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task IDs: **P3.9b2a**. The CPU wall-time fixed-feature route
  now accepts the same trusted checkpoint store and explicit resume
  flag. The file binds its wall-time protocol and per-head budget.
  Resume receives the original deadline minus saved active training
  seconds. Checkpoint snapshot/serialization/write time is paused out
  of the circadian head's active allowance so the baselines' declared
  learning budget is not changed. A post-sleep checkpoint is written
  before checking a deadline reached during guarded sleep, avoiding
  repetition of that event after restart. Full model, retry, and work
  counters still use the P3.9b1 checkpoint contract.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x
  tests/test_fixed_feature_checkpoint_resume.py -k wall_time_file`
  failed because the runner rejected a checkpoint with any wall-time
  budget. The three controlled-clock pre-sleep/accepted/rejected cases
  passed after implementation, as did the public forwarding/test-seal
  case. Focused checkpoint/matched-head/capacity/retry tests passed.
  Final `.venv\Scripts\pytest.exe -ra` passed **695 in 97.84 s,
  0 skipped**; `.venv\Scripts\mypy.exe src tests scripts` passed for
  **105 files**; `.venv\Scripts\ruff.exe check .`,
  `.venv\Scripts\ruff.exe format --check` on the three changed Python
  files, and `git -c core.safecrlf=false diff --check` passed.
- Experiment artifacts: four new deterministic cases in
  `tests/test_fixed_feature_checkpoint_resume.py` and ADR-0054. The
  fake clock advances one second per successful wake update and a
  separate quarter-second per file write. An uninterrupted and a
  resumed run both stop after five wake batches with equal head state,
  report counters, and 5.0 active seconds. A changed 6.0-second budget
  rejects before head mutation. The public wall-time route forwards the
  store and does not open final test on interruption. No ranking
  experiment, baseline tuning, selected seed, or metric change.
- Plan changes: split P3.9b2 into wall-time deadline P3.9b2a and
  memory/capacity/CUDA state P3.9b2b after inspecting their distinct
  clocks. Checked only b2a. P3.9b2b, P3.9b2, P3.9b, and P3.9 remain
  unchecked. Split P3.9c into NumPy toy/continual P3.9c1 and unmatched
  whole-image P3.9c2 because they own different data-order state;
  both and parent P3.9c stay unchecked. ADR-0053, README, app docs,
  and feature inventory now distinguish fixed-epoch timing from the
  resumed wall-time active budget. No acceptance criterion was weakened.
- Skipped tests and blockers: pytest skipped none. This host has
  `torch 2.14.0+cpu` and no CUDA. The RSS sampler currently finalizes
  peaks only after its wrapped trainer returns; an interrupted head
  checkpoint cannot claim an uninterrupted process-memory peak.
  P3.9b2b retains memory/capacity/CUDA work. P1.7/P1.8 still require
  actual CIFAR/CUDA and larger-data evidence.
- Exact next action: inventory the actual NumPy toy runner's seeded
  dataset/split identity, loop cursor, counters, replay inputs, and
  report writes. Add a failing trusted-file uninterrupted-versus-resumed
  fixture around structural/replay sleep, then implement P3.9c1 for
  toy and continual phase order without early held-out evaluation.
## 2026-09-25 — P3.9c1a NumPy toy-runner file resume

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Prior local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task IDs: **P3.9c1a**. The public toy API accepts a trusted
  local checkpoint store and explicit resume flag. It persists all three
  models, diagnostic histories, sleep counters, the next model in each
  configured training order, the pre/post-sleep stage, full circadian
  replay/topology state, and Python/NumPy process RNG. Config and actual
  development-role arrays bind the file. Baseline topology/traffic and
  report cursor are checked before restoring the fresh circadian model.
  Final-test scoring remains after training; both validation and legacy
  protocols retain their identities. Arbitrary stateful external policy
  resume is explicitly rejected; ordinary non-checkpoint use is unchanged.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_toy_checkpoint_resume.py`
  failed at collection because the store did not exist. The first
  uninterrupted/resumed state assertion exposed differing pickle
  serialization for an equal replay deque; comparing its arrays and
  priority fields resolved that test error. Final focused
  `.venv\Scripts\pytest.exe -q -x tests/test_toy_checkpoint_resume.py
  tests/test_experiment_runner.py tests/test_fixed_feature_checkpoint_resume.py`
  passed **36** cases. Final `.venv\Scripts\pytest.exe -ra` passed
  **704 in 84.61 s, 0 skipped** after the app-side array-role protocol
  cleanup; the preceding full run also passed 704/0 skipped.
  `.venv\Scripts\mypy.exe src tests
  scripts` passed for **107 files**; `.venv\Scripts\ruff.exe check .`,
  `.venv\Scripts\ruff.exe format --check` on the four changed Python
  files, and `git -c core.safecrlf=false diff --check` passed.
- Experiment artifacts: `tests/test_toy_checkpoint_resume.py`,
  `src/app/toy_checkpoint.py`, the new toy adapter in
  `src/infra/circadian_checkpoint_files.py`, and ADR-0055. Pytest
  temporary files exercised checksummed replacement. Nine new cases
  cover forward/reverse order, wake/pre-/post-sleep interruption around
  real splitting and replay, equal baseline/circadian contents, reports,
  next random draws, changed order/data, corrupt bytes, malformed report
  counters/arrays, legacy protocol, and sealed final-test access. No
  research ranking artifact, baseline tuning, chosen seed, or metric change.
- Plan changes: split P3.9c1 into P3.9c1a toy and P3.9c1b continual after
  inspection showed continual has separate seed/phase cursors and frozen
  phase-A models. Checked c1a only after its gate. P3.9c1b, P3.9c1,
  P3.9c, and parent P3.9 remain unchecked; their acceptance criteria
  are unchanged. README, architecture, and feature inventory describe
  the scoped route and trust boundary.
- Skipped tests and blockers: pytest skipped none. Continual and
  whole-image durable resume remain implementation tasks. The host has
  CPU-only Torch, so CUDA evidence remains blocked by an actual CUDA
  host. P3.9b2b retains process-memory/capacity/CUDA work; P1.7/P1.8
  retain larger-data/CIFAR/CUDA evidence.
- Exact next action: inventory continual seed-list accumulation,
  phase-A frozen model copies, phase-specific epoch counters, development
  role hashes, and final-test timing. Add a failing actual continual
  before-/after-structural-replay-sleep file-resume fixture spanning
  phases A and B, then implement P3.9c1b without changing metrics,
  baselines, or seed selection.

## 2026-09-25 — P3.9c1b continual NumPy file resume

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Prior local and
  unrelated user changes remain. No commit, reset, download, or sweep.
- Completed task IDs: **P3.9c1b, P3.9c1**. The public continual Python API
  now accepts a trusted local checkpoint store and explicit resume flag.
  It records the ordered seed list, phase-local/global epoch and model-step
  cursors, pre/post-sleep stage, both mutable baselines, full circadian
  replay/topology and process RNG, sleep report counters, phase-A-frozen
  models, and committed seed reports. It binds actual A/B development
  roles before training. Both held-out tests are scored only after that
  seed completes both phases; a completed-result test digest is recorded
  afterward. Completed seeds are validated on
  resume and never retrained or rescored. The ordinary path, both protocol
  IDs, baseline learning rules, metric definitions, and seed selection
  remain unchanged.
- Commands and outcomes: initial
  `.venv\Scripts\pytest.exe -q -x tests/test_continual_checkpoint_resume.py`
  failed at collection because the new store did not exist. The first
  focused run exposed the fixture's extra four post-training test-hash
  reads; the sealed-role assertion now distinguishes those from the 18
  scoring reads. Review found that a `seed_complete` resume skipped the
  saved process RNG, which was fixed and covered by next-draw tests.
  Focused `.venv\Scripts\pytest.exe -q -x
  tests/test_continual_checkpoint_resume.py
  tests/test_continual_shift_benchmark.py
  tests/test_toy_checkpoint_resume.py
  tests/test_fixed_feature_checkpoint_resume.py` passed **59** cases.
  Full `.venv\Scripts\pytest.exe -ra` passed **722 in 81.34 s, 0
  skipped**. A later test-only full-snapshot comparison passed all 18
  continual checkpoint cases. `.venv\Scripts\mypy.exe src tests scripts`
  passed for **110 files**; `.venv\Scripts\ruff.exe check .`,
  `.venv\Scripts\ruff.exe format --check` on six changed Python files,
  and `git -c core.safecrlf=false diff --check` passed.
- Experiment artifacts: `tests/test_continual_checkpoint_resume.py`,
  `src/app/continual_checkpoint.py`, shared NumPy role/model validation in
  `src/app/numpy_checkpoint_validation.py`, the continual adapter in
  `src/infra/circadian_checkpoint_files.py`, and ADR-0056. Pytest local
  temporary files exercised checksummed replacement. Eighteen new cases
  cover both model orders, partial B wake, A/B pre/post-sleep, A-to-B
  transition, first/terminal committed seed, structural replay, exact
  model/frozen-A state and process draws, changed config/seeds/data,
  malformed counters/frozen topology, corrupt bytes, legacy protocol,
  and sealed final tests. No research ranking artifact, baseline tuning,
  chosen seed, metric change, or large sweep.
- Plan changes: the previously planned P3.9c1b route is complete.
  P3.9c1a and c1b together satisfy the original P3.9c1 NumPy runner
  criteria, so c1 is checked. A small shared app module now hashes exact
  labeled roles and validates NumPy baseline arrays for both runners;
  the toy checkpoint digest and behavior remain unchanged. P3.9c2,
  P3.9c, and parent P3.9 remain unchecked with their original acceptance
  criteria. README, architecture, and feature inventory describe the
  scoped capability and trusted-file boundary.
- Skipped tests and blockers: pytest skipped none. The whole-image
  classifier/loader resume is the next implementation task. This host's
  Torch build is CPU-only; CUDA evidence needs an actual CUDA host.
  P3.9b2b retains process-memory/capacity/CUDA work; P1.7/P1.8 retain
  larger-data/CIFAR/CUDA evidence.
- Exact next action: inventory `src/app/resnet50_benchmark.py` and its
  unmatched vision training loop, full-classifier snapshot, DataLoader
  sampler/augmentation RNG, guard retry state, epoch/batch progress,
  and final-test boundary. Add a failing bounded CPU interrupted-versus-
  resumed fixture before/after guarded sleep, then implement P3.9c2
  without changing matched baseline budgets or final-test timing.

## 2026-09-25 — P3.9c2a seeded vision loader cursor

- Repository state: `master` still points to reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Existing local and
  unrelated user changes were preserved. No commit, reset, download,
  final-test evaluation artifact, or sweep.
- Completed task ID: **P3.9c2a**. `app.seeded_vision_loader` now holds the
  seeded v3 train-loader cursor. It captures the epoch and next logical batch,
  sampler state, entry/current Torch CPU state, and Python/NumPy state. Fresh
  loaders replay skipped batches to reconstruct zero-worker transforms and
  multiworker stochastic views. A sampler mismatch restores caller and loader
  RNG before raising. Checkpoint methods reject incompatible batch/prefetch
  settings and loader configurations that cannot be replayed reliably. The
  ordinary training seed schedule, baseline learning budgets, metrics, and
  final-test timing are unchanged.
- Commands and outcomes: initial `.venv\Scripts\pytest.exe -q -x
  tests/test_resnet50_benchmark.py -k "resumes_mid_epoch or
  invalid_resume_cursor"` failed on the missing `snapshot_state` method.
  A stronger fixture then failed on unequal next process draws, exposing the
  need to keep the epoch-entry Torch stream and restore cursor streams after
  replay. Focused four cases passed after the fix. The existing vision test
  file passed before extraction to the new module. Full
  `.venv\Scripts\pytest.exe -ra` passed **726 in 106.14 s, 0 skipped**;
  its final rerun after the resettable-loader validation passed **726 in
  114.07 s, 0 skipped**. A focused run after test formatting passed four cases.
  `.venv\Scripts\mypy.exe src tests scripts` passed **111 files**;
  `.venv\Scripts\ruff.exe check .`, `.venv\Scripts\ruff.exe format --check
  src/app/seeded_vision_loader.py`, the changed-test range format check, and
  `git -c core.safecrlf=false diff --check` passed.
- Artifacts and evidence: `src/app/seeded_vision_loader.py`, four new
  parameterized/rejection cases in `tests/test_resnet50_benchmark.py`, and
  ADR-0057. The CPU cases compare remaining batches, the next epoch, and
  next Torch/NumPy/Python draws for zero-worker Torchvision views, zero-worker
  mixed stochastic images, and two-worker mixed stochastic images, with
  process draws interleaved during training. Invalid cursor index/RNG,
  changed batch/prefetch settings, persistent workers, and a valid but wrong
  sampler state reject; the last case verifies no caller/loader RNG change.
  No scientific ranking, selected seed, or baseline tuning was produced.
- Plan changes: split P3.9c2 into loader replay proof P3.9c2a and durable
  whole-run integration P3.9c2b because prefetched worker queues need
  reconstruction while the runner still owns three model outcomes, full
  classifier state, guard/retry counters, data identity, and final-test
  isolation. Checked c2a only after its gate. P3.9c2b, c2, c, and P3.9
  retain the original acceptance criteria and remain unchecked. README,
  architecture, and app module docs identify this internal cursor boundary.
- Skipped tests and blockers: pytest skipped none. CUDA-specific continuation
  evidence remains unavailable on this CPU-only Torch host. Whole-image
  file resume and fixed-feature memory/capacity/CUDA task P3.9b2b are still
  unfinished; no large experiment was started.
- Exact next action: add a failing small CPU P3.9c2b actual file-resume
  fixture for the public unmatched runner at mid-wake, before sleep, and
  after accepted/rejected guarded sleep. Capture all three outcomes and
  counters plus the complete classifier and loader cursor, bind exact
  train/guard/validation data and protocol before restoration, and keep
  final-test scoring sealed until all training completes.

## 2026-09-28 — P3.9c2b1 seeded unmatched vision runner file resume

- Repository state: `master` still points to the reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Existing local and
  unrelated changes were preserved. No commit, reset, download, final-test
  experiment artifact, or large sweep.
- Completed task ID: **P3.9c2b1**. The public v3 unmatched vision Python
  runner now saves a trusted local file after each finished variant and at
  each circadian wake, pre-sleep, and post-sleep boundary. The payload binds
  the full config, protocol, model order, and exact raw train/guard/validation
  content. Completed backprop/predictive/circadian outcomes use detached full
  model state and trained hashes; active circadian state includes backbone,
  head, lineage/RNG, loader cursor, retry gate, report counters, and process
  streams. Resume validates a fresh candidate before restoring the process
  stream or training. Final-test scoring starts only after all three models
  finish. File I/O and loader replay are excluded from accumulated active
  circadian training seconds; elapsed times are not expected to match exactly.
- Commands and outcomes: the initial `.\.venv\Scripts\pytest.exe -q -x
  tests/test_vision_checkpoint_resume.py` regression first failed on the
  missing file route, then on trying to pickle a live Torch module reference,
  then on missing active circadian saves. Each exposed a required state
  boundary. The final focused file passed **25 cases**, including bounded
  real-backbone model resume and tiny-backbone repeated sleep fixtures.
  `.\.venv\Scripts\pytest.exe -q -x tests/test_vision_checkpoint_resume.py
  -k bad_active_progress` passed 8 rejection cases;
  `-k augmented_train_loader` passed zero- and two-worker cases;
  `-k all_trained_models` passed terminal resume; and
  `-k earlier_unmatched_protocols` passed 2 early unsupported checks.
  Final `.\.venv\Scripts\pytest.exe -ra` passed **751 in 172.66 s,
  0 skipped**. `.\.venv\Scripts\ruff.exe check .`,
  `.\.venv\Scripts\mypy.exe src tests scripts` (113 files),
  `.\.venv\Scripts\ruff.exe format --check src/app/vision_checkpoint.py
  tests/test_vision_checkpoint_resume.py`, and
  `git -c core.safecrlf=false diff --check` passed. A prior mypy run caught
  the test fixture's dynamic `torch.nn.Module` base; an explicit `torch.nn`
  import fixed it.
- Artifacts and evidence: `src/app/vision_checkpoint.py`,
  `src/app/resnet50_benchmark.py`, `src/app/seeded_vision_loader.py`,
  `src/infra/circadian_checkpoint_files.py`,
  `tests/test_vision_checkpoint_resume.py`, and ADR-0058. The real ResNet
  checkpoint fixture produced an approximately 270 MB temporary file;
  pytest owns that temporary artifact. Cases compare trained hashes,
  non-timing reports, split identities, sleep split/rollback counters, and
  next Python/NumPy/Torch draws with uninterrupted runs. Corrupt checksum,
  changed data/order/config, invalid completed state/report, active cursor,
  classifier, counter, and RNG reject before training or process RNG change.
  No scientific ranking, seed selection, baseline tuning, or metric change.
- Plan changes: split broad P3.9c2b into seeded CPU P3.9c2b1 and remaining
  legacy/device P3.9c2b2 because only v3 has a replayable logical loader
  cursor; v1/v2 share differently scoped loader streams and guard roles.
  Checked b1 after its gate. P3.9c2b2/b/c2/c/P3.9 remain unchecked with
  original acceptance intact. README, architecture, and module docs now
  describe the supported Python file route and trusted-pickle boundary.
- Skipped tests and blockers: pytest skipped none. The v1/v2 checkpoint
  requests explicitly reject before training; they require an exact
  continuation design or remain unsupported under b2. `torch 2.14.0+cpu`
  reports `cuda_available False`, so CUDA continuation was not run.
  P3.9b2b memory/capacity/CUDA work and larger P1.7/P1.8 evidence stay open.
- Exact next action: inspect the v1/v2 unmatched loader and guard-role
  behavior, then add a failing bounded file-resume fixture at a completed
  model boundary and at active circadian wake/sleep without changing their
  seed/order/metric semantics. If an exact cursor cannot be reconstructed,
  record that invariant and leave b2 unchecked; run CUDA evidence on an
  actual CUDA host.

## 2026-09-28 — P3.9c2b2a older unmatched vision CPU continuation

- Repository state: `master` remains at reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. The existing dirty
  checkout and unrelated user changes remain intact. No commit, reset,
  download, final-test experiment artifact, or sweep.
- Completed task ID: **P3.9c2b2a**. The trusted unmatched vision file route
  now resumes v1 and v2 on CPU. It carries the shared DataLoader generator
  across completed models. During circadian training a separate no-reseed
  cursor captures the shared sampler's epoch-entry/current states and
  process augmentation streams, replays completed logical batches, checks
  the sampler position, and restores streams before the next update. The
  ordinary v1/v2 loops retain their historical shared RNG behavior; v1
  still aliases validation as guard and v2 uses its distinct guard. The
  full classifier, retry gate, report counters, development-role content,
  and final-test boundary use the existing trusted runner checkpoint.
- Commands and outcomes: `.\.venv\Scripts\pytest.exe -q -x
  tests/test_vision_checkpoint_resume.py -k
  earlier_vision_file_resume_preserves_shared_loader` first failed on the
  explicit seeded-only error, then passed both v1/v2 cases after storing
  and restoring the shared generator. The active guarded-sleep fixture
  first failed because the older route had no in-epoch save, then passed
  eight wake/pre-/accepted-/rejected-sleep cases after the shared cursor.
  The focused checkpoint file passed **41** cases; the combined
  `tests/test_vision_checkpoint_resume.py
  tests/test_resnet50_benchmark.py` command passed. Final
  `.\.venv\Scripts\pytest.exe -ra` passed **767 in 166.30 s, 0 skipped**.
  `.\.venv\Scripts\mypy.exe src tests scripts` passed **114 files**;
  `.\.venv\Scripts\ruff.exe check .`,
  `.\.venv\Scripts\ruff.exe format --check` on the shared loader,
  vision checkpoint, and changed test file, plus
  `git -c core.safecrlf=false diff --check` passed. A first mypy run caught
  optional-state narrowing in the new cursor; it was fixed before the
  final gate.
- Artifacts and evidence: `src/app/shared_vision_loader.py`,
  `src/app/vision_checkpoint.py`, `src/app/resnet50_benchmark.py`,
  `tests/test_vision_checkpoint_resume.py`, and ADR-0059. Tests compare
  ordinary and resumed v1/v2 model hashes and reports, next process draws,
  sealed final test, eight guarded-sleep interruption cases, zero- and
  two-worker stochastic Torch/NumPy/Python views, and six changed-data,
  checksum, cursor, and shared-generator rejection cases. A valid but
  wrong replay entry state rejects without a training update or caller RNG
  change. Pytest owns its temporary checkpoint files; no persistent
  experimental ranking or selected seed was produced.
- Plan changes: split P3.9c2b2 into verified older-protocol CPU work
  P3.9c2b2a and CUDA-specific P3.9c2b2b. This follows the observed shared
  loader semantics and the CPU-only runtime. Checked only b2a. b2b, b2,
  b, c2, c, and P3.9 retain their original parent acceptance. README,
  architecture, app module docs, and ADR-0058 now point to ADR-0059.
  No baseline budget, learning rule, seed selection, metric, or final-test
  timing policy changed.
- Skipped tests and blockers: pytest skipped none. `.\.venv\Scripts\python.exe
  -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
  returned `2.14.0+cpu False`. CUDA continuation needs an actual CUDA
  host. P3.9b2b fixed-feature memory/capacity semantics remain unverified;
  larger P1.7/P1.8 evidence remains open.
- Exact next action: inspect `_run_three_head_fixed_feature_benchmark`,
  `_train_with_memory_telemetry`, and `_verify_fixed_width_capacity` in
  `src/app/matched_head_benchmark.py`. Define how process RSS and CUDA
  allocator peaks combine across resumed segments before exposing a memory
  report, then add a bounded CPU fixed-width capacity checkpoint fixture
  that checks unchanged head parameters, forced guarded sleep, and sealed
  final test. Leave CUDA tasks unchecked until run on suitable hardware.

## 2026-09-28 — P3.9b2b1 capacity-only CPU checkpoint continuation

- Repository state: the checkout is now `master` at
  `704886b1e39159726271294c7475d9e4cdf6910e` (“Add research controls
  and resumable experiments”), advanced externally from reviewed commit
  `8793c49ee4f9f8b07649e8db6571ed53746a9a06`. Existing uncommitted
  v1/v2 continuation and unrelated user changes were preserved. No commit,
  reset, download, seed sweep, or scientific ranking was made.
- Completed task ID: **P3.9b2b1**. An opt-in trusted checkpoint on the
  public fixed-width capacity runner uses the distinct
  `vision_three_head_fixed_width_capacity_checkpoint_v1` protocol. It
  reuses the CPU fixed-feature cursor, head, retry, counters, and process
  RNG while retaining the equal-width, forced guarded-sleep, unchanged
  parameter count, and pre-final-test capacity checks. Checkpointed results
  explicitly omit memory telemetry. Without a store, the existing
  `vision_three_head_fixed_width_capacity_memory_v1` behavior is unchanged.
- Commands and outcomes: `.\.venv\Scripts\pytest.exe -q -x
  tests/test_matched_head_capacity.py -k capacity_checkpoint` first failed
  on the absent `checkpoint_store` argument and passed **5 cases** after
  implementation. The capacity and fixed-feature checkpoint test files
  passed **39** cases together. `.\.venv\Scripts\pytest.exe -ra` passed
  **772 in 175.58 s, 0 skipped**. After tightening the test loader seal
  to cover resumed capacity verification and checking rejection in both
  protocol directions, `.\.venv\Scripts\pytest.exe -q -x
  tests/test_matched_head_capacity.py` passed **20 cases**. The final
  `.\.venv\Scripts\ruff.exe check .` passed; `.\.venv\Scripts\mypy.exe
  src tests scripts` passed **114 files**. After formatting the changed
  source file, `.\.venv\Scripts\ruff.exe format --check
  src/app/matched_head_benchmark.py tests/test_matched_head_capacity.py`
  passed; Ruff lint passed again.
  `git -c core.safecrlf=false diff --check` passed after the plan and log
  updates.
- Artifacts and evidence: `src/app/matched_head_benchmark.py`,
  `tests/test_matched_head_capacity.py`, README, architecture and app
  module docs, and ADR-0060. Four public-route interruption fixtures stop
  after a wake batch, before sleep, and after accepted or rejected sleep.
  Each compares uninterrupted versus resumed trained-head hashes,
  non-timing reports, initial/capacity metadata, sleep and rollback counts,
  and next Python/NumPy/Torch draws. A sealed loader rejects final-test
  access until resumed capacity verification finishes. Both protocol
  crossing directions reject before head restoration. The original
  memory-enabled capacity test still checks sampled RSS fields. Pytest
  owns the temporary checkpoint files; no durable experiment result or
  final-test ranking was produced.
- Plan changes: split P3.9b2b into capacity-only CPU continuation b1 and
  resumed memory/CUDA evidence b2. The existing checkpoint records no
  process segment peaks, so labeling the final process's sampled RSS as a
  whole-run peak would weaken the memory report. Checked b1 only after its
  acceptance gate; b2b2, b2b, b2, b, and P3.9 remain unchecked. The
  original memory-enabled route, baseline budgets, metrics, seeds, and
  learning rules were not changed.
- Skipped tests and blockers: pytest skipped none. CUDA random-state
  continuation still requires an actual CUDA host; the installed Torch
  runtime reports `2.14.0+cpu` with CUDA unavailable. CPU process-memory
  contract and process-isolated observation remain unblocked work.
- Exact next action: define P3.9b2b2's per-process RSS baseline, sample
  interval, segment data stored in the checkpoint, and aggregation rule.
  Add a bounded actual process-restart fixture that checks observations
  from both segments before exposing resumed memory telemetry. Keep CUDA
  branches open until measured on a CUDA host.

## 2026-09-28 — P3.9b2b2a checkpointed CPU RSS segments

- Repository state: `master` remains at
  `704886b1e39159726271294c7475d9e4cdf6910e`. All earlier dirty
  checkout work, including the uncommitted v1/v2 vision continuation and
  capacity-only checkpoint, was preserved. No reset, commit, network data
  fetch, seed selection, baseline tuning, or experiment sweep occurred.
- Completed task ID: **P3.9b2b2a**. CPU checkpoint-memory runs now use
  distinct fixed-epoch, wall-time, and fixed-width capacity protocol IDs.
  `ProcessRssSegment` captures PID, trainer-entry RSS, observed peak, sample
  count, and 5 ms interval. At every saved circadian boundary the payload
  carries prior segments plus the current observation. The completed report
  has a tuple of segments and an aggregate maximum absolute observed RSS;
  circadian aggregate start is `None`, and sample counts sum. Each baseline
  head has one completed-invocation segment. The old no-store memory
  protocols and default capacity-only checkpoint retain their behavior.
  CPU CUDA allocator fields remain empty. ADR-0061 defines scope and
  measurement limits.
- Commands and outcomes: `.\.venv\Scripts\pytest.exe -q -x
  tests/test_checkpoint_memory_resume.py` first failed because the public
  capacity route lacked `checkpoint_memory`; after implementation its
  initial two-process case passed. The focused five-file command
  (`tests/test_checkpoint_memory_resume.py`,
  `tests/test_matched_head_capacity.py`,
  `tests/test_fixed_feature_checkpoint_resume.py`,
  `tests/test_process_memory.py`, `tests/test_isolated_head_memory.py`)
  passed **55** cases. A typo naming nonexistent
  `tests/test_matched_head_memory.py` made an earlier collection command
  exit 1; the corrected command passed. Initial mypy found a test-only
  dynamic-method assignment, fixed with `setattr`; Ruff formatting was
  applied. `.\.venv\Scripts\pytest.exe -ra` passed **782 in 229.96 s,
  0 skipped** after the final source change. A later test-only old-file
  compatibility case passed, and final `.\.venv\Scripts\pytest.exe -q -x
  tests/test_checkpoint_memory_resume.py` passed **10** cases.
  `.\.venv\Scripts\ruff.exe check .` passed;
  `.\.venv\Scripts\mypy.exe src tests scripts` passed **115 files**;
  `.\.venv\Scripts\ruff.exe format --check` on the six changed Python
  files and `git -c core.safecrlf=false diff --check` passed.
- Artifacts and evidence: `src/shared/process_memory.py`,
  `src/app/fixed_feature_checkpoint.py`,
  `src/app/matched_head_benchmark.py`,
  `tests/test_checkpoint_memory_resume.py`,
  `tests/test_process_memory.py`,
  `tests/test_fixed_feature_checkpoint_resume.py`, README, architecture,
  evaluation protocols, app module docs, and ADR-0061. Two independent local Python processes
  verify the saved and resumed PIDs/segments, maximum/sum aggregation,
  and one segment per baseline. Wake/accepted/rejected cases compare
  learning hashes, non-timing reports, process draws, and final-test
  isolation with uninterrupted checkpoint-memory runs. Corrupt segment
  values and cross-protocol files reject before saved head/process-RNG
  restoration. A controlled-clock wall-time case retains the 5.0 s
  active budget; a simulated older capacity file without the optional
  field still resumes. Pytest owns all temporary checkpoint files.
  No durable scientific ranking or selected seed was produced.
- Plan changes: split P3.9b2b2 into CPU RSS task b2b2a and actual-host
  CUDA allocator/RNG task b2b2b before implementation. One-process
  start/peak fields could not honestly represent resumed absolute RSS;
  typed segments preserve each process baseline, and the aggregate is
  descriptive only. Checked b2b2a after the gate. b2b2b, b2b2, b2b,
  b2, b, and P3.9 remain unchecked; P1.7/P1.8 fairness work is the next
  unblocked priority. No learning rule, baseline budget, metric, guard
  decision, or final-test selection policy changed.
- Skipped tests and blockers: the full suite skipped none. The local
  runtime remains `torch 2.14.0+cpu` with CUDA unavailable; CUDA peak and
  random-stream continuation require an actual CUDA host. Actual CIFAR
  loader evidence and larger-data fairness confirmation remain open.
- Exact next action: inspect `src/infra/vision_datasets.py`,
  `src/app/resnet50_benchmark.py`, and the local dataset cache for P1.7.
  If CIFAR data are already present, add a bounded reverse-order CPU
  loader fixture without downloading or sweeping. Otherwise document
  the cache condition and audit strict-online replay/structural-noise
  streams next. Keep P1.7/P1.8 and CUDA branches unchecked until verified.

## 2026-09-28 — P1.7e actual CIFAR loader preparation remains open

- Repository state: the same `master` checkout and earlier uncommitted
  changes remain intact. `data/` is gitignored and initially absent;
  `data/cifar-10-batches-py` and `data/cifar-100-python` were absent. The
  local C: drive reported 362,168,508,416 free bytes before the attempt.
- Completed task IDs: none in this P1.7e increment. P3.9b2b2a above is
  complete, but P1.7e and parent P1.7/P1.8 remain unchecked.
- Commands and outcomes: `rg -n` on `src/infra/vision_datasets.py`,
  `src/app/resnet50_benchmark.py`, and relevant tests located the real
  torchvision CIFAR path and seeded v3 loader. `Test-Path` checks for the
  configured cache returned false. `.\.venv\Scripts\python.exe -c
  "from torchvision.datasets import CIFAR10; data=CIFAR10(root='data',
  train=True, download=True); print('CIFAR10 train count', len(data))"`
  started the official archive download, reached about **5.8%**, and was
  interrupted within the local time budget (exit 1). The incomplete
  `data/cifar-10-python.tar.gz` was 9,961,472 bytes and was removed after
  verifying its resolved path stayed under this workspace. No model
  experiment ran on actual CIFAR data. A first
  `.\.venv\Scripts\python.exe scripts/verify_cifar_loader_order.py
  --data-root data` failed because the script did not add the repository
  root to `sys.path`; after that fix it failed early with the intended
  “Complete CIFAR-10 cache required” message and attempted no download.
  `.\.venv\Scripts\pytest.exe -q -x
  tests/test_verify_cifar_loader_order.py` passed **1** bounded FakeData
  stochastic-view case with zero/two-worker forward/reverse order and
  sealed held-out roles. Initial mypy found the script under both a
  top-level and `scripts.` module name; adding `scripts/__init__.py`
  resolved this. Final `.\.venv\Scripts\ruff.exe check .` passed,
  `.\.venv\Scripts\mypy.exe src tests scripts` passed **118 files**, and
  `.\.venv\Scripts\ruff.exe format --check` on the three new Python
  files passed. `git -c core.safecrlf=false diff --check` passed after
  the final plan and documentation edits. The preceding P3.9b2b2a full
  gate remains **782/0
  skipped**, run before this standalone verifier was added.
- Artifacts and evidence: no dataset artifact was retained. New
  `scripts/verify_cifar_loader_order.py`, `scripts/__init__.py`, and
  `tests/test_verify_cifar_loader_order.py` provide a cache-gated local
  verifier; its fake-source pass validates the harness only. Source
  inspection confirms the loader uses the real CIFAR dataset class,
  seeded split indices and train sampler, separate deterministic
  guard/validation transforms, and disjoint role IDs. The documented
  strict-online contract is future work; the current continual runner is
  labeled offline. Its ordinary path creates phase-B roles after phase-A
  training; checkpoint preparation can pre-materialize both phases but
  does not claim strict-online label timing. This read-only audit is not
  evidence of real-CIFAR reproducibility.
- Plan changes: added P1.7e before the acquisition attempt to keep the
  actual-data gate explicit. Its acceptance remains unchanged and open.
  The attempted download changed no seeds, metrics, baseline settings,
  model order, or scientific result. No sweep or selection was launched.
- Skipped tests and blockers: the focused verifier test skipped none; the
  official download throughput did not meet the bounded local time
  budget; a complete trusted archive is needed to execute P1.7e. CUDA
  numerical and checkpoint evidence also require hardware unavailable
  on this host. P1.7e is unfinished, not marked complete or scientifically
  negative.
- Exact next action: obtain a complete trusted CIFAR-10 archive with a
  stated time/byte budget, then run
  `python scripts/verify_cifar_loader_order.py --data-root data` and add a
  bounded real-loader final-test seal. Keep P1.7/P1.8 and CUDA branches
  unchecked until their own gates pass.

## 2026-09-28 — P1.7e actual CIFAR-10 CPU loader and training seal

- Repository state: `master` at `704886b1e39159726271294c7475d9e4cdf6910e`;
  all earlier uncommitted research changes were preserved. The new dataset and
  JSON reports are under gitignored `data/`; no baseline, seed-selection,
  metric, or unrelated file was changed.
- Completed task IDs: **P1.7e**. P1.7 and P1.8 parents stay unchecked, as do
  actual CUDA and strict-online continual branches.
- Data acquisition and failures: torchvision's local `CIFAR10` declares
  `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` and MD5
  `c58f30108f718f92721af3b95e74349a`. After the previous official-source
  transfer stalled, the [Zenodo CIFAR-10 archive](https://zenodo.org/records/10089977)
  exposed the same 170,498,071-byte archive and MD5. Local `requests.head`
  checks returned HTTP 200/170,498,071 bytes for Zenodo, Brainchip, and Paddle;
  a bounded 1 MiB range probe favored Zenodo (2.61 s versus 2.89 s and
  15.44 s). A one-off eight-worker, 8 MiB-range downloader completed all
  170,498,071 bytes in 45.9 s, inside its 150 s wall budget. Whole-file MD5
  matched torchvision exactly before the archive was renamed to
  `data/cifar-10-python.tar.gz`. The first extraction guard rejected the
  archive's harmless root directory entry; a corrected path/member guard
  passed. Python 3.11 `tarfile.extractall(filter='data')` raised an unsupported
  keyword error; explicit regular-file/directory/path validation then safely
  extracted all nine members into `data/cifar-10-batches-py`.
- Experiment commands and outcome: `.\.venv\Scripts\python.exe
  scripts/verify_cifar_loader_order.py --data-root data --seed 73` exited 0.
  `.\.venv\Scripts\python.exe scripts/verify_cifar_loader_order.py
  --data-root data --seed 73 --check-training-seal >
  data\cifar-loader-seed73.json` exited 0. Both used
  `vision_guard_separated_seeded_unmatched_v3`, CPU, `download=False`, seed 73,
  8 train/4 guard/4 outer-validation/4 final-test examples, batch size 4,
  32-pixel images, and stochastic horizontal flip. Zero and two workers had
  identical role hashes and train IDs. Within **each** worker setting, all
  three forward and reversed model-order runs had exactly the same two
  shuffled batches and image/label tensor hashes. Zero versus two workers
  produced different stochastic view hashes, so cross-worker tensor identity
  is not asserted. All role ID sets were pairwise disjoint. The optional
  one-epoch real ResNet CPU pass used the random-feature control and opened
  the final-test loader six times only after backprop, PC, and circadian
  training completed; it emitted no model-ranking report. Final-test CIFAR
  dataset construction still occurs before training, while the training
  interface excludes that loader and no test batch is iterated before all
  models complete.
- Artifacts: ignored `data/cifar-10-python.tar.gz` (170,498,071 bytes),
  `data/cifar-10-batches-py/`, `data/cifar-loader-seed73.json` (2,295 bytes),
  and `data/cifar-loader-seed73-repeat.json` (1,860 bytes). The JSON retains exact
  train-batch IDs/view hashes, role hashes/counts, training order, three
  trained-model state hashes, and six post-training test iterations. Source
  code/test changes are `scripts/verify_cifar_loader_order.py` and
  `tests/test_verify_cifar_loader_order.py`; README and plan describe the
  optional real-training mode.
- Tests and static gates: the initial new seal unit test failed because its
  stub was not a dataclass and mypy rejected a statically typed stub call;
  both were corrected. The focused verifier file passed **2** tests.
  `.\.venv\Scripts\ruff.exe check .` passed;
  `.\.venv\Scripts\mypy.exe src tests scripts` passed **118 files**;
  `.\.venv\Scripts\ruff.exe format --check` on the two verifier Python
  files and `git -c core.safecrlf=false diff --check` passed.
  `.\.venv\Scripts\pytest.exe -ra` passed **785 in 229.97 s, 0 skipped**.
  A separate fresh-process command,
  `.\.venv\Scripts\python.exe scripts/verify_cifar_loader_order.py
  --data-root data --seed 73 > data\cifar-loader-seed73-repeat.json`,
  exited 0; comparing the entire loader report JSON with the training-seal
  run (excluding its extra training field) passed exact equality for both
  worker settings. No backend test was skipped in the full suite.
- Plan changes and limits: P1.7e is checked only after real-data loader and
  training-seal evidence. P1.7 remains open for actual CUDA numerical/order
  evidence; P1.8 remains open for larger-data matched-budget and real CUDA
  evidence. This check makes no fairness or performance claim. No model
  sweep, tuning, or final-test-based selection occurred. The local venv uses
  CPU-only Torch despite an NVIDIA card being present; a CUDA-enabled
  runtime and its own gates are required for CUDA claims. No tests were
  intentionally skipped in the focused run.
- Exact next action: inspect the shared-feature matched-head CIFAR path and
  write a predeclared, bounded CPU CIFAR-10 fairness confirmation task with
  equal validation trial counts, fixed seed(s), separate work/memory scopes,
  and final-test timing; then run its smallest validation-only gate before
  opening final test. Preserve all present archive and JSON evidence.

## 2026-09-28 — P1.8h, P1.8i1, P1.8i real-CIFAR matched budgets

- Repository state: same `master` commit
  `704886b1e39159726271294c7475d9e4cdf6910e`, with all earlier dirty
  work retained. New experiment JSON files are ignored under `artifacts/`
  and `data/`. No existing baseline, metric, seed, or final-test selection
  rule was changed after test access.
- Completed task IDs: **P1.8h** and **P1.8i1**; P1.8i completed after the
  unchanged-manifest retry and final gate recorded below. P1.8 parent remains
  unchecked for a justified larger-data comparison and actual CUDA evidence.
- Predeclaration and selection: added
  `scripts/run_cifar_matched_validation.py` for one local MD5-verified
  CIFAR-10 selection, seed 73, 32/16/16/16 train/guard/validation/test,
  one CPU epoch, frozen random ResNet features, fixed head width 16, and
  two equal optimization candidates per head (base rate and 0.8× rate).
  The sealed `test_loader` had zero iterations during six validation trials.
  The initial `.\.venv\Scripts\python.exe
  scripts/run_cifar_matched_validation.py` exited 1 only at manifest
  creation: `create_confirmation_manifest` requires 3–4 distinct
  confirmation seeds, while the request had 83/89. The first request and
  completed selection remain as
  `artifacts/benchmark_cifar_{request,selection}_smoke.json`; no final-test
  batch had been opened. We added seed 97 solely to satisfy that pre-existing
  invariant, before inspecting any test score, and versioned the output
  names. The identical selection command then exited 0 in 8.16 s total,
  reporting 5.42 s selection time. It saved request, selection, and manifest
  at `artifacts/benchmark_cifar_v2_{request,selection,manifest}_smoke.json`.
  All six attempts completed; two trials/head shared exact split, feature,
  backbone, and initial-head hashes, with 8 wake batches/32 samples each.
  PC/circadian each reported 16 relaxation steps; circadian's one guarded
  sleep raised guard exposures to 24 versus 8 for each baseline. Validation
  chose `a/a/a`. The saved manifest digest
  `eea8a7e0817b32684b0c0d778a0b4d1bbabf24f1aae7204f8bb6ec926b11fb02`
  was independently recomputed from JSON, and its saved selection digest
  matched. It fixes confirmation seeds 83/89/97, metrics accuracy and
  cross-entropy, fixed-data epoch, 0.05 s per-head wall time, and
  process-isolated fixed-width memory. No test score informed the choices.
- First exact-manifest confirmation: added a pure typed
  `restore_confirmation_manifest` with field and digest checks, plus
  `scripts/run_cifar_matched_confirmation.py`. A new test initially failed
  because restoration did not exist, then passed with JSON-list and
  tamper-rejection coverage. The first `.\.venv\Scripts\python.exe
  scripts/run_cifar_matched_confirmation.py` exited 1 after 15.19 s because
  `isolated_head_memory.py` rejected nonsynthetic data. Fixed-data and
  wall-time final-test computations had occurred in memory, but no complete
  result was saved; `artifacts/benchmark_cifar_v2_failure_smoke.json`
  retains the exact error. We did not alter the frozen manifest, candidate,
  seed, metric, scope, or baseline to address it.
- Process-isolated correction: an acceptance test for local CIFAR first
  failed at the synthetic-only guard. `isolated_head_memory.py` now keeps
  synthetic protocol v1 and allows only local CIFAR-10 with
  `dataset_download=False` and zero workers under distinct protocol
  `vision_three_head_fixed_width_cifar10_process_memory_v2`. Unsafe
  requests fail before spawn. `repeated_head_confirmation.py` checks the
  protocol appropriate to the declared dataset. The existing final-test
  property sentinel still forbids test-loader access in the measured
  worker; the parent verifies role, backbone, feature, initial-head, and
  fixed-capacity identity. `.\.venv\Scripts\python.exe
  scripts/verify_cifar_isolated_memory.py` exited 0 on seed 83, spawning
  three distinct PIDs, matching all development-role/hash/capacity fields,
  and writing `data/cifar-memory-seed83.json`. All three heads had 32,954
  unchanged parameters and 524,800 cached development-feature bytes;
  observed trainer RSS peaks were about 755 MB per child, including runtime,
  backbone, and cache.
- Completed exact-manifest retry: `.\.venv\Scripts\python.exe
  scripts/run_cifar_matched_confirmation.py` exited 0 in **58.91 s**, under
  the predeclared 240 s local limit. It preserved the first failure and
  wrote `artifacts/benchmark_cifar_v2_result_retry2_smoke.json` (199,864
  bytes). The result includes 9/9 fixed-data training/test rows, three
  wall-time runs (all nine heads reached `deadline`), and three three-child
  isolated-memory reports with matching per-seed identities and 32,954
  unchanged parameters. Fixed-data test-accuracy means across seeds
  83/89/97 were **0.125 backprop, 0.104 PC, 0.104 circadian**; wall-time
  means were **0.125, 0.146, 0.125**. This retained result shows no
  circadian advantage. One wall-time seed observed 148/85/40 wake batches
  and 0/170/80 relaxation steps for backprop/PC/circadian, illustrating why
  equal time is reported separately from equal epochs. Mean observed
  trainer RSS peaks were about 754.4/754.6/755.6 MB in that order; these
  process observations are not attributable head memory. With only 16
  final-test examples per seed and a random frozen backbone, accuracy is
  descriptive pipeline evidence, not a scientific ranking.
- Commands and quality gates: focused
  `.\.venv\Scripts\pytest.exe -q -x tests/test_matched_head_tuning.py
  tests/test_repeated_head_confirmation.py` passed **13** before the
  memory extension; `.\.venv\Scripts\pytest.exe -q -x
  tests/test_isolated_head_memory.py tests/test_repeated_head_confirmation.py`
  passed **10** after it. Initial failing red tests and the two experiment
  failures are retained above. `.\.venv\Scripts\ruff.exe check .`,
  `.\.venv\Scripts\mypy.exe src tests scripts` (**121 files**),
  `.\.venv\Scripts\ruff.exe format --check` on nine changed Python files,
  and `git -c core.safecrlf=false diff --check` passed.
  `.\.venv\Scripts\pytest.exe -ra` then passed **787 in 215.58 s,
  0 skipped**. No targeted test intentionally skipped a backend.
- Plan and documentation: split the actual-CIFAR fairness work into
  selection P1.8h, confirmation P1.8i, and the discovered isolated-memory
  capability P1.8i1. Added ADR-0062; README, evaluation protocols, and app
  module docs distinguish synthetic v1 from local-CIFAR v2. Parent P1.8
  remains open; P1.7 CUDA and strict-online continual work are also open.
- Exact next action: P1.8i is now checked after the full quality gate.
  Measure fixed-feature extraction throughput on the verified CIFAR cache,
  inspect local pretrained-backbone availability, then predeclare a
  larger-data CPU matched-head protocol with realistic role sizes and a
  local time limit before running any new final-test comparison. Do not
  adjust the saved 83/89/97 result or interpret this tiny random-feature
  run as a general benchmark.

## 2026-09-28 — P1.8j pretrained CIFAR development-feature cost gate

- Repository state: same `master` checkout at
  `704886b1e39159726271294c7475d9e4cdf6910e`; earlier working-tree
  changes and ignored experiment files remain intact.
- Completed task IDs: **P1.8j**. P1.8 parent, P1.7 CUDA evidence, and the
  unrelated strict-online continual study remain open.
- Commands and outcome: a read-only local check of
  `torchvision.models.ResNet50_Weights.IMAGENET1K_V2.url` and Torch's
  checkpoint cache found
  `resnet50-11ad3fa6.pth` (102,540,417 bytes) already present; no network
  request or new dependency was needed. The saved P1.8h selection ledger's
  six `train_seconds` values summed to 0.038 s, indicating that feature
  setup dominates this tiny route. Before execution, P1.8j fixed seed 101,
  128/64/64 train/guard/validation examples, 16-image batches, 32-pixel
  inputs, CPU, and a 60 s setup limit. `.\.venv\Scripts\python.exe
  scripts/profile_cifar_feature_setup.py` exited 0 and measured **1.875 s**
  for the full pretrained development-feature bank. A sealed test loader
  was never iterated; no head trained and no accuracy was computed.
- Artifacts: ignored `data/cifar-feature-profile-seed101-request.json`
  (5,069 bytes) was written before setup, and
  `data/cifar-feature-profile-seed101-result.json` (1,105 bytes) retains
  the 102,540,417-byte weight source and SHA-256
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`,
  backbone hash `be8306b3...e5ae9`, all three split/feature hashes,
  128/64/64 role counts, and 1,049,600/524,800/524,800 cached feature
  bytes. No failure artifact was created. `scripts/profile_cifar_feature_setup.py`
  is the reusable cache-gated, sealed development-only entry point.
- Validation and skipped tests: `.\.venv\Scripts\ruff.exe check .`,
  `.\.venv\Scripts\mypy.exe src tests scripts` (**122 files**),
  `.\.venv\Scripts\ruff.exe format --check
  scripts/profile_cifar_feature_setup.py`, and
  `git -c core.safecrlf=false diff --check` passed. The immediately prior
  full suite passed **787/0 skipped** after the core changes; this later
  standalone profiler changed no core behavior and was itself executed on
  real cached data. No test was intentionally skipped for this task.
- Plan changes and limits: P1.8j was added before the measurement and
  checked only after the sealed pretrained run and static gates. The
  1.875 s observation is a local setup cost for one small role bank, not
  a throughput guarantee for larger data or a fair model comparison.
  P1.8 remains unchecked; no model ranking, seed selection, metric change,
  or final-test access occurred in this step.
- Exact next action: use the saved pretrained setup time and hashes to
  predeclare a larger-data CIFAR matched-head CPU study with fixed role
  sizes, independent validation-selection and confirmation seeds, equal
  candidate counts, fixed-data/wall-time/isolated-memory scopes, and a
  local wall limit. Run its validation-only, final-test-sealed stage first;
  do not revise the prior seed-83/89/97 result based on new scores.

## 2026-09-28 — P1.8k/l frozen pretrained-CIFAR matched budgets

- Repository state: `master` at
  `704886b1e39159726271294c7475d9e4cdf6910e`. The existing dirty
  checkout and unrelated user changes were preserved. This increment adds
  two standalone scripts and ADR-0063, and updates the plan, README,
  evaluation protocol, and this log. No core learning rule or previous
  random-feature artifact changed.
- Completed task IDs: **P1.8k and P1.8l**. Parent P1.8, P1.7 CUDA evidence,
  strict-online continual timing, and CUDA checkpoint branches remain open.
  The 1.875 s no-score setup probe justified one bounded 1024/256/256/512
  train/guard/validation/test role study, not a larger sweep. Before any
  new final-test access, the plan fixed CPU, 32-pixel inputs, batch 32,
  one fixed-data epoch, frozen ImageNet ResNet-50 V2, width 16, selection
  seed 113, two equal base/0.8× learning-rate candidates per head,
  confirmation seeds 127/131/137, a 0.5 s per-head deadline, and separate
  process-isolated memory. Selection/confirmation local limits were
  120/480 s.
- Commands and selection outcome: `.\.venv\Scripts\ruff.exe format
  scripts/run_cifar_pretrained_validation.py` left the file unchanged;
  focused `.\.venv\Scripts\pytest.exe -q -x
  tests/test_matched_head_tuning.py tests/test_repeated_head_confirmation.py`
  passed 14/0 skipped. `.\.venv\Scripts\python.exe -c "import subprocess,
  sys; subprocess.run([sys.executable,
  'scripts/run_cifar_pretrained_validation.py'], check=True, timeout=125)"`
  exited 0. The script verified CIFAR archive MD5
  `c58f30108f718f92721af3b95e74349a` and the 102,540,417-byte
  ImageNet V2 checkpoint SHA-256
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`,
  wrote the request before training, and finished six equal validation
  trials in **3.89 s**. All trials saw 1,024 wake examples/32 batches and
  shared one train/guard/validation split, feature, backbone, and initial
  head hash. The circadian trials recorded a forced sleep attempt and
  additional guard work; no replay examples were used. The final-test
  sentinel observed **zero** iterations, and the selection held zero test
  confirmations. Outer validation selected `a/a/a`. Saved selection JSON
  matched the manifest's source digest; typed restoration preserved
  manifest digest
  `fd101a509411bfcfc2304e76b71edf9a3c47b7b44d7865204db0ab46d3a6fb70`.
- Confirmation preflight and outcome: a separate read-only preflight
  restored that manifest, checked the request/selection/selected configs,
  and rehashed both local inputs before any final-test read. Then
  `.\.venv\Scripts\python.exe -c "import subprocess, sys;
  subprocess.run([sys.executable,
  'scripts/run_cifar_pretrained_confirmation.py'], check=True,
  timeout=485)"` exited 0. The unchanged manifest completed in **100.11 s**
  with 9/9 fixed-data training/test rows, three wall-time reports with
  all nine heads stopping at `deadline`, and three three-child memory
  reports under `vision_three_head_fixed_width_cifar10_process_memory_v2`.
  An independent JSON audit found matching per-seed split/feature/backbone/
  initial hashes and unchanged capacity in every child, distinct PIDs,
  and observed trainer RSS samples. Deadline work differed as reported:
  across seeds backprop completed 32/40/41 epochs, PC 16/16/16, and
  circadian 9/12/9; latent relaxation, guard exposures, sleep attempts,
  examples, and overshoot remain in the JSON. Work is not inferred from
  equal wall time.
- Results retained without adjustment: one-epoch fixed-data mean accuracy
  was **0.4056 backprop, 0.1777 PC, 0.1523 circadian**; population SD was
  0.0526, 0.0412, 0.0380. Equal-wall-time means were **0.5788, 0.5072,
  0.3893**; population SD was 0.0171, 0.0179, 0.0254. The circadian head
  did not lead either scope. Per-seed scores, cross-entropy, work, and
  observed process RSS are retained. Process RSS includes runtime,
  backbone, feature bank, and sampler costs; it is not head-only memory.
  These 32-pixel, 1,024-example, CPU-only results do not establish a
  general head ranking. No seed, metric, baseline, or scope changed after
  final-test access.
- Experiment artifacts: ignored
  `artifacts/benchmark_cifar_pretrained_v1_request_smoke.json` (38,231
  bytes), `_selection_smoke.json` (77,735 bytes), `_manifest_smoke.json`
  (21,136 bytes), and `_result_smoke.json` (200,662 bytes). No failure
  artifact was created. The previous `benchmark_cifar_v2` request,
  failure, and result artifacts remain untouched. The scripts refuse
  overwrites and do not download data or weights.
- Validation and skipped tests: full `.\.venv\Scripts\pytest.exe -q`
  passed, with a separate collection hook confirming **787 tests** and
  no failure or skip markers. `.\.venv\Scripts\ruff.exe check .` passed;
  `.\.venv\Scripts\mypy.exe src tests scripts` passed on **124 files**;
  `.\.venv\Scripts\ruff.exe format --check
  scripts/run_cifar_pretrained_validation.py
  scripts/run_cifar_pretrained_confirmation.py` passed (**2 files**);
  `git -c core.safecrlf=false diff --check` passed. A broader
  `ruff format --check src tests scripts` failed because **43 pre-existing
  files** would be reformatted; no unrelated formatting was applied.
  No tests were intentionally skipped. No CUDA experiment or full-data
  sweep was run in this CPU session.
- Plan changes and blockers: P1.8k/l were added before selection, then
  checked only after their separate evidence gates. ADR-0063 and protocol
  docs record why this size and its limits. Read-only feasibility checks
  found NVIDIA GeForce RTX 3080 (10,240 MiB, driver 610.88) and ample local
  disk, but the verified `.venv` has `torch 2.14.0+cpu`, CUDA build `None`,
  and `torch.cuda.is_available()` false. Thus actual CUDA numerical,
  timing, and memory evidence is still blocked by the environment, not
  claimed from CPU results. P1.7/P1.8 stay unchecked.
- Exact next action: create an isolated CUDA-capable Torch environment
  without modifying the verified CPU `.venv`; run a tiny device smoke and
  then predeclare a device-specific order/reproducibility and matched-head
  manifest with numerical tolerance and a local time limit before any CUDA
  final-test access. Preserve this CPU negative result unchanged.

## 2026-09-28 — P1.7 CUDA isolation and P1.8m frozen CUDA selection

- Repository state: `master` still at
  `704886b1e39159726271294c7475d9e4cdf6910e`; the existing dirty
  checkout, previous CPU study artifacts, and unrelated user changes remain
  intact. This increment adds gitignored `.venv-cuda/`, four standalone
  scripts, ADR-0064, and protocol/README/plan updates. No core learning
  rule, previous selection, or prior negative score changed.
- Completed task IDs: **P1.7f, P1.7g, aggregate P1.7, and P1.8m**.
  **P1.8n and P1.8 parent remain unchecked.** Strict-online continual
  label timing and separate CUDA checkpoint branches also remain open.
- CUDA environment commands and outcome: `py -3.11 -m venv .venv-cuda` and
  `.\.venv-cuda\Scripts\python.exe -m pip install --no-input -r
  requirements.txt` exited 0. One 15-minute-capped pip attempt installed
  pinned `torch==2.14.0+cu130` (official 1,990.6 MB Windows wheel) and
  `torchvision==0.29.0+cu130` (6.4 MB) from
  `https://download.pytorch.org/whl/cu130`, without changing `.venv`.
  [PyTorch's 2.14 release](https://pytorch.org/blog/pytorch-2-14-release-blog/)
  lists CUDA 13.0 among its binary builds, and its
  [wheel index](https://download.pytorch.org/whl/cu130/torch/)
  contains the Python 3.11 Windows build. `pip check` found no broken
  requirements. The original `.venv` still reports Torch 2.14.0+cpu,
  torchvision 0.29.0+cpu, and CUDA unavailable.
- P1.7f smoke and retained failure: the first
  `.\.venv-cuda\Scripts\python.exe -c "import subprocess,sys;
  subprocess.run([sys.executable,'scripts/verify_cuda_environment.py'],
  check=True,timeout=65)"` failed before compute because
  `torch.cuda.reset_peak_memory_stats(device)` rejected an uninitialized
  device argument. The original request/failure JSON remains. A minimal
  CUDA context/device-0 diagnostic succeeded; the script then changed
  only allocator instrumentation and used versioned `retry1` artifact
  names with the same seed 109 and synthetic operations. Retry completed
  in **0.781 s** with finite convolution gradients and untrained ResNet-50
  logits, saved three SHA-256 output digests, and observed CUDA 13.0,
  cuDNN 92400, RTX 3080 capability 8.6, **112,169,984** peak allocated
  and **146,800,640** peak reserved bytes. No CIFAR role was accessed.
- P1.7g predeclaration and outcome: the plan fixed actual local CIFAR-10
  seed 149, 8/4/4/4 role sizes, zero workers, augmentation on, random
  frozen unmatched v3 backbones, one epoch, width 16, two inference
  steps, forced sleep, forward/reverse model orders, deterministic CUDA,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, exact state/role hashes, `1e-6`
  absolute metric tolerance, and 120 s per process before execution.
  `.\.venv-cuda\Scripts\python.exe scripts/verify_cuda_vision_order.py`
  exited 0: fresh forward/reverse CUDA processes took 3.594/3.453 s;
  train/guard/validation/test hashes and all three trained-model hashes
  matched exactly. All nine per-model validation accuracy, test accuracy,
  and final cross-entropy deltas were **0.0**. Each order opened final
  test six times only after all three models finished and made one forced
  circadian sleep attempt. Neither tiny CUDA order made a structural split;
  the prior CPU P1.7c/d gates cover noisy splits and prioritized replay.
  This is an unmatched reproducibility check, not a head ranking.
- P1.8m predeclaration and outcome: only after P1.7g passed, the plan fixed
  a separate frozen-shared-pretrained CUDA selection using the CPU study's
  1024/256/256/512 roles, batch 32, width 16, one epoch, equal base/0.8×
  learning-rate grids, seed 151, and confirmation seeds 157/163/167 with
  fixed-data, 0.5 s/head wall-time, and isolated-memory scopes. The
  120 s capped `.\.venv-cuda\Scripts\python.exe -c "import subprocess,
  sys; subprocess.run([sys.executable,
  'scripts/run_cifar_pretrained_cuda_validation.py'],check=True,
  timeout=125)"` exited 0. Six trials completed in **3.438 s**, shared
  train/guard/validation, feature, backbone, and initial-head hashes,
  and recorded 1,024 wake examples/32 batches each with separate
  relaxation, guard, and sleep work. The sealed final test had zero
  iterations, and selection contains zero test confirmations. Outer
  validation selected `a/a/a`. Request, selection, and typed-round-trip
  manifest were saved before any CUDA final-test access. The selection
  source digest matches the manifest's `source_selection_digest`; the
  manifest digest is
  `2965701512685175572f46b7499e8a28d462a32fe64a7f4f73d8c39a0e500062`.
- P1.8n preparation and deferred launch: the new confirmation script's
  `--preflight` restored that exact manifest and rehashed the archive and
  pretrained checkpoint with **zero** final-test reads. A first external
  three-reading observation found GPU utilization 24%/30%/27%. The formal
  480 s capped launch attempted the predeclared quiet gate and saved
  `artifacts/benchmark_cifar_pretrained_cuda_v1_deferred_20260929T022905846254Z_smoke.json`
  (UTC timestamp, 651 bytes) with utilization **30%/27%/37%** and free
  memory 5,867/5,861/5,785 MiB. It exited 2 by design before calling
  confirmation, with **zero** final-test iterations. No result or failure
  artifact exists. The selected candidates, seeds, metrics, wall budget,
  and manifest remain unchanged; no GPU matched-head score is claimed.
- Artifacts: ignored `data/cuda-env-seed109-{request,failure}.json` and
  `data/cuda-env-seed109-retry1-{request,result}.json` preserve the initial
  telemetry issue and successful device smoke. Ignored
  `data/cuda-vision-order-seed149-{request,forward,reverse,result}.json`
  preserve the actual CUDA raw order reports and comparison. Ignored
  `artifacts/benchmark_cifar_pretrained_cuda_v1_{request,selection,manifest}_smoke.json`
  are 38,378/77,742/21,140 bytes; the busy-GPU deferral is separate.
- Validation and skipped tests: focused CUDA
  `.\.venv-cuda\Scripts\pytest.exe -q -x tests/test_resnet50_benchmark.py`
  passed **32/0 skipped**. Full original CPU
  `.\.venv\Scripts\pytest.exe -q` passed, with collection confirming
  **787 tests** and no failure/skip markers. `.\.venv\Scripts\ruff.exe
  check .` passed; `.\.venv\Scripts\mypy.exe src tests scripts` passed
  on **128 files**; `ruff format --check` passed for all four new CUDA
  scripts; `git -c core.safecrlf=false diff --check` passed. The
  repository-wide formatter debt of 43 older files was not changed; that
  optional broad format check was not rerun. No full CUDA-environment
  suite or matched CUDA final-test experiment was run.
- Plan changes and limits: P1.7f/g and P1.8m were added before their
  respective device/model/selection runs and checked only after evidence.
  Aggregate P1.7 is now checked because the corrected CPU routes cover
  replay, noisy structure, data views, and model order, while the actual
  CIFAR CUDA reversal met exact state and declared numerical tolerance.
  ADR-0064 explains why CUDA runtime, order, validation selection, and
  wall-time confirmation are distinct gates. P1.8n and parent P1.8 stay
  unchecked; 32-pixel/limited-data and busy-GPU timing cannot support a
  general fairness ranking. Legacy v1/v2 reproduction streams and
  strict-online continual label timing remain separately scoped.
- Exact next action: when three GPU readings five seconds apart each show
  at most 10% utilization and at least 5 GiB free, run
  `.\.venv-cuda\Scripts\python.exe
  scripts/run_cifar_pretrained_cuda_confirmation.py` from the unchanged
  saved manifest under an external 485 s process cap, then audit all nine
  fixed-data, nine wall-time, and nine isolated-memory reports. While the
  GPU remains busy, implement the first strict-online continual phase-A
  label-access sentinel as a separate evaluation-isolation increment.

## 2026-09-28 — P1.3a continual future-phase label-arrival canary

- Repository state: `master` at
  `704886b1e39159726271294c7475d9e4cdf6910e`; the existing dirty
  checkout and all ignored experimental artifacts were preserved. A
  concurrent change to `.github/workflows/ci.yml` (adding Python 3.14 to
  its matrix) appeared during this increment; it was not edited here.
- Completed task ID: **P1.3a**. **P1.3b, P1.8n, and P1.8 parent remain
  unchecked.** This is a measured label-arrival gate, not a strict-online
  result or a change to the offline benchmark.
- Inspection: `_run_single_seed` constructs Phase B roles only after
  `_train_phase_a_models` returns. `_build_checkpoint_seed_data` constructs
  both phase roles before checkpointed training because v1's
  `continual_data_digest` binds them together. A training-helper-only
  sentinel would miss that early source access.
- Implemented `tests/test_continual_phase_label_arrival.py`: a Phase B source
  canary rejects construction before every Phase A model completes. Both
  ordinary model orders made two Phase A updates per model before the single
  Phase B source request. The same canary raised in the checkpoint route
  **before any Phase A training or checkpoint write**, retaining the
  negative access result as a passing expected-rejection assertion. The
  existing final-test sentinel test covers both ordinary model orders.
- Commands and outcomes: `.\.venv\Scripts\pytest.exe -q
  tests/test_continual_phase_label_arrival.py
  tests/test_continual_shift_benchmark.py
  tests/test_continual_checkpoint_resume.py` passed **34 tests**.
  `.\.venv\Scripts\pytest.exe -q` exited 0; collection counted **790 tests
  in 65 files**, with no skips in the run output. `.\.venv\Scripts\ruff.exe
  check .` passed. `.\.venv\Scripts\mypy.exe src tests scripts` passed on
  **129 files**. `ruff format --check` passed for the new canary and four
  CUDA scripts after formatting one line in the new test.
  `git -c core.safecrlf=false diff --check` passed. No CUDA full suite or
  P1.8n final-test confirmation was run in this increment.
- GPU condition: a read-only `nvidia-smi
  --query-gpu=utilization.gpu,memory.free,memory.used
  --format=csv,noheader,nounits` sample returned **71%** utilization and
  **4,307 MiB** free. It did not meet P1.8n's quiet gate, so its frozen
  manifest and prior deferral artifact remain untouched; no final-test
  batch was read.
- Artifacts: no new experiment JSON. The prior ignored CUDA validation,
  environment/order reports, and busy-GPU deferral remain at their recorded
  paths. This increment added only the source-controlled canary, ADR-0065,
  and protocol/plan/log documentation.
- Plan change and rationale: split a small P1.3a availability audit from
  P1.3b phase-specific checkpoint identity. The observed checkpoint
  prefetch requires a versioned data-digest and resume design; silently
  treating the existing corrected route as strict-online would weaken the
  label-arrival contract. The reviewed offline protocol and saved scores
  were not modified.
- Blockers: P1.8n needs three five-second-apart GPU readings at most 10%
  utilization and at least 5 GiB free. P1.3b needs a phase-A-only identity
  that preserves tamper detection and exact Phase A/B recovery; the current
  v1 checkpoint binds both phases before A. Neither blocks the other.
- Exact next action: while the GPU is busy, inspect
  `src/app/continual_checkpoint.py` and `_run_checkpointed_seeds` together,
  then add a failing Phase A checkpoint/resume canary that rejects Phase B
  source access before implementing a versioned phase-specific identity.
  Once P1.8n's separate quiet gate passes, run its unchanged saved CUDA
  manifest and audit all three frozen budget scopes.

## 2026-09-28 — P1.8n frozen CUDA matched-head confirmation

- Completed task ID: **P1.8n**. The parent **P1.8 remains unchecked** for
  representative scale and environment-limited timing interpretation.
  The earlier busy-GPU deferral and all frozen selection artifacts were
  preserved. No seed, candidate, metric, or wall budget changed.
- Preflight: `.\.venv-cuda\Scripts\python.exe
  scripts/run_cifar_pretrained_cuda_confirmation.py --preflight` exited 0,
  restored manifest digest
  `2965701512685175572f46b7499e8a28d462a32fe64a7f4f73d8c39a0e500062`,
  rehashed the saved archive and pretrained checkpoint, and reported zero
  final-test iterations. No result/failure file existed before launch.
- Launch: `.\.venv-cuda\Scripts\python.exe -c "import subprocess,sys;
  subprocess.run([sys.executable,
  'scripts/run_cifar_pretrained_cuda_confirmation.py'],check=True,
  timeout=485)"` exited 0. The three five-second-apart gate readings were
  **3%/3%/2% utilization** with **7,763/7,757/7,807 MiB free**. The frozen
  run completed in **98.172 s**, below the 480 s internal and 485 s
  external caps. Its post-run reading was 51% utilization and 6,442 MiB
  free. Background load during the run is therefore not ruled out for the
  0.5 s wall-time comparisons.
- Result: ignored
  `artifacts/benchmark_cifar_pretrained_cuda_v1_result_smoke.json` is
  **210,337 bytes**, SHA-256
  `41f4b4c1e430ec4561f2a9abbb339828c07e80eb804f0b0e1a176978cf83f9fe`.
  It retains 9/9 complete one-epoch fixed-data training/test rows, 9/9
  wall-time heads stopped at `deadline` with work/overshoot, and nine
  distinct process-memory child PIDs. Every child retained **32,954**
  head parameters and recorded CUDA allocator peaks and sampled train RSS
  separately. The prior ignored deferred artifact remains; no failure
  artifact was created.
- Independent audit: an initial inline Python assertion compared full
  fixed-data training-role hash dictionaries to wall-time reports and
  failed because only the latter include final-test hashes. The corrected
  audit compared train/guard/validation roles across all scopes, test
  hashes between confirmation and wall-time rows, and per-seed backbone,
  initial-head, selected candidate, trained-head/test row, capacity, PID,
  deadline, finite metric, GPU/RSS, quiet spacing, and manifest integrity
  invariants. It exited 0 with nine fixed rows, nine deadline heads, and
  nine distinct children. The first assertion was an audit-scope mistake,
  not an experiment failure; the raw result and manifest were untouched.
- Predeclared test-accuracy summaries: fixed-data means were **0.4492
  backprop, 0.1816 PC, 0.1595 circadian**; wall-time means were **0.5944,
  0.4095, 0.2708**. Per-seed values, population standard deviations,
  cross-entropy, work, and overshoot remain in the result. The circadian
  model did not lead either accuracy scope. These negative results were
  retained without retuning or seed changes.
- Tests and skipped tests: no new source-code test was required for this
  exact-manifest experiment. The preceding full CPU suite passed 790/0
  skipped, with Ruff and mypy passing; the independent result audit passed.
  A full CUDA pytest suite and a larger-scale CIFAR study were not run.
- Plan/documentation changes: checked P1.8n only after the independent
  audit; updated the plan, README, and evaluation protocol with the
  result and its environmental limits. P1.8 remains unchecked, since
  1,024 training images at 32 pixels and possible background GPU load do
  not prove a general fairness ranking.
- Exact next action: implement P1.3b's versioned phase-specific continual
  checkpoint identity, starting with a failing Phase A resume canary that
  rejects any Phase B source access. Preserve v1 offline recovery and
  final-test sealing. Separately predeclare a representative-scale P1.8
  follow-up only after its data and device budget is justified.

## 2026-09-28 — P1.3b versioned continual phase-arrival checkpoints

- Completed task ID: **P1.3b**. **P1.3c and P1.8 parent remain unchecked.**
  The new route is phase-source isolation within an offline schedule; it is
  not reported as a full strict-online experiment.
- Test-first evidence: the new Phase A interruption/resume canary initially
  failed with `Unknown continual benchmark protocol:
  continual_phase_arrival_v2`. A later final-test seal check caught a
  premature held-out hash at the Phase B boundary; production code now
  defers both final-test validation and hashing until both phases train.
  The existing v1 early-Phase-B canary remains a passing negative control.
- Implementation: opt-in `continual_phase_arrival_v2` writes format-2
  checkpoints. A fresh or resumed Phase A builds only Phase A roles and a
  namespaced train/validation digest. Phase B source construction and the
  combined development digest occur after Phase A training; its checkpoint
  carries only four arrived train/validation role hashes. Final-test hashes
  enter the completed report/checkpoint after Phase B training. The
  checkpoint training context exposes only training roles plus identity,
  never final-test objects. `hash_test=False` in the infra role builder
  preserves all default v1 hashing behavior. The new CLI protocol choice is
  opt-in; v1 and legacy defaults/checkpoints remain supported.
- Behavioral verification: the new test file exercises fresh Phase A and
  Phase A resume in both model orders with a future-Phase-B source canary;
  changed Phase A/B development roles fail before an update; both Phase A
  and Phase B interruptions resume to matching reports, baseline model
  states, circadian state/replay content, and sleep counters. A two-seed
  checkpoint resumes the second seed without retraining the first. A
  held-out sentinel rejects final-test role hashing/reads during either
  training phase and verifies the training context has no test objects.
  The old v1 checkpoint tests still pass. The corrected role test-hash
  comparison is delayed, while the generator itself still materializes
  Phase A test data as part of the seeded source.
- Commands and outcomes: `.\.venv\Scripts\python.exe -m pytest -q
  tests/test_continual_phase_label_arrival.py
  tests/test_continual_checkpoint_resume.py
  tests/test_continual_shift_benchmark.py
  tests/test_split_function_preservation.py -x` passed **44**. Full
  `.\.venv\Scripts\python.exe -m pytest -q` exited 0; collection counted
  **796 tests in 65 files**, with no skips in the run output.
  `.\.venv\Scripts\ruff.exe check .` passed, and
  `.\.venv\Scripts\python.exe -m mypy src tests scripts` passed on **129
  files**. `ruff format --check` passed for the five touched Python files;
  `git -c core.safecrlf=false diff --check` passed. A tiny local
  `.\.venv\Scripts\python.exe scripts/run_continual_shift_benchmark.py
  --protocol-id continual_phase_arrival_v2 --profile baseline --seeds 7
  --sample-count-phase-a 40 --sample-count-phase-b 40 --phase-a-epochs 2
  --phase-b-epochs 2 --hidden-dim 4 --hidden-dims 4
  --sleep-interval-phase-a 0 --sleep-interval-phase-b 0` smoke exited 0
  and reported the v2 protocol. Its scores are descriptive and were not
  used to choose settings.
- Artifacts and skipped work: no new experiment JSON or large sweep. Pytest
  used temporary local checkpoint files only. The earlier 210,337-byte
  P1.8n CUDA result and busy-GPU deferral remain untouched. A full CUDA
  pytest suite and a representative-scale fairness run were not repeated
  for this NumPy checkpoint change.
- Documentation/plan changes: added ADR-0066, updated the README,
  `docs/modules/app.md`, and evaluation protocol table. Checked P1.3b only
  after source-arrival, tamper, continuation, held-out, CLI, and quality
  gates. Added unchecked P1.3c to preserve the missing strict-online
  requirements: Phase A schedule cannot depend on the future B horizon,
  replay needs a declared count/byte budget, and guard/outer-selection
  arrival plus label/retention reporting need an explicit contract. The
  reviewed v1 and new v2 outputs are versioned separately.
- Repository state: the branch remains at
  `704886b1e39159726271294c7475d9e4cdf6910e`, with prior dirty work
  preserved. Concurrent unrelated edits in `.github/workflows/ci.yml` and
  `requirements.txt` were not modified here.
- Exact next action: split P1.3c into an initial train-only schedule gate,
  then add a failing test for an opt-in strict-online route showing that a
  change to the future Phase B duration cannot change Phase A sleep
  decisions or trained state. Preserve v1/v2 offline results, and leave
  P1.3c unchecked until replay, guard, label-arrival, reporting, and
  bounded confirmation criteria all pass.

## 2026-09-28 — Python 3.14 environment upgrade

- Installed standalone Python 3.14.7 from the Python Software Foundation
  with WinGet. The Python launcher now defaults to 3.14.7 and a normal PATH
  resolves `python` to that install. Python 3.11 and 3.13 remain installed so
  older environments can still be reproduced.
- Promoted Python 3.14.7 environments to `.venv` (CPU Torch 2.14.0),
  `gpu-venv` (CUDA 13.0 Torch 2.14.0), and `.venv-cuda` (same CUDA package
  set). Moved the original three Python 3.11 environments to ignored
  `*-py311-snapshot` folders; their interpreters and package sets remain
  usable with direct `python.exe -m ...` invocations.
- Added Python 3.14 to the CI matrix. Limited NumPy to `<2.5` because NumPy
  2.5.3's stubs use Python 3.12 type-alias syntax, which failed the existing
  mypy Python 3.11 target. NumPy 2.4.6 has CPython 3.14 wheels and passes the
  same type check.
- Validation: `pip check` passed in all three active environments; CPU and
  CUDA environments have identical non-Torch package inventories; a CUDA
  convolution forward/backward produced finite gradients on the RTX 3080.
  Ruff passed, mypy passed on 129 files, and the stable Python 3.14 CPU run
  passed **790 tests, 0 skipped, in 253.07 s**. An earlier run overlapped
  replacing NumPy and had one stochastic-loader failure; the full rerun after
  package installation completed passed.

## 2026-09-28 — session handoff after P1.8n and P1.3b

- Completed task IDs this session: **P1.8n** (frozen matched CUDA
  confirmation) and **P1.3b** (opt-in phase-arrival checkpoint identity).
  Full **P1.8** and **P1.3c** remain unchecked.
- Final commands/outcomes: `.\.venv\Scripts\python.exe -m pytest -q`
  exited 0 with **796 collected tests in 65 files** and no skips shown;
  `.\.venv\Scripts\ruff.exe check .`, `.\.venv\Scripts\python.exe -m mypy
  src tests scripts` (129 files), touched-file `ruff format --check`, and
  `git -c core.safecrlf=false diff --check` passed. The v2 tiny CLI smoke
  and the independent nine-row/nine-head/nine-child CUDA JSON audit passed.
  The active `.venv` and `.venv-cuda` interpreters currently report Python
  **3.14.7**; the separately recorded environment upgrade was preserved.
- Experiment artifacts: ignored
  `artifacts/benchmark_cifar_pretrained_cuda_v1_result_smoke.json`
  (210,337 bytes, SHA-256
  `41f4b4c1e430ec4561f2a9abbb339828c07e80eb804f0b0e1a176978cf83f9fe`)
  and the earlier busy-GPU deferral. No P1.3b score artifact or sweep was
  created. The CUDA result retains the negative circadian accuracy result;
  no metric, seed, or selected candidate changed.
- Skipped work and limits: no full CUDA pytest suite, no representative
  image-scale fairness run, and no strict-online result. The v2 phase
  isolation route still uses the known full A+B sleep schedule and lacks
  replay/guard/selection arrival reporting. Post-confirmation GPU load was
  51%, limiting interpretation of wall-time results.
- Plan changes: checked P1.8n and P1.3b on their acceptance evidence;
  added unchecked P1.3c for the full strict-online contract. Parent P1.8
  remains unchecked for representative scale. Existing dirty work,
  including the separate Python 3.14 CI/dependency changes, was preserved.
- Exact next action: add an initial P1.3c schedule-isolation subtask and a
  failing test for an opt-in strict-online route proving that changing only
  the future Phase B epoch count cannot alter any Phase A sleep decision or
  trained Phase A state. Keep v1/v2 offline behavior intact, then continue
  with declared replay memory, label-arrival/guard roles, and reporting.

## 2026-09-28 — P1.3c1 Phase A schedule isolation

- Completed task ID: **P1.3c1**. Parent **P1.3c** and new substeps
  **P1.3c2–c4** remain unchecked. The original P1.3c criteria were not
  weakened. Read `AGENTS.md`, the full living plan, and this log; reconciled
  the dirty `master` checkout at `704886b1e39159726271294c7475d9e4cdf6910e`
  with the prior P1.8n/P1.3b handoff. Preserved unrelated Python 3.14
  CI/dependency edits and all prior user work.
- Why this increment: both ordinary and checkpointed Phase A trainers passed
  `phase_a_epochs + phase_b_epochs` to progress-sensitive sleep. Core sleep
  uses completed/total progress to choose split/prune budgets. A tiny fixed
  seed-17, two-epoch A fixture with forced sleep exposed the real effect:
  under v2, B durations 1 and 7 made A sleep totals 3 and 9; the second A
  sleep made zero versus one split, with different trained state. This is a
  negative isolation result for v2, retained without changing its behavior.
- Added opt-in `continual_phase_local_schedule_v3` and checkpoint format 3.
  Its Phase A sleep receives only the A horizon in ordinary and checkpointed
  paths. It inherits v2's phase-arrival role boundary; B remains available
  only after A, when its own A+B sleep horizon is known. The v1/v2 defaults,
  schedule, data flow, and checkpoint format remain unchanged. ADR-0067,
  README, app-module docs, and the evaluation-protocol table describe the
  limited v3 claim. V3 is not labeled a full strict-online study.
- Test-first and validation commands/outcomes: the new focused file first
  failed four v3 cases on unknown protocol (the two v2 controls passed).
  After implementation, `python -m pytest tests/test_continual_phase_local_schedule.py
  tests/test_continual_phase_label_arrival.py tests/test_continual_checkpoint_resume.py
  tests/test_continual_shift_benchmark.py -q` passed **48/48**. The v3
  fixture retained identical A sleep progress, structural indices, and
  hashes of all three trained models across B durations 1/7 in both model
  orders and both ordinary/checkpointed routes. Interrupting at the final
  A `after_sleep` file checkpoint confirmed format 3, only A train/validation
  hashes, no Phase B source construction, and exact continuation versus an
  uninterrupted checkpointed run; final reports also matched ordinary
  execution. The v2 control retained its full-horizon sensitivity.
- Full quality gate: `.\.venv\Scripts\python.exe -m pytest -q` exited 0;
  collection counted **804 tests**, with no skips shown. `ruff check .`,
  `python -m mypy src tests scripts` (**130 files**), touched-file
  `ruff format --check`, and `git -c core.safecrlf=false diff --check` all
  passed. The tiny v3 CLI smoke (seed 17, 40 examples/phase, A2/B1) exited
  0 and printed the v3 protocol. Its backprop/PC/circadian balanced scores
  were 1.000/0.750/0.000; these are smoke outputs, not a selected ranking.
- Experiment artifacts: no new persistent score artifact or sweep. Tests
  used temporary trusted checkpoint files only. The earlier frozen CUDA
  confirmation JSON and its negative circadian outcome remain untouched.
  Skipped a full CUDA pytest run and representative-scale fairness study;
  neither is needed to validate this NumPy schedule boundary. No seeds,
  metrics, or baseline tuning were changed to favor circadian.
- Plan changes: split P1.3c into c1 schedule isolation, c2 replay budget,
  c3 guard/role/label arrival, and c4 full bounded confirmation. Checked c1
  only after the adversarial test, checkpoint continuation, focused/full
  tests, static checks, and CLI smoke. The v3 config/checkpoint identity
  still knows future B settings, and retained replay IDs/bytes and
  guard/selection arrival are unreported. These are open P1.3c2–c4 work,
  not accepted strict-online evidence. P1.8 representative-scale fairness
  and the separate P3.9 CUDA branches also remain open.
- Exact next action: inspect NumPy circadian replay capture/selection and
  continual training interfaces. Add a failing v3 test that prevents
  replay from selecting unobserved/future-phase examples and enforces a
  declared retained-example count and byte cap at A→B and after checkpoint
  resume. Keep v1/v2 replay unchanged, then implement P1.3c2.

## 2026-09-28 — P1.3c2 observed-example replay budget

- Completed task ID: **P1.3c2**. Parent **P1.3c** and substeps **P1.3c3–c4**
  remain unchecked. Rechecked the dirty `master` checkout at
  `704886b1e39159726271294c7475d9e4cdf6910e`, the current plan/log,
  and the actual replay/checkpoint code. All unrelated user changes,
  including Python 3.14 environment files and the frozen CUDA result,
  were preserved.
- Starting finding: NumPy `_store_replay_snapshot` copied the entire wake
  training batch into each deque entry. `replay_memory_size` therefore
  capped batches, not examples or bytes. There was no retained-ID or
  phase-boundary memory report. The first test file failed import on the
  missing bounded route, then passed after implementation.
- Added opt-in `continual_bounded_replay_v4` with a separate dataclass
  config/result and checkpoint format 4. V1/v2/v3 dataclass shapes,
  config digests, and historical whole-batch replay remain unchanged.
  V4 requires explicit positive example/byte caps, enabled replay, and
  component sleep. The NumPy core stores unique observed labeled rows
  by the smallest stable SHA-256 content hashes, limited by both caps.
  The report records declared caps and actual retained IDs/count/array
  bytes after A and after B. Bytes cover copied NumPy input and target
  arrays, excluding Python/deque and checkpoint overhead. ADR-0068,
  README, app/core-module docs, and the evaluation protocol explain the
  deterministic selection rule and possible uneven phase mix.
- Validation: `python -m pytest tests/test_continual_bounded_replay.py -q`
  passed **13/13**. Independent fixtures made the example limit bind at
  4/240 bytes and the byte limit bind at 10/48 bytes; legacy core replay
  still retained one whole batch. Instrumented sleep selection in both
  model orders used only arrived training content IDs. Ordinary and A/B
  resumed checkpoint reports matched, including retained IDs/count/bytes.
  Format-4 preflight rejected a future B train row in Phase A, a B
  validation row in the active B buffer, and a B row in frozen A state,
  all before another backprop update. A two-seed resume rejected a forged
  prior-seed retained-ID report and recovered when the original file was
  used. Phase A/B source arrival and final-test sealing inherit the
  previously tested v2/v3 path.
- A first tiny v4 CLI diagnostic used the legacy sleep mode and reported
  zero performed sleeps despite retained examples. That exposed a
  misleading configuration; v4 now requires explicit component sleep.
  The corrected tiny CLI smoke (`--profile strength-case --protocol-id
  continual_bounded_replay_v4 --sleep-mode components
  --replay-max-examples 4 --replay-max-bytes 96 --seeds 17`, with 40
  samples/phase and A2/B2) exited 0, performed four component sleeps,
  and reported 4 examples/96 bytes at both boundaries. Its descriptive
  balanced scores were backprop 1.000, PC 0.750, circadian 0.000;
  this negative tiny outcome was retained without selecting a seed,
  tuning a baseline, or calling it a strict-online ranking.
- Full quality gate: `.\.venv\Scripts\python.exe -m pytest -q` exited 0;
  collection counted **817 tests**, with no skips shown. `ruff check .`,
  `python -m mypy src tests scripts` (**131 files**), touched-file
  `ruff format --check`, and `git -c core.safecrlf=false diff --check`
  passed. The focused continual/core set also passed before the full run.
  No full CUDA pytest run or representative-scale fairness study was
  attempted for this NumPy replay increment.
- Experiment artifacts: no persistent new result or sweep. Tests used
  temporary trusted checkpoint files; the CLI smoke printed to stdout.
  The earlier CUDA JSON and negative result were not modified.
- Plan changes: checked P1.3c2 only after independent caps, provenance,
  checkpoint/resume, reporting, CLI, and full quality gates. V4 remains
  partial: inner guard/outer selection arrival, source/label timing and
  task-information ledger, final-test perturbation/state invariance, and
  bounded A→B confirmation remain under P1.3c3–c4 and parent c.
  P1.8 representative-scale fairness and separate P3.9 CUDA branches
  also remain open; no new external blocker was found.
- Exact next action: inspect v4 role construction and scoring boundaries,
  then add a failing sentinel test that identifies which Phase A/B labels
  and guard/outer-selection roles are visible at each decision point.
  Introduce disjoint, arrival-stamped roles and a final-test label
  perturbation/state check without changing v1/v2/v3 offline outputs.

## 2026-09-28 — P1.3c3a final-test label seal

- Completed task ID: **P1.3c3a**. Parent **P1.3c3** and substep
  **P1.3c3b**, **P1.3c4**, and parent **P1.3c** remain unchecked.
  Continued on dirty `master` at `704886b1e39159726271294c7475d9e4cdf6910e`;
  preserved unrelated user changes, including the Python 3.14 environment
  files and existing CUDA experiment records.
- Inspection found that checkpointed v4 already requested `hash_test=False`
  during Phase A/B role construction, but ordinary v4 used the splitter
  default and hashed held-out labels before training. A new raising role and
  hash sentinel failed in both ordinary model orders before Phase A updates;
  both checkpointed orders passed. This was a real access-timing gap even
  though the prior label-perturbation check showed no state change.
- Ordinary v4 now constructs A/B roles without held-out hashes, then binds
  both test hashes after all three models finish Phase B. V1/v2/v3 ordinary
  behavior and checkpointed behavior retain their prior routes. New tests
  exercise ordinary/checkpointed execution in both model orders: the final
  role raises on early input or label access, test hashing raises until B
  training finishes, and both hashes appear at reporting. Perturbing only
  the two final-test target arrays changes their hashes and backprop test
  accuracy, while six final/frozen trained-state hashes, sleep progress and
  structural decisions, replay selections, and four development-role hashes
  remain identical. Replay selection was nonempty in the fixture.
- Commands and outcomes from the workspace root (using `.venv` executables):

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_final_label_seal.py tests/test_continual_bounded_replay.py -q
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_final_label_seal.py tests/test_continual_bounded_replay.py tests/test_continual_phase_local_schedule.py tests/test_continual_phase_label_arrival.py tests/test_continual_checkpoint_resume.py tests/test_continual_shift_benchmark.py -q
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe format --check src/app/continual_shift_benchmark.py tests/test_continual_final_label_seal.py
  git -c core.safecrlf=false diff --check
  .\.venv\Scripts\python.exe scripts/run_continual_shift_benchmark.py --profile strength-case --protocol-id continual_bounded_replay_v4 --sleep-mode components --replay-max-examples 4 --replay-max-bytes 96 --seeds 17 --sample-count-phase-a 40 --sample-count-phase-b 40 --phase-a-epochs 2 --phase-b-epochs 2 --sleep-interval-phase-a 1 --sleep-interval-phase-b 1
  ```

  The two focused test commands passed **21** and **69**. Full pytest
  exited 0; collection counted **825 tests**, with no skips shown. Ruff,
  mypy (**132 source files**), touched-file format, and diff checks passed.
  The recorded tiny v4 CLI invocation exited 0; four component sleeps
  retained 4/96 at both boundaries. Its descriptive balanced scores were
  backprop 0.850, PC 0.600, circadian 0.450. This invocation used the
  printed strength-case defaults apart from the listed small-run flags;
  it does not replace the separate P1.3c2 smoke or constitute a ranking.
  No seed, baseline, or metric was changed to improve the negative result.
- Skipped work and artifacts: no new persistent experiment artifact or
  sweep; the CLI report was stdout and test checkpoints used temporary
  directories. No full CUDA pytest run or representative-scale fairness
  study was attempted for this NumPy access-timing increment. Prior CUDA
  result and busy-GPU deferral artifacts were untouched.
- Plan changes and rationale: split P1.3c3 into c3a (final-test access seal)
  and c3b (physical role/label arrival and decision boundaries) because the
  ordinary early hash could be fixed and verified independently. Checked
  c3a only after its red/green sentinel, perturbation, focused/full/static
  gates, and smoke. ADR-0069, README, app-module docs, and evaluation
  protocol explain the narrower result. Subsequent inspection found that
  the source generator still creates held-out labels and the splitter
  carries their array reference before training. The first c3a wording
  overclaimed “every read”; its recorded acceptance is now explicitly the
  runner's role validation/hash/score boundary, while physical release is
  an unchanged parent requirement assigned to open c3b. V4 also has no
  disjoint inner guard/outer selection arrival ledger or declared task
  information. These are unfinished c3b acceptance criteria, not accepted
  strict-online evidence. P1.8 representative-scale fairness and P3.9 CUDA
  branches remain separate open work; no new external blocker was found.
- Exact next action: inspect v4 phase source generation and role
  construction, add a failing source/label sentinel plus disjoint
  train/inner-guard/outer-selection role test with stable IDs/hashes and
  recorded arrival, then implement the versioned P1.3c3b contract while
  preserving the v4 final-test seal and v1/v2/v3 offline outputs.

## 2026-09-28 — P1.3c3b1 source-field release

- Completed task ID: **P1.3c3b1**. Parent **P1.3c3b**, substep
  **P1.3c3b2**, **P1.3c3**, **P1.3c4**, and **P1.3c** remain unchecked.
  Worked on dirty `master` at `704886b1e39159726271294c7475d9e4cdf6910e`;
  unrelated user changes and earlier CUDA artifacts were preserved.
- Inspection found that `split_training_validation(hash_test=False)` still
  read `DatasetSplit.test_input` and `test_target` to package a held-out
  role. A new source-level sentinel raised on the first test-input field
  request before Phase A training in **all four** ordinary/checkpointed ×
  model-order v4 cases. This was a distinct gap from the runner-level hash
  and role seal closed in P1.3c3a.
- Added `DeferredFinalTestRole` in `src/infra/datasets.py` and an explicit
  `defer_test_access` splitter option, valid only with `hash_test=False`.
  V4 A/B source splitters request it in both paths; v1/v2/v3 keep the
  previous default. The reference resolves source test input and target
  only when final hashing/scoring occurs after each seed's B training.
  The sentinel now passes in both paths/orders and separately counts
  post-training input and label reads. Existing v4 A/B interruption and
  resume checks still match ordinary/uninterrupted results. The exact
  small CLI smoke repeated its prior balanced scores (backprop 0.850,
  PC 0.600, circadian 0.450) and 4-example/96-array-byte retention at
  both boundaries with four component sleeps. No seed, metric, or baseline
  was changed to force a circadian advantage.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_final_label_seal.py -q -k releases_source
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_final_label_seal.py tests/test_continual_bounded_replay.py tests/test_dataset_roles.py -q
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe format --check src/infra/datasets.py src/app/continual_shift_benchmark.py tests/test_continual_final_label_seal.py
  git -c core.safecrlf=false diff --check
  .\.venv\Scripts\python.exe scripts/run_continual_shift_benchmark.py --profile strength-case --protocol-id continual_bounded_replay_v4 --sleep-mode components --replay-max-examples 4 --replay-max-bytes 96 --seeds 17 --sample-count-phase-a 40 --sample-count-phase-b 40 --phase-a-epochs 2 --phase-b-epochs 2 --sleep-interval-phase-a 1 --sleep-interval-phase-b 1
  ```

  The first command failed **4/4** before implementation at the intended
  source field. The focused three-file command passed **28** after the
  fix. Full pytest exited 0; collection counted **829 tests**, with no
  skips shown. Ruff, mypy (**132 files**), touched-file format, and diff
  checks passed. The final source sentinel assertion was strengthened to
  count inputs and labels independently, and the focused suite was rerun
  green before the final full/static gates.
- Experiment artifacts and skipped work: no new persistent result or
  sweep; the CLI report was stdout and test checkpoints were temporary.
  No full CUDA suite or representative-scale fairness experiment was run
  for this NumPy source boundary.
- Plan changes and rationale: split c3b into checked b1 (per-seed source
  field release) and open b2 (global setting freeze, disjoint inner guard
  and outer selection, IDs/hashes, label/task ledger). ADR-0070, README,
  evaluation protocol, and app/infra module docs describe the opt-in
  reference and its scope. The generator still allocates test arrays at
  phase construction; a completed seed is scored before later seeds
  train. Thus the parent final-test and role-arrival criteria remain open.
  No external blocker was found.
- Exact next action: add a two-seed source/label sentinel that raises if
  seed 17's final test is opened before seed 19 and all settings finish;
  record the current failure, then design a versioned run-level freeze
  with disjoint arriving train/inner-guard/outer-selection roles and
  explicit IDs, hashes, label times, and per-method task information.

## 2026-09-28 — P1.3c3b2a all-seed ordinary final-test seal

- Completed task ID: **P1.3c3b2a**. **P1.3c3b2b–c**, parent **P1.3c3b2**,
  **P1.3c3b**, **P1.3c3**, **P1.3c4**, and **P1.3c** remain unchecked.
  Continued on dirty `master` at `704886b1e39159726271294c7475d9e4cdf6910e`;
  unrelated Python 3.14, CUDA, and other user changes were preserved.
- Starting evidence: a two-seed source sentinel showed v4 opens seed 17's
  final-test input while seed 19 has not begun training. The v5 tests
  initially failed because the new config/route did not exist; after
  implementation, the same source sentinel passed in both model orders.
- Added opt-in `continual_global_test_seal_v5` with a distinct frozen
  config. The ordinary runner retains each seed's trained A/B models and
  deferred final-test roles, then binds hashes and scores only after every
  configured seed finishes. It inherits v4's phase-local sleep and
  observed-example replay caps. A v5 checkpoint request fails before
  source loading, since v4 format-4 `seed_complete` stores scored results
  and cannot represent pending unscored trained seeds. V1–v4 paths remain
  available; a fixed two-seed fixture produced exactly equal v4/v5
  per-seed reports and aggregate in both model orders. Perturbing only
  seed 17's final labels changed its test hash but left both serialized
  trained seed states and seed 19's entire report identical.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_global_test_seal.py -q
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_global_test_seal.py tests/test_continual_final_label_seal.py tests/test_continual_bounded_replay.py tests/test_continual_phase_local_schedule.py tests/test_continual_phase_label_arrival.py tests/test_continual_checkpoint_resume.py tests/test_continual_shift_benchmark.py -q
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe format --check src/app/continual_shift_benchmark.py scripts/run_continual_shift_benchmark.py tests/test_continual_global_test_seal.py
  git -c core.safecrlf=false diff --check
  .\.venv\Scripts\python.exe scripts/run_continual_shift_benchmark.py --profile strength-case --protocol-id continual_global_test_seal_v5 --sleep-mode components --replay-max-examples 4 --replay-max-bytes 96 --seeds 17,19 --sample-count-phase-a 40 --sample-count-phase-b 40 --phase-a-epochs 2 --phase-b-epochs 2 --sleep-interval-phase-a 1 --sleep-interval-phase-b 1 --hidden-dim 4
  ```

  The first v5 test run had **2 failures** on the absent config while its
  v4 negative canary passed. After implementation, the v5 test file
  passed **8**, the seven-file continual set passed **81**, and the
  full CPU suite passed **837** with zero skips shown. Ruff, mypy
  (**133 source files**), touched-file format, and diff checks passed.
  The tiny two-seed CLI exited 0 with four sleeps per seed and 4/96
  retained at both boundaries. Descriptive balanced means were backprop
  0.575, PC 0.650, circadian 0.250; the negative circadian result was
  retained without selecting seeds, retuning baselines, or changing metrics.
- Experiment artifacts and skipped work: no new persistent output or
  sweep; the CLI report was stdout and checkpoints in tests were temporary.
  Full CUDA pytest and representative-scale fairness work were not run for
  this NumPy access boundary. Earlier artifacts were untouched.
- Plan changes and rationale: split P1.3c3b2 into checked b2a (ordinary
  all-seed access seal), open b2b (versioned unscored-state checkpoint),
  and open b2c (disjoint arrived guard/outer roles, setting selection, IDs,
  hashes, label/task timing). ADR-0071, README, app docs, and evaluation
  protocol document the new route and pending-model memory cost. A single
  frozen config across seeds is narrower than all candidate settings;
  b2/b/c3/c parent criteria are unchanged. No external blocker was found.
- Exact next action: inspect v4's `seed_complete` transaction and define
  a separate v5 checkpoint format storing unscored completed seed states
  and development digests without final-test hashes. Add two-seed A/B and
  next-seed interruption/resume sentinels that keep seed 17's final labels
  sealed until seed 19 finishes, and reject tampered development state
  before another update.

## 2026-09-28 — P1.3c3b2b unscored v5 checkpoint continuation

- Completed task ID: **P1.3c3b2b**. **P1.3c3b2c**, parent **P1.3c3b2**,
  **P1.3c3b**, **P1.3c3**, **P1.3c4**, and **P1.3c** remain unchecked.
  Started on dirty `master` at `704886b1e39159726271294c7475d9e4cdf6910e`.
  During the session the shared checkout moved to
  `codex/research-protocols-and-resume`: code/docs entered `e1fec1b`,
  followed by the plan in `15dcf81`. This log entry is the remaining
  working-tree change. Unrelated Python 3.14, CUDA, and other user
  changes were preserved.
- Starting canary: the new v5 checkpoint source-seal test failed twice at
  the old explicit rejection. Added format-5 `unscored_seeds` records with
  detached trained baseline/circadian states and arrived development
  digests/hashes. Every active and terminal v5 checkpoint leaves completed
  report, development-result, and test-digest fields empty; it never stores
  test hashes, scores, or held-out arrays. Resume validates prior seed
  identities, model progress, circadian wake/replay provenance, and the
  active phase before another update. The terminal unscored checkpoint
  repeats scoring without retraining. V1–v4 retain their existing payload
  and checkpoint routes.
- Evidence: two-seed source-field sentinels keep final tests closed until
  both seeds train in either model order. Sixteen A/B and later-seed
  interruption/resume cases cover wake, before-sleep, after-sleep, seed
  boundaries, and terminal completion. Each gives the ordinary v5 report,
  runs only remaining Backprop updates, and matches uninterrupted saved
  model fields and replay rows. Whole-object pickle hashes initially
  differed in eight cases because object alias encoding changed across
  restore; field-wise model/RNG/snapshot and row-wise replay comparisons
  showed equal training state. Two prior-state tamper cases, one changed
  development role, one forged future-seed replay, and two changed
  setting/seed-list cases all reject before another update or source read.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_global_test_seal.py -q -k checkpoint_resume
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_global_test_seal.py -q
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_global_test_seal.py tests/test_continual_final_label_seal.py tests/test_continual_bounded_replay.py tests/test_continual_phase_local_schedule.py tests/test_continual_phase_label_arrival.py tests/test_continual_checkpoint_resume.py tests/test_continual_shift_benchmark.py -q
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe format --check src/app/continual_checkpoint.py src/app/continual_shift_benchmark.py tests/test_continual_global_test_seal.py
  git -c core.safecrlf=false diff --check
  ```

  The expanded interruption test passed **16** after semantic state
  comparison; the v5 test file passed **31**. The seven-file continual set
  passed **104**, and the final full CPU suite exited 0 with **860**
  collected tests and no skips shown. Ruff, mypy (**133 source files**),
  touched-file format, and diff checks passed. An initial format check
  found one layout change; `ruff format src/app/continual_shift_benchmark.py`
  fixed it before the final gates. The initial whole-object pickle-hash
  test failed **8/12** before comparison was corrected without changing
  the training implementation or acceptance target.
- Experiment artifacts and skipped work: no new persistent experiment
  result or sweep; checkpoint files were temporary pytest artifacts and
  one temporary diagnostic comparison. No full CUDA suite or
  representative-scale fairness experiment was run for this NumPy
  checkpoint boundary. The CLI has no continual checkpoint flag, so the
  documented Python API was used. Earlier negative circadian results and
  their artifacts were untouched.
- Plan changes and rationale: checked b2b only after the full resume and
  static gates. ADR-0072, README, evaluation protocol, and app/infra
  module docs describe format 5 and its trusted-file boundary; ADR-0071
  now points to the later checkpoint increment. A single declared config
  is bound by digest, but no candidate setting search, disjoint inner
  guard/outer selection, or label/task arrival ledger exists. Thus b2c,
  b2/b/c3/c, and c4 remain open. No external blocker was found.
- Exact next action: add a failing two-phase canary that assigns separate
  stable train, inner-guard, outer-selection, and final-test IDs/hashes;
  records source/label release and per-method task information; and
  raises if guard or selection opens an unarrived role. Then implement
  the smallest v5-only role split and decision wiring, with final-test
  access still after all candidate settings freeze.

## 2026-09-28 — P1.3c3b2c1 four-role source contract

- Completed task ID: **P1.3c3b2c1**. **P1.3c3b2c2–c3**, parent
  **P1.3c3b2c**, **P1.3c3b2**, **P1.3c3b**, **P1.3c3**, **P1.3c4**, and
  **P1.3c** remain unchecked. The previous goal turn made progress by
  completing b2b. This session started clean at `e46e91d` on
  `codex/research-protocols-and-resume`. During the session the shared
  branch advanced through `7721c28` (cross-platform memory checks),
  `3205ed0` (the plan split), and `ec7634d` (Pillow for figure tests).
  Those unrelated changes were preserved; c1 implementation/docs/log
  remain working-tree changes at this handoff.
- Inspection found v5 intentionally reproduces v4's one-validation-split
  scores and format-5 development identity. Changing that split in place
  would invalidate checked evidence. Split b2c into c1 source identity,
  c2 versioned runner/inner guard, and c3 outer setting selection without
  changing the parent acceptance criteria. A new c1 test first failed at
  `ModuleNotFoundError` for the missing `continual_roles` module.
- Added `src/infra/continual_roles.py`: it stratifies each phase's arrived
  training fields into class-covered train, inner-guard, and
  outer-selection roles. Stable IDs name original source rows, phase,
  and seed; hashes bind IDs, roles, and values. The split predeclares
  final-test IDs/count and a global-freeze availability policy but leaves
  final values/hash absent. `release_final_test` explicitly reads,
  validates, and hashes final fields. The synthetic generator may still
  allocate held-out arrays internally; this is field release, not a
  measured runtime event ledger. No benchmark runner consumes these
  roles yet. ADR-0073 and README/protocol/infra docs state that scope.
- Evidence: five new cases cover A/B raising final-source sentinels,
  repeated and changed-seed identities, disjoint role and phase IDs,
  both classes in every development role, changed final labels leaving
  development identity unchanged, explicit final release, and invalid
  fractions/class coverage/final count. Existing v1–v5 orchestration was
  not edited. Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_decision_roles.py -q
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_decision_roles.py tests/test_dataset_roles.py -q
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_decision_roles.py tests/test_dataset_roles.py tests/test_continual_global_test_seal.py tests/test_continual_bounded_replay.py tests/test_continual_phase_label_arrival.py -q
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe format --check src/infra/continual_roles.py tests/test_continual_decision_roles.py
  git -c core.safecrlf=false diff --check
  ```

  The new test file passed **5** after implementation; new and existing
  dataset roles passed **8**. The focused continual set passed **61**;
  full CPU pytest exited 0 with **865** collected tests and no skips
  shown. Ruff, mypy (**135 source files**), touched-file format, and
  diff checks passed. Intermediate mypy/format failures from new test
  typing and layout were fixed before these final gates.
- Experiment artifacts and skipped work: no training experiment, sweep,
  or persistent result was produced. The red import error and validation
  output were terminal evidence only. Full CUDA pytest and
  representative-scale fairness work were not run for this source
  contract. Earlier negative circadian outcomes and artifacts were
  untouched.
- Plan changes and rationale: c1 alone is checked. The declared
  `phase_a_arrival`/`phase_b_arrival` and `global_freeze` policy does not
  prove actual access timing, per-method task information, guard
  decisions, checkpointed role preflight, or setting selection. C2–c3
  and all parents remain open; no acceptance criterion was weakened.
  No external blocker was found.
- Exact next action: add a failing v6 ordinary-run canary that forbids
  Phase B role construction until every Phase A model finishes, forbids
  inner guard access before its own phase arrival, and records each
  model's observed task information. Then add a separately named v6
  config/result and wire the new split into train-only phase helpers;
  extend the same boundary to a distinct checkpoint format before
  checking c2.

## 2026-09-28 — P1.3c3b2c2a ordinary arrived-role runner

- Completed task ID: **P1.3c3b2c2a**. **P1.3c3b2c2b**, parent
  **P1.3c3b2c2**, **P1.3c3b2c3**, **P1.3c3b2c**, **P1.3c3b2**,
  **P1.3c3b**, **P1.3c3**, **P1.3c4**, and **P1.3c** remain unchecked.
  The checkout began at `ec7634d` on
  `codex/research-protocols-and-resume` with the prior c1 changes in
  the working tree. They and unrelated branch content were preserved.
- Inspection found that v5 still gives every model one development
  split and has no guard decision or observed release ledger. Splitting
  its training data in place would break its v4-equivalent scores and
  format-5 identity. The plan split c2 into c2a ordinary proof and c2b
  a distinct checkpoint, without weakening c2 or parent criteria. The
  initial v6 canary failed at import before the new module existed.
- Added `src/app/continual_arrived_benchmark.py` with an opt-in v6
  config/result and a fixed-setting ordinary run. A roles are built at
  A arrival; B source and roles are built only after all A models finish.
  The declared Phase B exposure fraction is applied before B role split,
  with original source row IDs carried through `continual_roles.py`.
  Each model update consumes only its phase train role; attempted
  circadian sleeps read only that phase's inner guard, accept within the
  declared drop tolerance, and restore the complete snapshot on a
  rejected event. The runner records actual development/final field
  release, model updates, guard access/decisions, and per-method arrived
  phase information. All seeds finish training before any final source
  field is released or scored. The outer role is released but unused.
  Existing v1–v5 helper defaults remain unchanged. ADR-0074,
  README, evaluation protocol, and app/infra module docs describe the
  boundary. The fixed two-seed smoke entry point is
  `scripts/run_continual_arrived_smoke.py`.
- Evidence: the new tests cover both model orders with raising Phase B
  and final-source sentinels; repeated report equality and order
  isolation; changed outer labels leaving all trained state hashes,
  guard decisions, and metrics unchanged; changed inner labels reaching
  the guard while Backprop/PC reports stay unchanged; changed first-seed
  final labels leaving all trained states, guard decisions, and the
  other seed's report unchanged; harmful sleep rollback; and invalid
  run identity rejection before source access. The c1 tests additionally
  verify original B source-row ID mapping. The smoke output for seeds
  17/19 repeated byte for byte: 4,489 characters; SHA-256
  `a01df3eccf4dd378eba1bc4d66b457d02a7587661bd12413118fe00933c73ef8`.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_arrived_runner.py tests/test_continual_decision_roles.py -q
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_arrived_runner.py tests/test_continual_decision_roles.py tests/test_continual_global_test_seal.py tests/test_continual_final_label_seal.py tests/test_continual_bounded_replay.py tests/test_continual_phase_local_schedule.py tests/test_continual_phase_label_arrival.py tests/test_continual_checkpoint_resume.py tests/test_continual_shift_benchmark.py -q
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe format --check src/app/continual_arrived_benchmark.py src/app/continual_shift_benchmark.py src/infra/continual_roles.py tests/test_continual_arrived_runner.py tests/test_continual_decision_roles.py scripts/run_continual_arrived_smoke.py
  git -c core.safecrlf=false diff --check
  .\.venv\Scripts\python.exe -m scripts.run_continual_arrived_smoke
  ```

  Final new-role tests passed **14**; the focused continual set passed
  **118**. The full CPU suite exited 0 with **874** collected tests and
  no skips shown. Ruff, mypy (**138 source files**), six-file format,
  and diff checks passed. One intermediate format check found only the
  new test file and was fixed with Ruff format; interim B source-ID type
  and callback-typing failures were corrected before final gates. A
  final audit-order edit moved each final-release event immediately after
  its successful field release; the 14 new-role tests and static gates
  passed again after that edit. The full suite was run before this
  audit-only ordering change.
- Experiment artifacts and skipped work: no persistent experiment result
  or sweep was produced. The fixed deterministic smoke JSON was a local
  terminal report; its digest is above. Full CUDA pytest and a
  representative-scale fairness study were not run for this tiny NumPy
  correctness step. Existing negative circadian results were untouched.
- Plan changes and limitations: c2a alone is checked. V6 changes training
  role sizes, so its scores are descriptive and cannot be compared as
  v5-equivalent. The generator can allocate held-out arrays before the
  observed field-release gate. The ordinary runner holds all seed states
  in memory. There is no v6 checkpoint, candidate setting selection, or
  full strict-online claim; c2b–c3 and all parents stay open. No external
  blocker was found.
- Exact next action: add a failing two-seed v6 checkpoint canary at
  Phase A wake/sleep and later-seed boundaries that matches the ordinary
  role IDs/hashes, guard decisions, access cursor, trained state, and
  globally sealed final fields. Design a distinct checkpoint format and
  validate changed arrived role content and replay provenance before
  any resumed update, then implement the smallest resume boundary.

## 2026-09-28 — P1.3c3b2c2b1 completed-seed v6 checkpoint

- Completed task ID: **P1.3c3b2c2b1**. **P1.3c3b2c2b2**, parent
  **P1.3c3b2c2b**, **P1.3c3b2c2**, **P1.3c3b2c3**, **P1.3c3b2c**,
  **P1.3c3b2**, **P1.3c3b**, **P1.3c3**, **P1.3c4**, and **P1.3c**
  remain unchecked. The previous goal turn made progress by completing
  c2a. This session began at `ec7634d` on
  `codex/research-protocols-and-resume` with the c1/c2a working-tree
  changes present; they and unrelated branch content were preserved.
- Inspection showed that v5's format-5 record cannot express the v6
  four-role IDs, inner guard decisions, or observed access cursor. Split
  c2b before implementation into b1 completed-seed storage and b2 active
  wake/sleep transactions. The full c2b acceptance criteria remain
  unchanged. The first b1 canary failed at import for the missing v6
  store; after adding it, the canary failed at the missing checkpoint API.
- Added `src/app/continual_arrived_checkpoint.py` and a distinct trusted
  local v6 file header/store in `src/infra/circadian_checkpoint_files.py`.
  A format-6 `seed_complete` transaction stores a detached trained state,
  six arrived development role IDs/hashes, observed role-access/guard/task
  records and their digest, and the full config/ordered-seed digest. It
  stores no source object, final-test field/hash/score, or scored result.
  The active state/cursor fields are reserved but rejected in b1. The v6
  runner validates the header before source access, regenerates each
  prior seed's development roles, and verifies role content, baseline
  steps, circadian wake state, replay budget/membership in train-only
  examples, and event cursor before training a later seed. Terminal
  resume only regenerates roles and scores after all seeds are present;
  it does not retrain or rewrite the file. ADR-0075, README, evaluation
  protocol, and module docs describe the boundary. No v1–v5 checkpoint
  schema or route was edited.
- Evidence: interrupted seed-17 runs resume seed 19 and equal the
  ordinary v6 report in both model orders, with no seed-17 retraining.
  A two-seed raising source sentinel forbids both final source fields
  until both seeds train. Terminal resume avoids training in both orders.
  Changed run config/seed list rejects before source access; changed
  prior A or B labels, future-seed replay, a missing event, and paired
  guard-event omission all reject before another update in both orders.
  Changing seed-17 final labels after a terminal checkpoint changes only
  its final hash/report; seed 19 and the saved checkpoint bytes stay
  identical. The new checkpoint test file passed **17** cases.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_arrived_checkpoint.py -q
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_arrived_checkpoint.py tests/test_continual_arrived_runner.py tests/test_continual_decision_roles.py tests/test_continual_global_test_seal.py tests/test_continual_final_label_seal.py tests/test_continual_bounded_replay.py tests/test_continual_phase_local_schedule.py tests/test_continual_phase_label_arrival.py tests/test_continual_checkpoint_resume.py tests/test_continual_shift_benchmark.py -q
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe format --check src/app/continual_arrived_benchmark.py src/app/continual_arrived_checkpoint.py src/infra/circadian_checkpoint_files.py tests/test_continual_arrived_checkpoint.py
  git -c core.safecrlf=false diff --check
  ```

  The focused continual set passed **134** before the added
  reverse-order terminal case; the checkpoint file passed **17** after
  that case. The final full CPU suite exited 0 with **891** collected
  tests and no skips shown. Ruff, mypy (**140 source files**), four-file
  format, and diff checks passed. One intermediate format-only failure
  in the v6 runner was corrected with Ruff format before final gates.
- Experiment artifacts and skipped work: no persistent experiment
  result or sweep was produced. Trusted checkpoint files were confined
  to pytest temporary directories and are not retained as scientific
  artifacts. Full CUDA pytest and representative-scale fairness work
  were not run for this NumPy checkpoint boundary. Earlier negative
  circadian outcomes and their artifacts were untouched.
- Plan changes, limits, and blockers: b1 alone is checked. The new
  format-6 schema currently accepts only completed-seed transactions;
  it cannot resume inside A/B training. B2 retains wake, before-sleep,
  after-sleep, A/B transition, and active role/guard cursor preflight.
  C2b, c2, c3, and all parents remain open. Scores remain descriptive,
  with no outer setting selection. No external blocker was found.
- Exact next action: add a failing two-seed v6 checkpoint test that
  interrupts after the first Phase A model update and at the following
  before-sleep and after-sleep transactions in both model orders, then
  extends to Phase B/A–B transition. Persist active model/role/audit
  state in the reserved format-6 fields and validate its cursor, replay,
  and arrived role identity before any resumed update.

## 2026-09-28 — P1.3c3b2c2b2 active v6 checkpoint transactions

- Completed task IDs: **P1.3c3b2c2b2**, **P1.3c3b2c2b**, and
  **P1.3c3b2c2**. **P1.3c3b2c3** and its strict-online parents remain
  unchecked. This session continued at `ec7634d` on
  `codex/research-protocols-and-resume` with prior c1/c2a/b1 changes
  already in the working tree. Existing unrelated changes were preserved.
- The first active A wake canary failed as expected because format 6
  saved only completed seeds. Added `src/app/continual_arrived_transactions.py`
  to save/restore an active seed after nonterminal model wake updates,
  before sleep, after sleep, and Phase B arrival. Active format-6 records
  hold only arrived development role IDs/hashes, detached models and
  circadian snapshot, observed release/update/guard/task ledger and its
  digest, cursor, and prior unscored seeds. The ordinary runner and
  v1–v5 checkpoint formats remain separate. ADR-0076 and the README,
  architecture, evaluation protocol, and module docs describe this
  boundary and its trusted local file assumption.
- Evidence: sixteen active interruption/resume cases cover A/B wake,
  before-sleep, after-sleep, A/B transition, and later-seed interruption
  in both model orders. Reports match ordinary v6; every saved trained
  state field and replay row matches an uninterrupted checkpointed run.
  The interrupted and resumed executions together perform only the
  expected remaining updates for all three methods. Earlier b1 tests
  cover both terminal resume orders. Raising A/B source sentinels keep
  Phase B from arriving during A; a two-seed final-source sentinel keeps
  final input and label fields sealed until every model trains. Twenty
  active tamper cases reject changed A/B role content, future/unarrived/
  outer replay, missing or paired-omission access/guard events, forged
  baseline steps, changed position, and a future B role hash before any
  resumed update. Existing ordinary outer/final-label perturbation tests
  and b1 prior-seed/terminal tests still pass.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_arrived_checkpoint.py -q -x
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_arrived_checkpoint.py tests/test_continual_arrived_runner.py tests/test_continual_decision_roles.py tests/test_continual_global_test_seal.py tests/test_continual_final_label_seal.py tests/test_continual_bounded_replay.py tests/test_continual_phase_local_schedule.py tests/test_continual_phase_label_arrival.py tests/test_continual_checkpoint_resume.py tests/test_continual_shift_benchmark.py -q
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe format --check src/app/continual_arrived_benchmark.py src/app/continual_arrived_checkpoint.py src/app/continual_arrived_transactions.py tests/test_continual_arrived_checkpoint.py
  git -c core.safecrlf=false diff --check
  ```

  The checkpoint file passed **53** cases; the focused continual set
  passed **171**. The full CPU suite exited 0 with **927** collected
  tests and no skips shown. Ruff, mypy (**141 source files**), touched
  format, and diff checks passed. The initial active canary failed as
  expected before implementation; one intermediate Ruff format check
  found only layout changes, which were formatted before final gates.
- Experiment artifacts and skipped work: format-6 checkpoint files
  existed only in pytest temporary directories; no persistent scientific
  experiment or sweep was produced. Full CUDA pytest and representative
  fairness experiments were not run for this NumPy checkpoint boundary.
  Earlier negative circadian outcomes and artifacts were untouched.
- Plan changes, limits, and blockers: b2, b, and c2 now meet their
  stated gates and are checked, with evidence above. C3 candidate
  setting selection, b2c and strict-online parents remain open. V6
  scores are descriptive; no candidate selection or comparison was
  performed and no metric, seed, or baseline was tuned to favor the
  circadian model. The trusted checkpoint format does not protect
  against arbitrary malicious pickle execution. No external blocker
  was found.
- Exact next action: add a failing tiny two-candidate v6 test that
  predeclares equal-effort settings and chooses from arrived outer roles
  only, then run it through ordinary and checkpointed paths with a
  two-seed final-source sentinel. Prove final-label perturbation cannot
  change choice or trained states, record each trial/exposure and a
  frozen selection before any final release, and implement the smallest
  passing candidate runner without a large sweep.

## 2026-09-29 — P1.3c3b2c3a ordinary outer selection

- Completed task ID: **P1.3c3b2c3a**. **P1.3c3b2c3b**, parent
  **P1.3c3b2c3**, **P1.3c3b2c**, **P1.3c3b2**, **P1.3c3b**,
  **P1.3c3**, **P1.3c4**, and **P1.3c** remain unchecked. The previous
  goal turn completed c2b2/c2b/c2, so this turn made progress on the
  next isolation gate. The checkout remained at `ec7634d` on
  `codex/research-protocols-and-resume`; all prior and unrelated dirty
  changes were preserved.
- Inspection showed that calling the public v6 runner separately for
  candidate settings would score final tests before all settings finish.
  Split c3 into c3a ordinary selection and c3b durable candidate-manifest
  recovery without weakening parent criteria. The first c3a canary
  failed at import because the selection API did not exist.
- Added `src/app/continual_arrived_selection.py`, a separate opt-in
  `continual_arrived_outer_selection_v7` route. It validates two to four
  ordered candidates and at most eight candidate-seed trials per method
  before source access. Candidate configs share seeds, role splits,
  training examples, phase epochs, model order, inference, guard, sleep,
  and replay budgets; only distinct per-method learning rates may vary.
  All candidate/seed models train before any outer evaluation. The fixed
  objective is the seed mean of `0.5*(A_post+B_post)` on arrived outer
  roles, with the first declared candidate winning an exact tie.
  The output retains all candidate configs and twelve trial rows in the
  tiny fixture, including development IDs/hashes, role and guard events,
  per-method task information, wake-example/update counts, guard/outer
  exposures, replay IDs/bytes, sleep counts, and scores. A digest binds
  the candidate manifest, full trial ledger, and independent choices for
  all three methods before any final source release. Only selected model
  states receive final scoring. ADR-0077, README, architecture, evaluation
  protocol, and app-module docs record the ordinary-only boundary.
- Evidence: two-order two-candidate/two-seed raising source sentinels
  require all four candidate-seed training runs and the freeze before
  either final input or label is read. Final-label perturbation changes
  the released hash while leaving every serialized trained state, outer
  trial, choice, and freeze unchanged in both orders. Flipping only outer
  labels changes outer hashes and at least one outer score while leaving
  all trained states unchanged. A forced mixed-choice test verifies that
  each method's final score receives exactly its selected candidate's
  model. Exact ties choose the first candidate; repeated and reversed
  model-order runs retain the same scores and choices. Invalid IDs,
  unequal fixed work, and duplicate per-method rates fail before source
  access. The runner uses the existing v6 phase-arrival and inner guard
  training boundary; v1–v6 outputs and checkpoint formats were not edited.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_arrived_selection.py -q -x
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_arrived_selection.py tests/test_continual_arrived_checkpoint.py tests/test_continual_arrived_runner.py tests/test_continual_decision_roles.py tests/test_continual_global_test_seal.py tests/test_continual_final_label_seal.py tests/test_continual_bounded_replay.py tests/test_continual_phase_local_schedule.py tests/test_continual_phase_label_arrival.py tests/test_continual_checkpoint_resume.py tests/test_continual_shift_benchmark.py -q
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe format --check src/app/continual_arrived_selection.py tests/test_continual_arrived_selection.py scripts/run_continual_arrived_selection_smoke.py
  git -c core.safecrlf=false diff --check
  .\.venv\Scripts\python.exe -m scripts.run_continual_arrived_selection_smoke
  ```

  The new selection file passed **9** tests; the focused continual set
  passed **180**. The full CPU suite exited 0 with **936** collected
  tests and no skips shown. Ruff, mypy (**144 source files**), touched
  format, and diff checks passed. The smoke output repeated byte for
  byte, SHA-256
  `46c80b3934bcbc84b80a426412a1b6108054c81b27d19d4f68d6ec90cbac7dd9`.
  Initial mypy loop-variable typing and Ruff layout findings were fixed
  before final gates.
- Experiment artifacts and skipped work: the fixed smoke printed JSON
  locally; no persistent scientific result file or sweep was produced.
  Its twelve trial rows all tied across the two declared rates, so the
  first candidate was selected for every method. Final balanced scores
  were seed 17: backprop 0.95, PC 0.70, circadian 0.00; seed 19: backprop
  0.15, PC 0.50, circadian 0.35. This tiny synthetic outcome is retained
  without changing seeds, rates, objective, or metrics to favor any model.
  Full CUDA pytest and representative-scale fairness work were not run
  for this NumPy selection boundary. Prior negative artifacts were
  untouched.
- Plan changes, limits, and blockers: c3a alone is checked. The new
  route holds all candidate states in memory and has no durable
  candidate-manifest selection checkpoint. Its counted training examples
  are wake-role reads; replay retention and guard/outer exposures are
  reported separately. The synthetic generator may allocate held-out
  arrays before release, although the runner source-field sentinel
  proves no early read. No external blocker was found.
- Exact next action: add a failing two-candidate/two-seed checkpoint
  canary that interrupts inside candidate one, between candidates, and
  after selection freeze in both model orders. Persist the ordered
  candidate manifest, completed outer trials/exposures, and frozen
  per-method choices; validate changed settings, outer role content,
  event cursor, and choice before any resumed update or final source
  release. Keep c3 and every parent unchecked until this and their full
  acceptance criteria pass.

## 2026-09-29 — P1.3c3b2c3b manifest-bound selection checkpoint

- Completed task IDs: **P1.3c3b2c3b**, parent **P1.3c3b2c3**, and the
  audited role/selection parents **P1.3c3b2c**, **P1.3c3b2**,
  **P1.3c3b**, and **P1.3c3**. **P1.3c4** and **P1.3c** remain
  unchecked. The previous goal turn completed c3a, so this turn made
  progress on its checkpointed counterpart. The checkout stayed at
  `ec7634d` on `codex/research-protocols-and-resume`; prior and unrelated
  working-tree changes were preserved.
- Inspection found that format 6 binds one candidate and cannot prove a
  global all-candidate freeze by itself. The first c3b canary failed at
  collection because no format-7 selection store existed. Added
  `src/app/continual_arrived_selection_checkpoint.py` for the typed
  manifest/cursor and source-free header preflight,
  `src/app/continual_arrived_selection_resume.py` for run-level recovery,
  and `TrustedLocalArrivedSelectionCheckpointStore` with a distinct magic
  header in `src/infra/circadian_checkpoint_files.py`. The v7 API now
  accepts `checkpoint_store` and `resume_from_checkpoint` without changing
  v6's single-setting format. ADR-0078, README, architecture, evaluation
  protocol, and app/infra module docs describe the boundary.
- Format 7 atomically stores the full ordered candidate IDs/configs,
  seeds and fixed-objective manifest digest, each completed candidate's
  detached unscored v6 seed states and all outer trial/exposure rows,
  a nested active format-6 transaction, and a final frozen per-method
  choice. The file contains no final-test value, hash, score, or released
  source object. Resume verifies run and nested v6 headers before source
  access, rehydrates completed candidates and their arrived role/replay/
  model/event state, recomputes their outer trials before updates, and
  resumes only the active candidate's remaining A/B transactions. At the
  frozen stage it recomputes all three choices from the complete trial
  ledger before final release.
- Evidence: twelve two-order interruption cases cover first-candidate A/B
  wake, between-candidate boundary, later-candidate A wake/B after-sleep,
  and frozen choice. Resumed results equal ordinary and uninterrupted
  checkpointed reports; saved model fields/replay rows match field by
  field, and interruption plus resume performs exactly the fixed **24**
  model updates. Twenty two-order tamper cases reject changed requested
  or stored candidate manifest, settings/seeds, completed outer role
  content, trial rows even with recomputed digest, completed/active event
  or role cursor, and a forged frozen choice before any update or final
  release. The source-property sentinel requires frozen stage for every
  final input/label read. In both model orders, changing seed-17 Phase A
  final labels after terminal freeze leaves trials, trained states,
  choices, and checkpoint bytes unchanged while changing that final role
  hash; seed 19's result remains identical. The v6 store rejects the v7
  file header. The fixed ordinary smoke's tied choices and low circadian
  outcome were retained; no seed, rate, objective, or metric was changed.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_arrived_selection_checkpoint.py -q -x
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_arrived_selection_checkpoint.py tests/test_continual_arrived_selection.py tests/test_continual_arrived_checkpoint.py tests/test_continual_arrived_runner.py tests/test_continual_decision_roles.py tests/test_continual_global_test_seal.py tests/test_continual_final_label_seal.py tests/test_continual_bounded_replay.py tests/test_continual_phase_local_schedule.py tests/test_continual_phase_label_arrival.py tests/test_continual_checkpoint_resume.py tests/test_continual_shift_benchmark.py -q
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\python.exe -m pytest tests/test_readme_figures_protocol.py -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe format --check src/app/continual_arrived_selection.py src/app/continual_arrived_selection_checkpoint.py src/app/continual_arrived_selection_resume.py src/infra/circadian_checkpoint_files.py tests/test_continual_arrived_selection_checkpoint.py
  git -c core.safecrlf=false diff --check
  ```

  The new checkpoint file passed **35** cases; the focused continual set
  passed **215**. The full CPU suite exited 0 with **971** collected tests
  and no skips shown. The README protocol tests passed **6** after docs
  edits. Ruff, mypy (**147 source files**), touched format, and diff
  checks passed. Intermediate format-only and mypy loop-variable typing
  findings were fixed before final gates.
- Experiment artifacts and skipped work: v7 checkpoint files existed
  only in pytest temporary directories; no persistent scientific result
  file or large sweep was produced. Full CUDA pytest and representative
  fairness experiments were not run for this NumPy checkpoint gate.
  Earlier negative circadian artifacts and the fixed c3a smoke output
  were untouched.
- Plan changes, limits, and blockers: c3b and the audited c3/b2c/b2/c3b/c3
  parents are checked with their stated gates preserved. The separate
  c4 bounded A→B confirmation and full strict-online parent c remain
  open. Format 7 is a trusted local pickle with checksum, not an
  authenticated untrusted-data format. Candidate states and trials are
  deliberately held under the small local budget. No external blocker
  was found.
- Exact next action: predeclare and save a small fixed c4 request with
  candidate configs, seeds, phase/work budget, both model orders,
  selection objective, and required outcome fields before opening final
  tests. Add a failing end-to-end confirmation canary covering ordinary
  and checkpointed A/B interruption/resume, final-source sealing,
  role/state/order equality, and complete trial/negative-outcome
  reporting. Run only that budgeted comparison after its gates pass;
  leave c4 and strict-online parent c unchecked until verified.

## 2026-09-29 — P1.3c4 bounded strict-online A→B confirmation

- Completed task IDs: **P1.3c4** and the audited **P1.3c** parent. P1.8
  representative-scale fairness, P3.9 CUDA branches, and deferred P2.6a
  remain unchecked. The checkout stayed at `ec7634d` on
  `codex/research-protocols-and-resume`; prior and unrelated dirty changes
  were preserved without a commit.
- The first end-to-end canary failed at collection because the confirmation
  script did not exist. Added a fixed two-candidate/two-seed request writer
  and runner. Request validation, Windows newline handling, a sealed-source
  property, and semantic model-field hashing were corrected during the
  red/green loop. None changed the predeclared seeds, rates, selection
  objective, final metric, work budget, or earlier negative result. ADR-0079,
  README, architecture, evaluation protocol, and adapter docs now describe
  the experiment boundary and its limited inference.
- The ignored request was saved before this turn's first c4 final-test
  access at `data/continual_arrived_confirmation_v1_request.json`
  (21,777 bytes; SHA-256
  `2f68c48ecea17dd2c761b08b1d02fc67db80f5f3ee044e4d10b8e4ace4399cee`).
  It fixes seeds 17/19, both model orders, default and 0.7× rate candidates,
  40 examples and one epoch per phase, two inference steps, component
  sleep, four-example/96-byte replay caps, the outer-only mean balanced
  objective and first-candidate tie rule, A/B wake interruption points,
  96 total model updates, a 180 s local cap, and all required outcome
  fields. The runner rejects missing/changed requests and existing result
  or checkpoint paths before source access.
- The unchanged request completed in **0.704 s** at
  `data/continual_arrived_confirmation_v1_result.json` (417,348 bytes;
  SHA-256
  `5d69dc91ab05d31eb8aab9f8c44f7989f4888c1b2f516bda2340edcfb0ba2665`).
  Forward/reverse checkpoint artifacts are under
  `data/continual_arrived_confirmation_v1_checkpoints/` (74,711 bytes
  each). Each order completed 12 outer trials and two final seed rows;
  ordinary and A/B-interrupted-resumed runs matched in reports and six
  trained model-state field digests for each of four candidate/seed groups.
  A source sentinel logged exactly eight final input/label reads per path,
  all after global freeze; the independent JSON audit confirmed disjoint
  four-role IDs and the full trial grid. Both orders had the same choices,
  trained-state digests, role hashes, and method scores. No final value or
  score was written to the v7 checkpoint.
- All three methods selected the predeclared default on the fixed outer
  objective. Final balanced scores for backprop/PC/circadian were
  **0.95/0.70/0.00** at seed 17 and **0.15/0.50/0.35** at seed 19. The
  result retains circadian-minus-backprop differences of -0.95/+0.20 and
  circadian-minus-PC differences of -0.70/-0.15, every outer trial, work,
  role/label/task ledger, replay IDs/count/bytes, and per-seed final row.
  This is a negative circadian result at a tiny synthetic scale; it does
  not justify a general ranking or a revised baseline.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_arrived_confirmation.py -q -x
  .\.venv\Scripts\python.exe -m scripts.run_continual_arrived_confirmation prepare --request data/continual_arrived_confirmation_v1_request.json
  .\.venv\Scripts\python.exe -m pytest tests/test_continual_arrived_confirmation.py tests/test_continual_arrived_selection_checkpoint.py tests/test_continual_arrived_selection.py tests/test_continual_arrived_checkpoint.py tests/test_continual_arrived_runner.py tests/test_continual_decision_roles.py -q
  .\.venv\Scripts\python.exe -m scripts.run_continual_arrived_confirmation run --request data/continual_arrived_confirmation_v1_request.json --result data/continual_arrived_confirmation_v1_result.json --checkpoint-dir data/continual_arrived_confirmation_v1_checkpoints
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe format --check scripts/run_continual_arrived_confirmation.py tests/test_continual_arrived_confirmation.py
  git -c core.safecrlf=false diff --check
  ```

  The two new canaries, **113** focused tests, and **973** collected/full
  CPU tests passed with no skips shown. Ruff, mypy (**149 source files**),
  touched-file format, and diff checks passed. The independent artifact
  audit passed. Intermediate red/green failures were fixed before final
  gates; the fixed artifact was not rerun or overwritten.
- Skipped work: no CUDA pytest, GPU timing probe, representative-scale
  fairness run, or large sweep was launched. The existing CPU/CUDA
  negative matched-baseline artifacts were untouched. The synthetic
  generator may allocate held-out arrays early, but the v7 runner does
  not release their source fields before all settings freeze; this
  distinction remains documented in ADR-0073–0079.
- Plan changes and blockers: checked c4 and parent c only after all their
  stated gates passed. Added open P1.8o to measure a development-only
  representative-scale feasibility point before freezing another matched
  comparison; P1.8 remains unchecked and its criteria were not weakened.
  There is no external blocker for the next read-only/predeclaration step.
- Exact next action: inspect the saved P1.8n CUDA result plus verified
  CIFAR archive/pretrained-weight hashes and current GPU telemetry. Then
  save one larger-input/larger-train **development-only** feasibility
  request with source hashes, a quiet-device gate, and a 120 s limit
  before loading its data. Record examples/batches, feature time, scoped
  RSS/allocator, and any failure; freeze separate equal-trial selection
  and fixed-data, wall-time, and isolated-memory confirmation budgets
  from that cost evidence before any new final-test access.

## 2026-09-29 — P1.8o development-only representative-scale feasibility

- Completed task ID: **P1.8o**. The prior turn's P1.3c4/parent completion
  was verified in the current checkout; this turn advanced the next
  matched-baseline gate. **P1.8p1**, **P1.8p2**, and P1.8 parent remain
  unchecked. The branch stayed `codex/research-protocols-and-resume` at
  `ec7634d`; existing dirty and unrelated changes were preserved.
- Read the P1.8n saved CUDA request/result and prior seed-101 feature cost.
  The existing CUDA study used 32-pixel/1,024-example roles, completed in
  98.172 s, and ended at 51% GPU utilization; its negative circadian
  accuracy and memory scopes were left unchanged. Verified the local
  170,498,071-byte CIFAR archive (MD5
  `c58f30108f718f92721af3b95e74349a`, SHA-256
  `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`)
  and 102,540,417-byte ImageNet V2 weights (SHA-256
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`).
- Inspection found `build_torchvision_vision_dataloaders` constructed the
  CIFAR `train=False` dataset even in an otherwise development-only
  feature bank. The new fake-CIFAR canary first failed on the missing
  `include_final_test` option. Added opt-in false through infra, the
  benchmark loader boundary, and `_build_seed_bank`: it constructs only
  train/guard/validation roles, omits final IDs/hashes, and returns a
  raising final loader. The default path and all earlier outputs retain
  their behavior. A fake source rejects final construction; a separate
  app test proves the bank requests the opt-in boundary. ADR-0080 and
  architecture/module/evaluation docs explain why.
- Before the cost run, saved ignored
  `data/cifar-representative-feasibility-v1-request.json` (5,736 bytes,
  SHA-256 `7743a404824f92c35e2aafb4f5bfaa709c05e65ffbdcfd6c43098fa99ee16d3a`).
  It fixes seed 173, 224-pixel input, 4,096/512/512 development roles,
  batch 32, source hashes, three quiet readings five seconds apart at
  <=10% utilization and >=5 GiB free, and a hard 120 s child timeout.
  The actual readings were **3%/1%/1%** with 8,139/8,204/8,203 MiB
  free; post-run utilization was 2%.
- The one worker completed in **9.522 s** at
  `data/cifar-representative-feasibility-v1-result.json` (3,306 bytes,
  SHA-256 `a1180542abd4b1e51cd545022f6b501842ea3a340609e9888d38d9c35bb08649`).
  It processed **128/16/16** train/guard/validation feature batches and
  **4,096/512/512** examples. Role times were 6.518/0.800/0.789 s plus
  1.416 s setup; all role/source/feature/backbone hashes and feature
  bytes were retained. The observed worker RSS start/peak was
  759,087,104/1,555,025,920 bytes; PyTorch CUDA allocated/reserved peaks
  were 447,518,208/660,602,880 bytes. These are feature-setup scopes,
  not head-training memory. No final CIFAR source was constructed or
  iterated, and no head or candidate was trained.
- From that cost evidence, saved the **unexecuted**
  `data/cifar-representative-study-v1-request.json` (39,182 bytes,
  SHA-256 `baa4bcaf5138946d01ad0d7c1b8babc21350ae18e53e24bc4e37a9cedf3dc773`).
  It fixes 224-pixel 16,384/2,048/2,048 development roles and 4,096
  final examples, selection seed 179, confirmation seeds 181/191/193,
  two base/0.8× optimization candidates per head, outer-only validation
  choice, and distinct one-epoch fixed-data, 5 s/head wall-time, and
  fresh-child fixed-width memory scopes. Limits are 180 s selection,
  240/240/600 s per confirmation scope, and 1,080 s total. Linear
  feature-cost projection is ~33.84 s/seed; it is not a runtime guarantee.
  The independent JSON audit verified the probe digest, disjoint seeds,
  candidate grid, role sizes, and budget fields. No new final test was
  opened or scored, and no previous seed/rate/metric was revised.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_vision_datasets.py::test_should_build_development_roles_without_opening_cifar_final_source -q -x
  .\.venv\Scripts\python.exe -m pytest tests/test_cifar_representative_study.py tests/test_cifar_representative_feasibility.py tests/test_vision_datasets.py tests/test_matched_head_tuning.py -q -x
  .\.venv-cuda\Scripts\python.exe -m scripts.profile_cifar_representative_feasibility prepare
  .\.venv-cuda\Scripts\python.exe -m scripts.profile_cifar_representative_feasibility run
  .\.venv-cuda\Scripts\python.exe -m scripts.prepare_cifar_representative_study
  .\.venv-cuda\Scripts\python.exe -m pytest tests/test_cifar_representative_feasibility.py tests/test_cifar_representative_study.py tests/test_vision_datasets.py tests/test_matched_head_tuning.py -q -x
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\python.exe -m pytest tests/test_readme_figures_protocol.py -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe format --check scripts/profile_cifar_representative_feasibility.py scripts/prepare_cifar_representative_study.py tests/test_cifar_representative_feasibility.py tests/test_cifar_representative_study.py src/app/matched_head_tuning.py tests/test_matched_head_tuning.py
  git -c core.safecrlf=false diff --check
  ```

  Focused **18** tests passed in both CPU and CUDA environments; the full
  CPU suite passed **978** collected tests with no skips shown. README
  protocol tests passed **6**. Ruff, mypy (**153 source files**), the six
  new/formatted-file checks, and diff checks passed. An attempted
  whole-file formatter check on legacy touched vision files reported
  existing formatting differences; checking their HEAD versions produced
  the same failure. No broad formatting rewrite was made.
- Skipped work: the full CUDA pytest suite, a larger head selection or
  confirmation, and any new final-test access were intentionally not run.
  The saved matched-study request is a prospective budget, not a result.
  The 16,384-example CIFAR subset and frozen pretrained backbone still
  limit any eventual ranking; P1.8 remains unchecked.
- Plan changes and blockers: checked P1.8o only after its development
  source, cost, memory, and budget-freeze gates passed. Added open P1.8p1
  validation-only selection and P1.8p2 confirmation with the parent
  fairness criteria preserved. No external blocker exists for p1.
- Exact next action: add a failing CIFAR final-source-construction canary
  for matched-head **selection** (not just the feature-cost helper), then
  implement an opt-in development-only selection path. Restore unchanged
  study request SHA-256 `baa4bcaf...dc773`, verify archive/weight/probe
  digests and the same quiet GPU gate, run exactly six seed-179 outer
  validation trials under 180 s, and save all attempts/trials and a
  digest-checked manifest for seeds 181/191/193 before any final-test
  access. Keep failed or negative trials without retuning.

## 2026-09-29 — P1.8p1 frozen representative validation selection

- Completed task ID: **P1.8p1**. P1.8p2 confirmation and P1.8 parent stay
  unchecked. Branch `codex/research-protocols-and-resume` remained at
  `ec7634d`; the existing dirty tree and unrelated changes were preserved.
- Re-read the current task, existing probe/study artifacts, candidate grid,
  tuning/manifest code, and checkout status. A failing app canary first
  exposed the missing development-only tuning option. The new opt-in path
  requires CIFAR and `confirm_test=False`; the default behavior is unchanged.
  A fake CIFAR factory rejects `train=False` construction, the development
  loader has no final role labels/IDs/hashes and raises on final iteration,
  and the actual worker installs the same physical construction trap. The
  worker returned zero final-source constructions and iterations; it saved
  no final score.
- Inspection found a hard worker timeout could erase completed candidate
  evidence. Added an optional app attempt observer and a flushed/synced
  local JSONL journal for start, complete, and failure events. The parent
  checks the journal against all six result rows and saves partial events
  in a failure artifact on timeout or worker failure. Tests cover timeout
  retention, failed-worker trials, exact-request rejection, equal manifest
  binding, source isolation, and first-candidate ties. ADR-0081 records why.
- Verified the exact 39,182-byte study request SHA-256
  `baa4bcaf5138946d01ad0d7c1b8babc21350ae18e53e24bc4e37a9cedf3dc773`,
  saved probe digest, local CIFAR archive and ImageNet V2 weight hashes
  before launch. Candidate counts remained two per head, seeds remained
  179 and 181/191/193, and scope limits remained 180 s selection,
  240/240/600 s confirmation, and 1,080 s total. Quiet readings were
  **2%/3%/5%** utilization with **8,232/8,234/8,233 MiB** free; post-run
  utilization was 1%.
- The one bounded CUDA worker completed in **47.551 s** under 180 s. Six
  complete attempts/trials and 12 durable journal events cover the fixed
  224-pixel 16,384/2,048/2,048 train/guard/outer roles. All trials share
  identical role, feature, backbone, and initial-head hashes; each trained
  on 16,384 examples in 512 wake batches. Candidate a/b outer accuracies:
  backprop **0.8662109375/0.865234375**, predictive
  **0.77734375/0.75732421875**, circadian
  **0.716796875/0.67236328125**. Each independent head chose a. The
  circadian result is lower and no seed, rate, metric, or request was
  changed. These are validation scores only on a CIFAR subset with a
  frozen ImageNet backbone.
- Ignored local artifacts, retained without overwriting:

  | Path | Bytes | SHA-256 |
  | --- | ---: | --- |
  | `data/cifar-representative-selection-v1-result.json` | 81,233 | `3746e32459201c47198fa8eb529e03049a08da0bfa406e0a2dc678305927b1c6` |
  | `data/cifar-representative-selection-v1-manifest.json` | 23,241 | `6566678860366ee0435e70436fad1cd7fad667917823ce29fda079fcd60b8a6a` |
  | `data/cifar-representative-selection-v1-attempts.jsonl` | 84,229 | `49afd5c236354aa8b55ef3d413ababc1258c38acb57edefa0bb30e9746cc7741` |

  The independent post-save audit recomputed the selection digest
  `c6db28cd...0f068`, typed manifest digest `1b9ab9ca...55de`, and
  outer freeze digest `3cd7fe0f...a147c8af`, restored the typed manifest,
  and matched all six trial/candidate configs and all three saved budgets.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_cifar_representative_selection.py tests/test_matched_head_tuning.py -q -x
  .\.venv-cuda\Scripts\python.exe -c "from scripts.run_cifar_representative_selection import _verify_saved_request,REQUEST_PATH; r,s=_verify_saved_request(REQUEST_PATH); print(r['selection_seeds'], bool(s))"
  .\.venv\Scripts\python.exe -m pytest tests/test_cifar_representative_selection.py tests/test_matched_head_tuning.py tests/test_cifar_representative_study.py tests/test_cifar_representative_feasibility.py tests/test_vision_datasets.py -q -x
  .\.venv-cuda\Scripts\python.exe -m scripts.run_cifar_representative_selection run
  .\.venv-cuda\Scripts\python.exe -m pytest tests/test_cifar_representative_selection.py tests/test_matched_head_tuning.py tests/test_cifar_representative_study.py tests/test_cifar_representative_feasibility.py tests/test_vision_datasets.py -q -x
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest tests/test_readme_figures_protocol.py -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\ruff.exe format --check scripts/run_cifar_representative_selection.py scripts/prepare_cifar_representative_study.py tests/test_cifar_representative_selection.py tests/test_matched_head_tuning.py src/app/matched_head_tuning.py
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  git -c core.safecrlf=false diff --check
  ```

  Focused **23** tests passed in both CPU and CUDA environments. The full
  CPU suite passed **983** collected tests with no skips shown; README
  protocol tests passed **6**. Ruff, mypy (**155 source files**), touched
  file formatting, and diff checks passed. The result/manifest digest audit
  also passed after persistence. The first red canary failed for the
  missing development-only option and passed after implementation.
- Skipped tests/work: the full CUDA pytest suite, final-test confirmation,
  the three 181/191/193 confirmation seeds, and any large sweep were not
  run. P1.8p2 has no final result. Existing CPU/CUDA negative matched
  artifacts and the P1.8o request/probe were not rewritten.
- Plan changes and blockers: checked P1.8p1 only after all saved selection,
  isolation, ledger, tie, and static gates passed; added ADR-0081 and
  updated README/architecture/evaluation/module documentation. P1.8p2
  and parent P1.8 remain open with their original acceptance criteria.
  There is no external blocker for the next implementation step.
- Exact next action: implement a read-only restore/preflight of the three
  P1.8p1 artifacts against the unchanged P1.8o request, archive/weight
  hashes, six-trial ledger, and both manifest digests; reject tampering or
  missing evidence before final-source construction. Then add a failing
  final-access sentinel and separate bounded runners for 240 s fixed-data,
  240 s wall-time, 600 s fresh-child isolated-memory, and 1,080 s total
  confirmation. Recheck the saved quiet CUDA gate before running only
  seeds 181/191/193, and retain every failure and negative score.

## 2026-09-29 — P1.8p2a exact-artifact restore before final access

- Completed task ID: **P1.8p2a**, split from open P1.8p2 because the first
  final-test access needs a separately testable, read-only evidence gate.
  **P1.8p2b**, P1.8p2 parent, and P1.8 parent remain unchecked. The
  checkout stayed on `codex/research-protocols-and-resume` at `ec7634d`;
  existing dirty and unrelated changes were preserved.
- Re-inspected AGENTS.md, the P1.8p2 acceptance criteria and current
  handoff, checkout status, P1.8p1 artifact bytes, and the existing
  repeated-confirmation and process-memory APIs. A first test failed at
  collection because the restore module was absent. Added
  `scripts/restore_cifar_representative_selection.py` with a typed result
  and a CLI `--preflight` entry point. It performs no dataset construction,
  head training, or final scoring.
- The restore gate compares exact SHA-256 bytes for the saved P1.8p1 result,
  manifest, and journal before source hashing. It then restores the frozen
  request SHA-256 `baa4bcaf...dc773`, cost-probe digest, CIFAR archive, and
  ImageNet V2 weight hashes. It validates six complete seed-179 equal-grid
  trials, ordered start/completion journal pairs, shared role/feature/
  backbone/initial hashes, outer-only choices, zero final constructions and
  iterations, the prior 2%/3%/5% quiet selection readings, selected head
  configs, typed manifest digest, and outer freeze digest with all saved
  240/240/600 s scopes plus the 1,080 s total.
- The actual ignored P1.8p1 artifacts passed this read-only preflight:
  result SHA-256 `3746e32459201c47198fa8eb529e03049a08da0bfa406e0a2dc678305927b1c6`,
  manifest SHA-256 `6566678860366ee0435e70436fad1cd7fad667917823ce29fda079fcd60b8a6a`,
  journal SHA-256 `49afd5c236354aa8b55ef3d413ababc1258c38acb57edefa0bb30e9746cc7741`.
  The CLI reported six trials, zero final iterations, selection digest
  `c6db28cdf786e616beb29c02577f29002c60481398949dd45f8d1f535290f068`,
  typed manifest digest `1b9ab9caf7e6cb517fa1372af0fa3392e0f8015848c1116d4b816869613f55de`,
  and outer freeze digest
  `3cd7fe0f6f5c1fa8c9dff78f07ff1ce8aef3cf0d51cfe8126930123fa147c8af`.
  None of those saved artifacts or the frozen request was rewritten.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_cifar_representative_restore.py -q -x
  .\.venv-cuda\Scripts\python.exe -m scripts.restore_cifar_representative_selection --preflight
  .\.venv\Scripts\python.exe -m pytest tests/test_cifar_representative_restore.py tests/test_cifar_representative_selection.py tests/test_matched_head_tuning.py tests/test_repeated_head_confirmation.py -q -x
  .\.venv-cuda\Scripts\python.exe -m pytest tests/test_cifar_representative_restore.py tests/test_cifar_representative_selection.py tests/test_matched_head_tuning.py tests/test_repeated_head_confirmation.py -q -x
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\python.exe -m pytest tests/test_readme_figures_protocol.py -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\ruff.exe format --check scripts/restore_cifar_representative_selection.py tests/test_cifar_representative_restore.py
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  git -c core.safecrlf=false diff --check
  ```

  Six new tests reject changed result/manifest bytes before source access,
  missing journal evidence, changed completed trial events, malformed
  journal data, and nonzero final iterations. Focused **26** passed in
  both CPU and CUDA environments. Full CPU **989** collected tests passed,
  README protocol **6** passed, and Ruff, mypy (**157 source files**),
  touched-file format, and diff checks passed. The first red collection
  failure was expected and resolved by the implementation.
- Skipped tests/work: full CUDA pytest, every P1.8p2b fixed-data/wall-time/
  memory confirmation scope, the new quiet launch window, and final-test
  scoring were intentionally not run. No large sweep was launched and no
  result, seed, candidate, metric, or budget was altered to favor circadian.
- Plan changes and blockers: split P1.8p2 into p2a restore and p2b bounded
  execution to isolate the irreversible final-access gate. Checked p2a
  only after its local artifact, tamper, focused, full CPU, and static gates
  passed. Preserved every original P1.8p2 criterion in open p2b/parent;
  added ADR-0082 and updated README, architecture, adapter, and evaluation
  docs. There is no external blocker for p2b implementation.
- Exact next action: add a failing final-access sentinel and tests for
  240 s fixed-data, 240 s wall-time, 600 s fresh-child memory, 1,080 s
  total timeout, and partial failure retention. Implement a local adapter
  that invokes the completed exact-artifact restore before any CIFAR
  final-source construction, takes the unchanged three-reading quiet CUDA
  gate, then runs only seeds 181/191/193 under separate hard scope caps.
  Independently audit role/feature/backbone/initial/capacity identity,
  work/relaxation/guard/sleep/deadline counts, final accuracy dispersion,
  and scoped RSS/CUDA peaks before checking p2b or P1.8p2 parent.

## 2026-09-29 — Frozen representative three-scope matched confirmation

- Completed task IDs: **P1.8p2b, P1.8p2, P1.8**. The original P1.8 fairness
  budget closes only for the declared shared-feature comparison. P3.9's two
  CUDA resume branches and deferred P2.6a remain unchecked. The checkout
  stayed on `codex/research-protocols-and-resume` at `ec7634d`; pre-existing
  dirty and unrelated files were preserved, and no commit was made.
- Before implementation, re-read AGENTS.md, the complete development plan
  and latest log, checked the actual checkout against the reviewed commit,
  and restored the frozen P1.8p1 result/manifest/journal and P1.8o request.
  No saved request, selected configuration, seed, metric, or budget was
  changed. The first new test failed at collection because the confirmation
  adapter did not yet exist. Eight new tests now cover restore/runtime
  rejection before final source, busy-GPU deferral, separate/total timeouts,
  retained partial fixed-data journal and failure artifact, gate and prior
  scope requirements, and tampered saved scope rejection.
- Added `scripts/run_cifar_representative_confirmation.py` to repeat the
  exact-artifact restore, require a new quiet CUDA gate, run the three
  frozen scopes in separate hard-limited child processes, and retain a
  synced fixed-data attempt journal plus completed wall/memory seed files.
  `scripts/audit_cifar_representative_confirmation.py` restores typed
  reports and rejects incomplete or mismatched seed/head, role, feature,
  backbone, initial-head, capacity, work, deadline, or scoped memory
  evidence before publishing a final result. Each child repeats the
  artifact/gate checks before loading data. ADR-0083 records why these
  boundaries and the inference limit were chosen.
- The unchanged confirmation ran once with seeds 181/191/193. Its quiet
  readings were 2%/3%/8% GPU utilization, with 8,248/8,220/8,199 MiB
  free; post-run was 9% and 8,176 MiB free. Fixed-data completed in
  142.232 s < 240 s, wall-time in 178.568 s < 240 s, and fresh-child
  memory in 356.843 s < 600 s; total 677.735 s < 1,080 s. The saved
  runner SHA-256 `9a9b0b1f0b31e79aab79c46ef171be5d4a038379c0ab00587dea12aedb60413f`
  still matches the source. There is no failure artifact for this success.
- The independent post-run audit restored all scope JSON from disk and
  exactly matched the final aggregate. It found nine fixed-data test rows,
  nine wall-time heads stopping at their five-second deadlines with
  recorded overshoot, and nine separate memory-child PIDs. All heads kept
  32,954 parameters and matched per-seed role, feature, backbone, and
  initial-head identity. Each fixed-data head saw 16,384 examples/512 wake
  batches; PC and circadian each made 1,024 latent steps; circadian made
  one sleep attempt; replay stayed zero. Wall-time work counts differed,
  as expected under equal time. RSS and CUDA allocated/reserved peaks were
  recorded as distinct scoped observations.
- Mean fixed-data final accuracies (backprop/PC/circadian) were
  **0.8793/0.7749/0.7087**; population standard deviations were
  0.00865/0.00487/0.01305. Five-second wall-time means were
  **0.8934/0.8757/0.8040**; standard deviations were
  0.00090/0.00509/0.00516. Fresh-child observed trainer RSS means were
  1,908,999,509/1,936,063,147/2,151,658,837 bytes. The circadian
  result was lower than matched backprop in both accuracy scopes, with no
  retuning or seed selection. These are frozen ImageNet ResNet-50 features
  from 224-pixel CIFAR-10 with a 16,384-example training subset, not a
  full-data or trainable-backbone architecture ranking. The unmatched
  image-level reference remains labeled separately.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_cifar_representative_confirmation.py tests/test_cifar_representative_restore.py tests/test_cifar_representative_selection.py tests/test_matched_head_tuning.py tests/test_repeated_head_confirmation.py tests/test_isolated_head_memory.py -q -x
  .\.venv-cuda\Scripts\python.exe -m pytest tests/test_cifar_representative_confirmation.py tests/test_cifar_representative_restore.py tests/test_cifar_representative_selection.py tests/test_matched_head_tuning.py tests/test_repeated_head_confirmation.py tests/test_isolated_head_memory.py -q -x
  .\.venv-cuda\Scripts\python.exe -m scripts.restore_cifar_representative_selection --preflight
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv-cuda\Scripts\python.exe -m scripts.run_cifar_representative_confirmation run
  .\.venv\Scripts\python.exe -m pytest tests/test_readme_figures_protocol.py -q
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\ruff.exe format --check scripts/run_cifar_representative_confirmation.py scripts/audit_cifar_representative_confirmation.py tests/test_cifar_representative_confirmation.py
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  git -c core.safecrlf=false diff --check
  ```

  Focused CPU and CUDA environments each passed **39** tests. Full CPU
  passed **997** tests with no skips; README protocol **6** passed.
  Ruff, touched-file format, mypy (**160 source files**), and diff checks
  passed. The read-only preflight again passed with six prior trials and
  zero prior final iterations. A separate Python invocation of
  `read_saved_selection()` and `audit_saved_scopes(Path('data'), restored)`
  round-tripped the saved final aggregate exactly after the run. The
  exploratory raw `asdict`/JSON assertion differed only by Python tuple
  versus JSON list representation; a JSON-normalized comparison passed.
- Ignored experiment artifacts: `data/cifar-representative-confirmation-v1-result.json`
  (212,552 bytes, SHA-256
  `705318902dc728700ed62d7f9813750261fcd1df72b2834d0621b3ffbe31088a`),
  `data/cifar-representative-confirmation-v1-gate.json`,
  `data/cifar-representative-confirmation-v1-fixed-data-attempts.jsonl`
  (18 start/completion events), `data/cifar-representative-confirmation-v1-fixed-data.json`,
  `data/cifar-representative-confirmation-v1-wall-time-seed{181,191,193}.json`,
  and `data/cifar-representative-confirmation-v1-memory-seed{181,191,193}.json`.
  The fixed-data journal SHA-256 is
  `f4f8e7a55304e055c53465006bbf5c3de08f798b17aeab279ea9a79549a3e726`;
  all per-scope artifact digests are bound in the final result. Earlier
  frozen selection and negative control artifacts were not rewritten.
- Skipped tests/work: the full CUDA pytest suite and any additional seed,
  candidate, or large sweep. The three confirmation scopes were each run
  once; there was no post-test tuning. No P3.9 CUDA checkpoint test was
  run in this session.
- Plan changes and blockers: checked p2b/p2/P1.8 only after scope, identity,
  audit, test, static, and interpretation gates passed. Updated README,
  architecture, adapter, and protocol documentation and added ADR-0083.
  P3.9b2b2b and P3.9c2b2b still need actual-device checkpoint evidence,
  but their old CPU-only-host blocker is stale: this checkout now has a
  working isolated `.venv-cuda` on an RTX 3080. Deferred P2.6a still lacks
  a deeper attribution protocol. No acceptance criterion was weakened.
- Exact next action: inspect the fixed-feature checkpoint memory payload
  and current CUDA RNG handling for P3.9b2b2b. Add a failing, budgeted
  actual-RTX-3080 fixture for pre-sleep, accepted sleep, rejected sleep,
  and incompatible CUDA state; define allocator start/peak/reset and
  cross-process segment aggregation before accepting resumed CUDA memory.
  Preserve unsupported modes until the full device gate passes, then
  address P3.9c2b2b unmatched-vision CUDA continuation separately.

## 2026-09-29 — Actual-CUDA fixed-feature checkpoint RNG gate

- Completed task ID: **P3.9b2b2b1**. P3.9b2b2b2, the b2b2b/b2b2/b2b/b2/b
  parents, P3.9c2b2b, and P3.9 remain unchecked. The previous goal turn
  made authoritative progress by completing the frozen P1.8 confirmation;
  this session continued the next open checkpoint task. The checkout
  remained `codex/research-protocols-and-resume` at `ec7634d`; existing
  dirty and unrelated files were preserved, and no commit was made.
- Re-read AGENTS.md, the living plan and latest log, inspected the actual
  checkout, ADR-0052/0061, fixed-feature runner, combined checkpoint,
  head-local split generator, RSS segment payload, and CPU resume tests.
  The isolated `.venv-cuda` reports Torch 2.14.0+cu130 and an RTX 3080.
  The first new actual-device test failed at the explicit CPU-only
  checkpoint guard, as expected. Split P3.9b2b2b into b1 learning/RNG and
  b2 allocator segments before changing behavior; original acceptance
  remains unchanged.
- `src/app/circadian_checkpoint.py` now binds CUDA checkpoints to the
  canonical device of the live head tensor and saves that device's process
  CUDA RNG alongside Python, NumPy, and Torch CPU streams. The existing
  head snapshot already saves the local CUDA split generator. A temporary
  CUDA generator validates the saved state and device before any live head,
  retry, or process RNG restoration. Optional fields preserve older CPU
  checkpoint files. `src/app/matched_head_benchmark.py` permits the tested
  fixed-epoch CUDA checkpoint route and rejects head/device alias mismatch
  before training. CUDA checkpoint memory, wall-time, and fixed-width
  capacity modes still fail before dataset loading pending b2 evidence.
- The new `tests/test_cuda_fixed_feature_checkpoint.py` uses actual CUDA
  head learning and guarded sleep. Wake, accepted-sleep, and rejected-sleep
  interruptions restore to the uninterrupted head state, including the
  local split-generator bytes; every non-timing report field and the next
  Python/NumPy/Torch CPU/CUDA draws match. Missing, malformed, and
  wrong-device CUDA states reject with unchanged head and process streams.
  A public three-head fixture opens final test only after resumed training.
  Device-alias mismatch and the three unsupported CUDA modes reject before
  training/data. An older CPU payload without either CUDA field restores.
  ADR-0084 and README, architecture, app-module, and protocol docs explain
  the new boundary and its memory limit.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv-cuda\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  .\.venv-cuda\Scripts\python.exe -c "import subprocess,sys; p=subprocess.run([sys.executable,'-m','pytest','tests/test_cuda_fixed_feature_checkpoint.py','tests/test_combined_circadian_checkpoint.py','tests/test_fixed_feature_checkpoint_resume.py','tests/test_checkpoint_memory_resume.py','-q','-x'],timeout=120); raise SystemExit(p.returncode)"
  .\.venv\Scripts\python.exe -m pytest tests/test_combined_circadian_checkpoint.py tests/test_fixed_feature_checkpoint_resume.py tests/test_checkpoint_memory_resume.py -q -x
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest tests/test_readme_figures_protocol.py -q
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\ruff.exe format --check src/app/circadian_checkpoint.py src/app/fixed_feature_checkpoint.py src/app/matched_head_benchmark.py tests/test_cuda_fixed_feature_checkpoint.py tests/test_fixed_feature_checkpoint_resume.py tests/test_combined_circadian_checkpoint.py
  git -c core.safecrlf=false diff --check
  ```

  Final focused CUDA set: **73 passed**, exit 0 under the external 120 s
  cap, completing in 27.65 s. Focused CPU checkpoint set: **62 passed**.
  Full CPU suite: **998 passed, 10 CUDA-only skips**, exit 0, before the
  final CUDA-only alias test was added; the alias and all 11 CUDA-only
  tests passed in the later focused CUDA run. Final collection counted
  **1,009** tests. README protocol **6 passed**. Ruff, touched-file format,
  mypy (**161 source files**), and diff checks passed after the last code
  change. An intermediate new unsupported-mode test failed because its
  fixed-width config omitted the pre-existing enabled split/prune contract;
  the fixture was corrected without changing production behavior.
- Experiment artifacts: trusted checkpoint files were created only under
  pytest temporary directories; no persistent GPU experiment artifact or
  new accuracy result was written. No baseline, seed, metric, or model
  setting was tuned from a final-test outcome. Full CUDA pytest and any
  sweep were skipped; actual-device work was the bounded focused set.
- Plan changes and blockers: checked b2b2b1 only after actual-device
  continuation, incompatible-state, CPU compatibility, sealed final-test,
  and static gates. Kept b2b2b2 and all parents open. No external host
  blocker remains; the missing evidence is defined CUDA allocator
  start/peak/reset and committed cross-process segment aggregation, plus
  actual-device memory, capacity, and deadline continuation. Unmatched
  whole-image CUDA continuation remains P3.9c2b2b. No acceptance criterion
  was weakened.
- Exact next action: for P3.9b2b2b2, define a typed per-invocation CUDA
  allocator segment with process/device identity, allocated/reserved starts,
  peak reset/read boundaries, and maximum absolute aggregation across
  committed segments. Add a failing, 120-second-capped two-process RTX 3080
  checkpoint-memory fixture, then implement fixed-epoch, fixed-width
  capacity, and cumulative-deadline routes with invalid-segment rejection
  before live restoration. Leave those CUDA modes unsupported until their
  actual-device evidence passes.

## 2026-09-29 — Checkpointed CUDA allocator segments and fixed-feature closure

- Completed task IDs: **P3.9b2b2b2**, then **P3.9b2b2b**, **P3.9b2b2**,
  **P3.9b2b**, **P3.9b2**, and **P3.9b** after auditing the earlier CPU and
  CUDA subgates. P3.9c2b2b and the unmatched-vision/P3.9 parents remain
  unchecked. The checkout stayed on `codex/research-protocols-and-resume` at
  `ec7634d`. Existing dirty and unrelated files were preserved; no commit
  was made.
- Re-read AGENTS.md, the full living plan and latest log, inspected the
  checkout, fixed-feature payload and runner, ADR-0061/0084, and the actual
  `.venv-cuda`/RTX 3080 runtime. A new two-process CUDA checkpoint-memory
  test first failed at the explicit unsupported-mode guard. The smallest
  change was typed allocator segments and device-specific protocol IDs,
  leaving the original CPU and noncheckpointed memory scopes intact.
- `CudaAllocatorSegment` records PID, canonical device, allocated/reserved
  starts, and absolute peaks. Each invocation synchronizes, reads starts,
  resets the PyTorch peak counter once, and synchronizes before peak reads.
  The last saved pre-save observation commits an interrupted process's
  segment; the completing process adds its final observation. Circadian
  allocated/reserved aggregates are maxima of absolute segment peaks with
  no common start; RSS retains its 5 ms sampler, max observed absolute RSS,
  and summed samples. Baselines each have one completing-process segment.
  Missing, malformed, wrong-device, and mismatched-PID CUDA tuples reject
  before live head/retry/process-RNG restoration. Fixed-epoch, cumulative
  deadline, and fixed-width capacity use distinct CUDA checkpoint-memory
  v2 protocol IDs. ADR-0085 records scope and non-attribution limits.
- The actual-device fixture restarts fixed-epoch, capacity, and deadline
  runs in independent Python processes, each capped at 35 seconds and the
  focused command capped at 120 seconds. Fixed-epoch and capacity resumed
  runs match uninterrupted trained hashes and every non-timing report
  field. Capacity metadata keeps initial/final parameter counts equal and
  guarded sleep occurs; the deadline route reaches its cumulative active
  deadline. The malformed-segment fixture fails before the monkeypatched
  head restore. A fake allocator verifies synchronize/start/reset/peak call
  order without a GPU. The prior b2b2b1 actual-device accepted/rejected
  learning/RNG tests and prior CPU wall-time/RSS/capacity tests complete the
  parent acceptance. No baseline, seed, metric, or model setting was changed
  to favor circadian.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv-cuda\Scripts\python.exe -c "import subprocess,sys; p=subprocess.run([sys.executable,'-m','pytest','tests/test_cuda_checkpoint_memory_resume.py','tests/test_cuda_fixed_feature_checkpoint.py','tests/test_cuda_allocator_contract.py','-q','-x'],timeout=120); raise SystemExit(p.returncode)"
  .\.venv\Scripts\python.exe -m pytest tests/test_cuda_allocator_contract.py tests/test_checkpoint_memory_resume.py tests/test_matched_head_benchmark.py tests/test_fixed_feature_checkpoint_resume.py -q -x
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest tests/test_readme_figures_protocol.py -q
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\ruff.exe format --check src/app/fixed_feature_checkpoint.py src/app/matched_head_benchmark.py tests/test_cuda_checkpoint_memory_resume.py tests/test_cuda_allocator_contract.py
  git -c core.safecrlf=false diff --check
  ```

  Final focused CUDA: **16 passed**, exit 0 under the external 120 s cap.
  Focused CPU checkpoint set: **48 passed**. Full CPU collection: **1,014**;
  full run **999 passed, 15 CUDA-only skipped**, exit 0. README protocol:
  **6 passed**. Ruff, four-file format, mypy (**163 source files**), and
  tracked diff checks passed after the final code change. Full CUDA pytest
  and any new scientific sweep were skipped because this gate needs only
  bounded correctness evidence.
- Experiment artifacts: trusted checkpoint files and worker JSON existed
  only in pytest temporary directories; no persistent experiment output or
  new accuracy ranking was written. The README, architecture, app-module,
  and protocol docs now describe the tested CUDA scope, with ADR-0084/0061
  linked forward to ADR-0085. The plan checks the six completed fixed-feature
  tasks only after both CUDA subgates and parent mode/report criteria passed.
  The fixed-feature work has no remaining blocker. The separate unmatched
  vision CUDA checkpoint task is unblocked on the same GPU.
- Exact next action: inspect the unmatched-vision checkpoint payload, full
  classifier snapshot, loader cursor, and process/model-local CUDA RNG
  owners for **P3.9c2b2b**. Add a failing 120-second-capped actual-device
  wake, accepted/rejected sleep, and wrong-device fixture, then implement
  continuation without changing the verified v1/v2/v3 CPU streams or
  opening final test early.

## 2026-09-29 — Actual-CUDA unmatched vision checkpoint continuation

- Completed task IDs: **P3.9c2b2b**, then **P3.9c2b2**, **P3.9c2b**,
  **P3.9c2**, **P3.9c**, and **P3.9** after auditing prior CPU/NumPy and
  fixed-feature subgates. P3.10 and later phases remain unchecked. The
  checkout remained `codex/research-protocols-and-resume` at `ec7634d`;
  existing dirty and unrelated files were preserved, and no commit was made.
- Re-read AGENTS.md, the living plan, and latest log; inspected the actual
  checkout, unmatched vision payload, file store, full classifier snapshot,
  seeded RNG fork, v1/v2 shared loader, preflight, and CPU resume fixtures.
  The previous goal turn had completed the fixed-feature CUDA gate; this
  session continued its exact next unblocked unmatched-vision task. The
  first actual RTX 3080 checkpoint test failed at the CPU-only guard as
  expected. An intermediate v1/v2 fixture incorrectly requested custom
  model order; it was corrected to the unchanged ordinary order without
  changing production behavior.
- `VisionRunnerCheckpoint` now optionally stores the canonical selected
  CUDA device and its process RNG stream. `VisionCircadianProgress` stores
  the outer entry CUDA stream beside the existing CPU stream. The full
  classifier snapshot already owns the head-local CUDA split generator.
  Seeded v3 restores the outer CPU/CUDA streams before entering its RNG
  fork, then restores the saved active process streams after the classifier
  and loader reconstruct; fork exit returns to the saved outer streams.
  V1/v2 keep their shared CPU loader generator and directly resume the
  selected-device CUDA process stream. Preflight validates device, CUDA
  byte states, full classifier, and loader cursor before live restoration;
  failed preflight restores caller Python/NumPy/Torch CPU/CUDA streams.
  Optional fields retain older CPU pickle compatibility. ADR-0086 records
  the selected-device scope and why two CUDA streams are needed for v3.
- `tests/test_cuda_vision_checkpoint_resume.py` uses eight synthetic
  examples per role and a small real CUDA classifier whose backbone
  consumes process CUDA draws. Six v1/v2/v3 wake or accepted/rejected
  sleep cases match uninterrupted checkpointed hashes, every non-timing
  report field, and next Python/NumPy/Torch CPU/CUDA draws after deliberate
  stream perturbation. Saved files expose the full classifier, local
  generator, and loader cursor. Wrong device, missing/malformed process or
  outer CUDA state, and incompatible local generator device reject before
  training with caller streams unchanged. A separate second-process wake
  test verifies durable file loading, distinct PIDs, trained hashes, and
  next draws; each child has a 35 s limit. Two CPU compatibility tests
  verify an older payload without CUDA fields and an early unavailable
  device error. These are correctness fixtures, not an accuracy study.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv-cuda\Scripts\python.exe -c "import os,subprocess,sys; env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8'}; p=subprocess.run([sys.executable,'-m','pytest','tests/test_cuda_vision_checkpoint_resume.py','tests/test_cuda_vision_checkpoint_process.py','tests/test_cuda_fixed_feature_checkpoint.py','tests/test_cuda_checkpoint_memory_resume.py','tests/test_cuda_allocator_contract.py','-q','-x'],env=env,timeout=120); raise SystemExit(p.returncode)"
  .\.venv\Scripts\python.exe -m pytest tests/test_vision_cuda_checkpoint_compatibility.py tests/test_vision_checkpoint_resume.py -q -x
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pytest tests/test_readme_figures_protocol.py -q
  .\.venv\Scripts\python.exe -m mypy src tests scripts
  .\.venv\Scripts\ruff.exe check .
  .\.venv\Scripts\ruff.exe format --check src/app/vision_checkpoint.py src/app/resnet50_benchmark.py tests/test_cuda_vision_checkpoint_resume.py tests/test_cuda_vision_checkpoint_process.py tests/test_vision_cuda_checkpoint_compatibility.py
  git -c core.safecrlf=false diff --check
  ```

  Combined focused CUDA: **24 passed**, exit 0 under an external 120 s
  cap (eight new unmatched cases and sixteen fixed-feature regressions).
  Focused CPU vision: **43 passed** across the 41 prior resume cases and two
  new compatibility cases; the prior 42-case run happened before the final
  unavailable-device test. Full CPU collection: **1,024**; final full run
  **1,001 passed, 23 CUDA-only skipped**, exit 0. README protocol **6
  passed**. Ruff, five-file format, mypy (**166 source files**), and tracked
  diff checks passed after final code formatting. Full CUDA pytest and any
  large experiment sweep were skipped as unnecessary for this bounded
  correctness gate.
- Experiment artifacts: trusted checkpoint files and JSON worker output
  were confined to pytest temporary directories. No persistent experiment
  result, new accuracy ranking, or selection decision was produced. README,
  architecture, app-module, protocol docs, and ADR-0058/0059 now point to
  ADR-0086. The plan closes P3.9 only after the prior CPU/NumPy and
  fixed-feature criteria and this actual-device subgate passed. The fixture
  uses a tiny synthetic classifier rather than full ResNet/CIFAR, so it
  proves the checkpoint contract, not full-scale performance. No baseline,
  seed, metric, model rule, protocol ID, or final-test timing was changed.
- Blockers: none for P3.9. P3.10 typed sleep telemetry remains unfinished;
  deferred P2.6a remains separately open. Exact next action: inventory
  NumPy/Torch sleep results and runner guard-report fields for P3.10, split
  the typed event contract from propagation tests while retaining every
  required field, and add a failing focused event-contract test before
  implementation.

## 2026-09-29 — Typed sleep-event contract

- Completed task IDs: **P3.10a** only. P3.10b/c and the P3.10 parent remain
  unchecked. The checkout stayed on `codex/research-protocols-and-resume` at
  `ec7634d`; existing dirty and unrelated files were preserved. No commit
  was made.
- Re-read AGENTS.md, the full living plan, current log, and actual branch/
  status. Audited both core `SleepEventResult` classes, model sleep paths,
  replay, stable IDs, runner scheduling/guards, and rollback behavior. Core
  owns budgets/lineage/replay/chemistry; runners own triggers, guard scores,
  cooldown, and durable report. The guarded Torch routes currently discard
  the rejected core result after restoring the model. This justified the
  P3.10a contract, P3.10b core capture, and P3.10c runner/persistence split
  before implementation, with the original parent criteria retained.
- Added `src/core/sleep_telemetry.py`: frozen, versioned, JSON-safe budget,
  stable-ID proposal/application, guard, replay, chemical, duration, and
  outcome records. Validation rejects nonfinite or contradictory facts;
  rollback and error outcomes keep proposals but require the final state to
  match the entry state. Scheduled prunes do not change width, and older
  pending prunes may finalize during replay. `tests/test_sleep_telemetry_contract.py`
  first failed at collection because the module did not exist, then passed
  accepted structural/no-topology, rolled-back structure/replay, skipped,
  pre-guard error, float-roundoff, and malformed cases. ADR-0087,
  `docs/modules/core.md`, and `ARCHITECTURE.md` describe the boundary. No
  model or runner emits telemetry yet; baseline, seed, metric, and learning
  behavior were not changed.
- Commands and final outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_sleep_telemetry_contract.py -q -x  # red: ModuleNotFoundError
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_sleep_telemetry_contract.py  # 19 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q  # 1,020 passed, 23 skipped
  .\.venv\Scripts\python.exe -m mypy src tests scripts  # success, 168 source files
  .\.venv\Scripts\ruff.exe check .  # all checks passed
  .\.venv\Scripts\ruff.exe format --check src/core/sleep_telemetry.py tests/test_sleep_telemetry_contract.py  # 2 files already formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- The final full CPU suite took 237.77 s. The 23 skips are existing
  actual-CUDA tests; they were not run because this increment is pure value
  validation with no device code. No scientific sweep was run. Experiment
  artifacts: none; no accuracy ranking or selection decision was produced.
- Plan changes: P3.10 was split into a pure contract, two-backend core fact
  capture, and runner/resume persistence because their facts have different
  owners. Marked only P3.10a complete after focused, full, and static gates.
  P3.10b/c and the parent keep every required field and exit test open.
- Blockers: none for P3.10b; deferred P2.6a remains separately open. Exact
  next action: add a failing focused NumPy core test for forced split,
  delayed-prune/replay finalization, replay-only, no-op, and skipped facts.
  Then add compare-excluded core telemetry and measure exact replay examples/
  updates and core duration without changing learning state or existing
  result equality; verify Torch separately before checking P3.10b.

## 2026-09-29 — Model-owned NumPy and Torch sleep facts

- Completed task IDs: **P3.10b1**, **P3.10b2**, then their **P3.10b** parent
  after both backend gates. P3.10c1–c5 and P3.10 remain unchecked. The
  checkout stayed on `codex/research-protocols-and-resume` at `ec7634d`;
  preexisting dirty/unrelated changes were preserved. No commit was made.
- Re-read AGENTS.md, the living plan, current log, and actual checkout.
  Reconciled the prior contract with NumPy replay and Torch's post-split
  pruning. Before implementation, split P3.10b into backend subgates and
  retained all parent fields. NumPy can finalize an older pending prune
  during replay; Torch can immediately prune a new child and has no replay
  path under P3.1b. The replay-only fixture therefore applies to NumPy,
  while every Torch core record must explicitly report zero replay. The
  plan states this capability interpretation rather than silently inventing
  a Torch replay feature.
- Added compare-excluded `telemetry` to both `SleepEventResult` classes.
  NumPy now reports resolved split/prune/replay limits, stable split pairs,
  selected/scheduled/removed and applied prune IDs, exact replay examples
  and updates, pre/post primary/fast/slow chemistry, width, and core
  duration. Torch records pairs before post-split pruning, so a child
  removed in the same event remains identifiable; it reports zero replay.
  Both backends return explicit applied or skipped reasons and leave runner
  guard fields absent. Telemetry is never stored in model snapshots or
  deterministic result equality. Construction stays inside the existing
  atomic sleep boundary, so a failed telemetry build restores model state
  and local RNG. ADR-0088/0089, README, and core module docs record this.
- `tests/test_numpy_sleep_telemetry.py` first failed on the missing field,
  then passed 9 split, immediate/delayed prune, older-pending finalization,
  replay-only, executed no-op, four skip, JSON, budget, chemistry, and
  injected-failure continuation cases. A test fixture initially used
  component switches under legacy mode; the fixture was corrected to the
  unchanged valid legacy configuration. `tests/test_torch_sleep_telemetry.py`
  first failed on the missing Torch field, then passed 9 CPU cases for
  split, parent/child removal, prune-only and minimum-width no-op, four
  skips, zero replay, JSON, chemistry, and injected-failure continuation.
  Its tiny actual-device CUDA case passed on the local RTX 3080.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_numpy_sleep_telemetry.py -x  # red: missing telemetry; final 9 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_torch_sleep_telemetry.py -x  # red: missing telemetry; final 9 passed, 1 CUDA skip
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_numpy_sleep_telemetry.py tests/test_atomic_sleep_core.py tests/test_numpy_sleep_components.py tests/test_sleep_event_lineage.py tests/test_prune_metadata_alignment.py tests/test_sleep_clocks.py tests/test_toy_checkpoint_resume.py tests/test_continual_checkpoint_resume.py  # 76 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_torch_sleep_telemetry.py tests/test_torch_sleep_components.py tests/test_torch_neuron_lineage.py tests/test_atomic_sleep_core.py tests/test_torch_full_snapshot.py tests/test_vision_checkpoint_resume.py  # 83 passed before final new GPU-only/chemistry assertions
  .\.venv-cuda\Scripts\python.exe -c "import os,subprocess,sys; env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8'}; p=subprocess.run([sys.executable,'-m','pytest','-o','addopts=','-q','-x','tests/test_torch_sleep_telemetry.py','tests/test_cuda_vision_checkpoint_resume.py','tests/test_cuda_vision_checkpoint_process.py','tests/test_cuda_fixed_feature_checkpoint.py'],env=env,timeout=120); raise SystemExit(p.returncode)"  # 29 passed in 17.46 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q  # final 1,038 passed, 24 skipped in 242.79 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_readme_figures_protocol.py  # 6 passed
  .\.venv\Scripts\python.exe -m mypy src tests scripts  # success, 170 source files
  .\.venv\Scripts\ruff.exe check .  # all checks passed
  .\.venv\Scripts\ruff.exe format --check src/core/circadian_predictive_coding.py src/core/resnet50_variants.py tests/test_numpy_sleep_telemetry.py tests/test_torch_sleep_telemetry.py  # 4 files already formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- The final CPU suite's 24 skips comprise actual-CUDA fixtures, including
  the new hardware case; the bounded CUDA set above exercised the new case
  and checkpoint continuation on the actual device. Full CUDA pytest and
  scientific sweeps were skipped because this gate required only bounded
  correctness and reproducibility evidence. Experiment artifacts: no
  persistent runs or accuracy rankings; checkpoint files stayed in pytest
  temporary directories. No baseline, seed, metric, training rule, or
  final-test timing was changed.
- Plan changes: completed b1/b2/b only after their backend and full gates;
  split P3.10c into toy, continual, fixed-feature, vision, and cross-runner
  artifact/exit gates because each has a separate report/checkpoint owner.
  The original P3.10 trigger/guard/persistence fields and Phase 3 exit gate
  remain unchecked. Blockers: none for P3.10c1; deferred P2.6a remains
  separately open. Exact next action: write a failing toy runner fixture for
  periodic/adaptive/skipped event sequences through wake/before-sleep/
  after-sleep checkpoint resume, model-order variation, JSON output, and
  sealed final-test timing. Then attach runner-owned trigger, reason, and
  attempt duration to the NumPy core record without changing legacy counts
  or baseline trajectories.

## 2026-09-29 — Toy NumPy sleep-event history and JSON result

- Completed task ID: **P3.10c1**. P3.10c2–c5, P3.10, deferred P2.6a,
  and later phases remain open. The checkout remained on
  `codex/research-protocols-and-resume` at `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`;
  preexisting dirty and unrelated changes were preserved. No commit was made.
- Re-read AGENTS.md, the active plan/handoff and current log, and inspected
  the actual checkout. The new toy test first failed because the report had
  no event sequence. The runner now records exactly one typed sleep decision
  per completed epoch, including periodic, adaptive, both, not-due, and
  disabled cases. Actual attempts retain the model's split/prune/replay,
  chemistry, budget, width, and core duration; the runner adds its trigger
  and measured attempt duration. Unscheduled epochs read chemistry without
  calling core sleep. A second scheduled legacy call in the fixed fixture
  made no topology change, so its attempt was recorded while the historical
  event count remained one. No learning, seed, baseline, metric, protocol,
  or final-test timing changed.
- Version-2 toy checkpoints persist complete typed event history at wake,
  before-sleep, and after-sleep boundaries. Preflight rejects version-1 or
  incomplete/incorrectly indexed histories before training. The existing
  checksum envelope is unchanged. The baseline CLI's `--json-result` writes
  a complete finite JSON report after final scoring to a new local path and
  rejects existing paths before training. ADR-0090, README, and app/infra
  module docs explain the boundary and version choice.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_toy_sleep_telemetry.py -x  # red: missing CircadianSleepSummary.events; final 10 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_toy_checkpoint_resume.py tests/test_experiment_runner.py tests/test_sleep_event_accounting.py tests/test_numpy_sleep_telemetry.py tests/test_toy_sleep_telemetry.py  # 52 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_toy_sleep_telemetry.py tests/test_toy_checkpoint_resume.py tests/test_sleep_event_accounting.py tests/test_experiment_runner.py tests/test_readme_figures_protocol.py  # final focused 50 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q  # 1,048 passed, 24 skipped in 248.10 s
  .\.venv\Scripts\python.exe -m mypy src tests scripts  # success, 173 source files
  .\.venv\Scripts\ruff.exe check .  # all checks passed
  .\.venv\Scripts\ruff.exe format --check src/app/toy_sleep_decisions.py src/app/toy_checkpoint.py src/app/experiment_runner.py src/core/circadian_predictive_coding.py src/infra/toy_result_files.py src/adapters/cli.py tests/test_toy_sleep_telemetry.py  # 7 files already formatted at the P3.10c1 gate; helper renamed in P3.10c2a
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- The 24 full-suite skips are actual-CUDA fixtures; no CUDA run was needed
  for this NumPy-only increment. Large sweeps and full CUDA pytest were
  skipped. Experiment artifacts: no persistent study results or rankings;
  JSON and checkpoint files existed only in pytest temporary directories.
- Plan changes: checked c1 only after focused, full, and static acceptance;
  retained c2–c5, c/P3.10, and their guard/cooldown/exit criteria. The
  version-2 checkpoint choice is explicit because version 1 cannot reconstruct
  a complete prior event sequence. Blockers: none for c2; deferred P2.6a
  remains separately open. Exact next action: inspect ordinary and
  checkpointed continual NumPy scheduling, guard, report, and persistence
  across supported protocols; write a failing focused fixture for accepted,
  rejected, and skipped events with retained proposed core facts, disjoint
  guard scores, and phase A/B interruption/resume before changing that runner.

## 2026-09-29 — Historical continual NumPy sleep history

- Completed task ID: **P3.10c2a**. P3.10c2b/c, c2 parent, c3–c5,
  P3.10, deferred P2.6a, and later phases remain unchecked. Branch
  `codex/research-protocols-and-resume` stayed at
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; preexisting dirty and
  unrelated changes were preserved. No commit was made.
- Re-read AGENTS.md, the active plan and log, and inspected the checkout.
  The historical continual v0–v5 ordinary/checkpointed paths never pass an
  inner guard, although their low-level helper can accept one. Arrived v6/v7
  own a real disjoint inner role. Split P3.10c2 into historical schedule,
  arrived guard, and v7 selection gates without dropping any parent field.
- The new fixture first failed on missing per-seed sleep events. Each v0–v5
  run now emits one typed event per completed global epoch with phase-local
  periodic/adaptive/not-due/disabled decision, the NumPy core facts when
  called, outer attempt duration, and explicit absent guard. The existing
  sleep call, historical counters, baseline updates, metrics, seeds,
  protocols, and final-test release timing are unchanged. The toy route now
  shares the read-only NumPy schedule adapter. A generic infra writer adds
  complete finite `--json-result` output to the continual CLI while keeping
  its text output and overwrite protection.
- An initial design put measured events in `_ContinualTrainingState` and
  failed the existing final-label trained-state hash test. The final design
  keeps history beside state: ordinary pending seed/report, active trusted
  checkpoint, completed result, or v5 unscored seed record. It preserves
  model-state serialization. A separate history extension version on the
  existing v0–v5 checkpoint formats rejects old eventless files; preflight
  validates contiguous global epoch/wake indexes at wake/before/after-sleep
  cursors before training. Existing format numbers and protocol IDs stay.
  ADR-0091, README, and app/infra module docs record the decision.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_sleep_telemetry.py -x  # red: missing CircadianShiftReport.sleep_events; then 18 passed; final 27 passed before committed-seed cases
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_sleep_telemetry.py tests/test_continual_checkpoint_resume.py tests/test_continual_global_test_seal.py tests/test_toy_sleep_telemetry.py -x  # first failed: variable timing changed trained-state hash; after moving history beside state, 77 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests -k continual -x  # first failed on an exact helper-argument seal; after asserting the new callback while retaining train-role checks, 251 passed, 848 deselected
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q  # 1,075 passed, 24 skipped in 244.70 s, before two additional committed-seed fixtures
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_sleep_telemetry.py tests/test_readme_figures_protocol.py -x  # post-add 35 passed
  .\.venv\Scripts\python.exe -m mypy src tests scripts  # success, 175 source files
  .\.venv\Scripts\ruff.exe check .  # all checks passed
  .\.venv\Scripts\ruff.exe format --check src/app/numpy_sleep_decisions.py src/app/experiment_runner.py src/app/continual_shift_benchmark.py src/app/continual_checkpoint.py src/infra/local_result_json.py src/infra/toy_result_files.py scripts/run_continual_shift_benchmark.py tests/test_continual_sleep_telemetry.py tests/test_continual_shift_benchmark.py  # 9 files already formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- The 24 full-suite skips are actual-CUDA fixtures. No CUDA run or large
  scientific sweep was needed for this NumPy-only change. Experiment
  artifacts: none persistent; JSON and checkpoint files stayed in pytest
  temporary directories. No ranking or selection decision was produced.
- Plan changes: added c2a/c2b/c2c because guard facts exist only on arrived
  v6/v7; checked c2a after six-protocol, resume, role-seal, broad, full, and
  static evidence. C2b/c and the parent keep guard scores, rollback/error
  proposal facts, v7 selection, and cross-protocol audit open. Blockers:
  none for c2b; P2.6a remains separately deferred. Exact next action:
  inspect v6 ordinary `_RoleAudit`, active v6 checkpoint validation, and
  event digest; write a failing accepted/rejected/skip phase A/B fixture that
  retains proposed core facts on guard rollback and matches ordinary versus
  interrupted/resumed typed events before modifying v6 or v7 behavior.

## 2026-09-29 — Arrived v6 guarded sleep history and trusted restart

- Completed task IDs: **P3.10c2b1** and **P3.10c2b2**. P3.10c2b3/c2b,
  c2c/c2 parent, c3–c5, P3.10, deferred P2.6a, and later phases remain
  unchecked. Branch `codex/research-protocols-and-resume` stayed at
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; the preexisting dirty
  working tree and unrelated changes were preserved. No commit was made.
- Re-read AGENTS.md, the plan handoff and current log, and inspected the
  checkout. The v6 ordinary helper recorded guard scores in `_RoleAudit`
  but returned no typed sleep events. The checkpointed transaction similarly
  persisted only the role ledger. The v7 selection trial digest depends on
  that existing ledger and must remain stable.
- A new ordinary fixture first failed with zero per-seed events; the first
  checkpoint fixture then failed with empty resumed history. The adapter now
  attaches the actual phase inner-guard hash and two measured accuracy
  evaluations to each attempted NumPy core event. It leaves cross-entropy
  absent, preserving the old work count and guard decision. On rejection,
  it retains core proposed split/prune/replay facts and duration, clears
  applied effects, and records the restored final width/chemistry. The focused
  rollback fixture verifies a real split and two replay examples/one replay
  update remain proposed with zero applied work. Schedule
  misses remain explicit skipped events. The v6 smoke JSON includes the
  events. No baseline, seed, metric, role release, or sleep decision changed.
- Typed history lives beside trained state in ordinary pending seeds, active
  checkpoint cursors, and completed unscored records. A version-one history
  extension and separate digest bind it to the unchanged role-event ledger;
  preflight validates typed invariants, contiguous global epochs, phase-local
  trigger, inner role hash, score/tolerance/acceptance, and active/completed
  cursor length before model restoration or final scoring. Old/missing and
  tampered histories fail before updates. The existing format-6 identity,
  role event digest, and v7 trial digest stay unchanged. ADR-0092, README,
  architecture, and app/core module docs record this choice.
- Commands and outcomes from the workspace root:

  ```powershell
  python -m pytest -q tests/test_continual_arrived_sleep_telemetry.py  # collection failed: shell `python` resolves to Anaconda 3.9, lacking typing.Self; no repo code defect
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_sleep_telemetry.py -x  # red: ordinary v6 returned zero events; then red: checkpointed resume returned empty history; final 20 passed, repeated after adding replay proposal assertion
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_checkpoint.py tests/test_continual_arrived_selection.py tests/test_continual_arrived_selection_checkpoint.py -x  # first failed on changed config-error wording; restored wording, 97 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_sleep_telemetry.py tests/test_continual_arrived_checkpoint.py tests/test_continual_arrived_selection.py tests/test_continual_arrived_selection_checkpoint.py -x  # 117 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_sleep_telemetry_contract.py tests/test_readme_figures_protocol.py -x  # 26 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q  # 1,098 passed, 24 CUDA-only skipped in 252.72 s
  .\.venv\Scripts\python.exe -m scripts.run_continual_arrived_smoke  # finite JSON with typed sleep_events; exit 0
  .\.venv\Scripts\python.exe -m mypy src tests scripts  # success, 177 source files
  .\.venv\Scripts\ruff.exe check .  # all checks passed
  .\.venv\Scripts\ruff.exe format --check src/core/sleep_telemetry.py src/app/numpy_sleep_decisions.py src/app/continual_shift_benchmark.py src/app/continual_arrived_benchmark.py src/app/continual_arrived_checkpoint.py src/app/continual_arrived_transactions.py src/app/continual_arrived_sleep_history.py scripts/run_continual_arrived_smoke.py tests/test_continual_arrived_sleep_telemetry.py tests/test_sleep_telemetry_contract.py  # 10 changed files formatted
  git diff --check  # exit 0; Git reported existing LF/CRLF worktree conversion notices
  ```

- `python scripts/run_continual_arrived_smoke.py` also failed because direct
  script execution does not put the repository root on `sys.path`; the
  documented `python -m scripts.run_continual_arrived_smoke` path passed.
  A broad `ruff format --check .` found 39 already unformatted repository
  files, including Markdown code fences and unrelated user changes; it was
  deliberately not used to mass-reformat them. Targeted changed-code
  formatting passed. No actual CUDA test ran in this NumPy-only increment.
  Experiment artifacts: no persistent files; checkpoint/JSON cases used
  pytest temporary paths and the smoke emitted stdout only. No sweep,
  retuning, baseline change, or ranking result was produced.
- Plan changes: split c2b into ordinary guarded decisions (b1), active and
  completed checkpoint history (b2), and error/retry representation (b3).
  Marked b1/b2 complete only after red/green focused, existing v6/v7,
  full-suite, and static evidence. The original c2b error requirement and
  c2c selection propagation remain unchecked; the split preserves their
  acceptance criteria. There is no external blocker. Exact next action:
  inspect `_apply_scheduled_sleep` at core/post-guard exceptions and the v6
  before-sleep transaction, then write a failing fixture for a typed error
  attempt plus retry of the same epoch in ordinary/checkpointed history.
  Define explicit multi-attempt epoch indexing, retain any returned core
  proposal on a post-guard error, and preserve fail-loud behavior and
  all-seed final-test sealing before implementing P3.10c2b3.

## 2026-09-29 — Failed arrived sleep attempts and same-epoch retry

- Completed task IDs: **P3.10c2b3** and the **P3.10c2b v6 parent**.
  P3.10c2c/c2, c3–c5, P3.10, deferred P2.6a, and later phases remain
  unchecked. The previous goal turn was progress: it completed c2b1/b2 and
  changed the authoritative v6 runner/checkpoint history. This continuation
  re-read AGENTS.md, the active plan and log, and inspected the same dirty
  checkout at `ec7634d174e98a83b6623a5cdb3c3960f215c4cb` on
  `codex/research-protocols-and-resume`. No commit was made, and preexisting
  unrelated user changes were preserved.
- Direct core/post-guard error tests first failed because `_apply_scheduled_sleep`
  emitted no event. After adding typed partial guard facts, a checkpointed
  error-resume test failed at the old one-event-per-epoch validator. The
  ordinary opt-in retry test first failed on an absent API parameter. Those
  red cases now pass. The runner snapshots before guard scoring, restores on
  pre/core/post errors, emits an `error` attempt with role hash, known score,
  canonical reason, and elapsed attempt duration, then raises by default.
  A post-guard error retains the returned split/replay proposal and measured
  core duration with zero applied work. A core exception has no returned
  proposal; zero core seconds denotes unavailable core timing while the
  measured outer attempt duration remains recorded. No cross-entropy or
  missing accuracy/delta is fabricated.
- Checkpointed v6 saves the failed event at the same `before_sleep` cursor,
  without advancing wake or guard-decision counts. A later explicit resume
  retries that epoch. Ordinary v6 can opt into a nonnegative bounded
  `sleep_error_retries` value; the default still raises, and a checkpoint
  store rejects local retries in favor of explicit resume. Completed epoch
  history now validates zero or more errors followed by one final decision;
  a pending before-sleep epoch permits only errors. Role hash, phase-local
  trigger, score/reason consistency, typed facts, and digest remain checked
  before model restoration or final scoring. No new checkpoint field or
  format was needed; existing version-one histories remain valid. The old
  role ledger and v7 trial digest are unchanged (ADR-0093).
- Evidence covers pre/post nonfinite and core/post exceptions, real returned
  split and replay proposals after post-guard failure, two separate errors
  at one epoch, phase A/B and both model orders, two-seed ordinary retry,
  checkpoint terminal reload, JSON-safe partial values, no final source
  release on failure, malformed error role/reason rejection before restore,
  unchanged legacy counters and guard ledgers, and exact baseline/circadian
  model fields after resume. A raw pickle of the entire replay-memory deque
  differed despite equal contents; comparing each replay snapshot array and
  metadata plus all other model fields confirmed state parity. The full
  suite had already passed before that stronger test assertion; its eight
  checkpoint error cases passed afterward.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_sleep_telemetry.py -x  # red: no error event; later red: active history length; then 36 passed before broader failure stages
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_sleep_telemetry.py::test_ordinary_v6_can_explicitly_retry_one_failed_guard_attempt -x  # red: unexpected sleep_error_retries argument; then passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_sleep_telemetry.py tests/test_sleep_telemetry_contract.py -x  # 66 passed before final core/phase expansion
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests -k continual -x  # 289 passed, 850 deselected
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_sleep_telemetry.py tests/test_sleep_telemetry_contract.py tests/test_continual_arrived_checkpoint.py tests/test_continual_arrived_selection.py tests/test_continual_arrived_selection_checkpoint.py -x  # 163 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q  # 1,124 passed, 24 CUDA-only skipped in 256.58 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_sleep_telemetry.py::test_checkpointed_error_attempt_is_kept_before_retry_and_final_scoring -x  # first raw-deque-pickle assertion failed; corrected semantic replay comparison, 8 passed
  .\.venv\Scripts\python.exe -m mypy src tests scripts  # success, 177 source files after fixing one new test type-ignore
  .\.venv\Scripts\ruff.exe check .  # all checks passed
  .\.venv\Scripts\ruff.exe format --check src/core/sleep_telemetry.py src/app/numpy_sleep_decisions.py src/app/continual_shift_benchmark.py src/app/continual_arrived_benchmark.py src/app/continual_arrived_sleep_history.py src/app/continual_arrived_transactions.py tests/test_continual_arrived_sleep_telemetry.py tests/test_sleep_telemetry_contract.py  # 8 changed files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Skipped tests: the full suite's 24 actual-CUDA fixtures; no CUDA run was
  necessary for this NumPy-only increment. Experiment artifacts: none
  persistent. Checkpoint files were pytest temporary artifacts; no sweeps,
  seeds, baselines, objectives, or selection metrics were tuned or changed.
  README, architecture/module docs, and ADR-0093 describe the retry and
  failure record. The optional ordinary retry is bounded and uses the same
  trained model/role/final-test boundaries; checkpointed runs use explicit
  resume. There is no external blocker.
- Plan changes: checked c2b3 and c2b only after direct, active/completed,
  both-order A/B, two-seed, state parity, full-suite, and static evidence.
  Kept c2c/c2 and later runner gates open; none of their v7/Torch criteria
  were weakened. Exact next action: inspect v7 ordinary candidate training
  and checkpointed `selection_resume` propagation, then write a failing
  two-candidate/two-seed fixture requiring each candidate's typed v6 sleep
  history in ordinary and resumed v7 artifacts. Preserve trial digest,
  choice objective, final-test timing, and baseline work while covering
  accepted/rejected/error/skip phase A/B sequences before closing c2c.

## 2026-09-29 — v7 candidate and selected sleep history

- Completed task: **P3.10c2c1**. The checkout remains on
  `codex/research-protocols-and-resume` at `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`;
  cumulative tracked and untracked research changes were preserved without
  a commit. The previous v6 b3 increment also finished a combined focused
  verification of 169 passed after its strengthened model parity assertion.
- A red two-candidate/two-seed v7 test first failed because
  `ArrivedOuterSelectionResult` had no `candidate_sleep_histories`. The
  selector now returns one typed history per candidate/seed and sends the
  chosen circadian candidate's history into final seed metrics. Ordinary,
  fresh checkpointed, and candidate-boundary resumed runs retain the same
  phase A/B non-timing facts in both model orders. Completed checkpoint
  records already carry their validated v6 histories; this increment did
  not change the v7 format or trial/freeze digest semantics. Measured
  durations remain compare-excluded. The smoke script prints strict JSON
  with candidate and selected histories (ADR-0094).
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_selection_checkpoint.py::test_selection_exposes_candidate_and_chosen_sleep_history -x  # red: missing candidate_sleep_histories; then 2 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_selection.py tests/test_continual_arrived_selection_checkpoint.py tests/test_continual_arrived_sleep_telemetry.py -x  # first exposed an old positional-only test spy; after forwarding keyword args, 91 passed
  .\.venv\Scripts\python.exe -m scripts.run_continual_arrived_selection_smoke | .\.venv\Scripts\python.exe -c "import json,sys; report=json.load(sys.stdin); assert len(report['candidate_sleep_histories']) == 4; assert len(report['final_seed_scores']) == 2; assert all(len(row['selected_sleep_events']) == 2 for row in report['final_seed_scores']); print('v7 smoke JSON: 4 candidate histories, 2 selected histories')"  # passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q  # 1,126 passed, 24 skipped in 256.08 s
  .\.venv\Scripts\python.exe -m mypy src tests scripts  # success, 177 source files
  .\.venv\Scripts\ruff.exe check .  # all checks passed
  .\.venv\Scripts\ruff.exe format --check src/app/continual_arrived_selection.py scripts/run_continual_arrived_selection_smoke.py tests/test_continual_arrived_selection.py tests/test_continual_arrived_selection_checkpoint.py  # one new script line needed formatting, then all four passed
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Skipped tests: the full suite's 24 actual-CUDA cases; no device-specific
  code changed. Experiment artifacts: none persistent; selection checkpoints
  lived under pytest temporary paths. No sweep, seed choice, baseline,
  objective, score, or final-test release order changed. A repository-wide
  Ruff formatter check during the session flagged 39 files, including
  Markdown examples; touched Python files pass after formatting the script.
  There is no external blocker.
- Plan changes: split c2c into public candidate/final exposure (c2c1) and
  the remaining trial/freeze provenance and guard/count gate (c2c2).
  Checked c2c1 only after red/green, interrupted restart, strict JSON,
  full CPU, and static evidence. Kept c2c2/c2c/c2 unchecked with all
  original acceptance criteria. Exact next action: write a failing
  two-candidate/two-seed v7 test requiring complete history on each
  circadian trial and independently bound frozen checkpoint provenance.
  Reject active/completed/frozen tampering before update or final release,
  then cover accepted/rejected/error/skip phase A/B attempts and count
  reconciliation while preserving the historical trial digest and choice.

## 2026-09-29 — v7 trial and freeze sleep provenance

- Completed tasks: **P3.10c2c2, P3.10c2c, and P3.10c2**. The checkout remains
  on `codex/research-protocols-and-resume` at
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`. Cumulative unrelated
  tracked/untracked changes were preserved; no commit was made.
- A red two-candidate/two-seed fixture failed on the missing trial
  `sleep_events` field. Circadian trials now carry the full v6 typed history
  and its role-ledger-bound digest; baseline trials carry explicit empty
  fields. Candidate records and the frozen choice bind the ordered histories
  with separate digests. `_trial_digest` hashes only the historical score/work
  fields, so the trial and choice identities and objective remain unchanged.
  Checkpoint format 8 rejects earlier format-7 files without trial/freeze
  provenance (ADR-0095). The smoke command produces strict JSON with four
  circadian trial histories and the freeze sleep digest.
- A red ordinary v7 error fixture failed on the absent opt-in retry
  argument. Ordinary v7 now passes a bounded nonnegative retry limit to
  the existing v6 typed-error path; its default still raises. Checkpointed
  v7 rejects local retries and persists the failed nested v6
  `before_sleep` cursor for explicit resume. One fixed fixture covers a
  phase-A core error, same-epoch rejected retry, phase-B accepted attempt,
  and skips in both phases. It uses two candidates and two seeds in both
  model orders. Ordinary and resumed non-timing candidate histories,
  selected metrics, frozen choices, and exact candidate baseline/circadian
  model fields match. Failed guard examples remain in telemetry; the
  historical trial guard-exposure counter counts completed two-pass
  decisions only. Accepted, rejected, error, and skipped events reconcile
  with guard decisions and legacy sleep counters.
- Active history tamper with a forged accepted reason and recomputed digest
  reached a model update, exposing a missing v6 resolved-reason preflight.
  The validator now checks canonical reasons for accepted, rolled-back,
  guarded-skipped, disabled, and not-due outcomes. The expanded v7 tamper
  matrix rejects old format and active, completed-trial, candidate, and
  frozen history changes before a resumed update or final release, even
  when the edited value remains well typed. No final-test source opened
  during the failed first attempt.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_selection_checkpoint.py::test_selection_binds_each_trial_history_to_frozen_checkpoint -x  # red: absent trial sleep_events; then 2 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_selection_checkpoint.py::test_selection_reconciles_error_rejection_acceptance_and_skips_across_resume -x  # red: absent sleep_error_retries; then 2 passed, including the later non-timing parity assertion
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_selection_checkpoint.py::test_should_reject_tampered_selection_before_update_or_final_release -x  # red active_sleep allowed a resumed update; then 32 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_continual_arrived_selection.py tests/test_continual_arrived_selection_checkpoint.py tests/test_continual_arrived_sleep_telemetry.py tests/test_continual_arrived_checkpoint.py -x  # 164 passed
  .\.venv\Scripts\python.exe -m scripts.run_continual_arrived_selection_smoke | .\.venv\Scripts\python.exe -c "import json,sys; row=json.load(sys.stdin); trials=[x for x in row['trials'] if x['method']=='circadian_predictive_coding']; assert len(trials)==4 and all(x['sleep_events'] and len(x['sleep_history_digest'])==64 for x in trials); assert len(row['freeze']['sleep_history_digest'])==64; print('v7 trial/freeze JSON: 4 histories and independent digest')"  # passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q  # 1,146 passed, 24 skipped in 259.32 s
  .\.venv\Scripts\python.exe -m mypy src tests scripts  # success, 177 source files
  .\.venv\Scripts\ruff.exe check .  # all checks passed
  .\.venv\Scripts\ruff.exe format --check src/app/continual_arrived_selection.py src/app/continual_arrived_selection_checkpoint.py src/app/continual_arrived_selection_resume.py src/app/continual_arrived_sleep_history.py tests/test_continual_arrived_selection_checkpoint.py  # five changed Python files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Skipped tests: the full suite's 24 actual-CUDA cases, which concern
  other device routes; no CUDA result was inferred from CPU. Experiment
  artifacts: none persistent. Checkpoints were pytest temporary files;
  there was no sweep or adjustment to seeds, baselines, selection metrics,
  objective, or final-test release order. No external blocker exists.
- Plan changes: checked c2c2, c2c, and c2 only after ordinary/resumed,
  tamper, strict JSON, count, exact-state, full CPU, and static evidence.
  P3.10c/P3.10 remain open with their original Torch and artifact criteria.
  Exact next action: inspect fixed-feature Torch guard/cooldown scheduling,
  report, local JSON, and CPU/CUDA checkpoint state, then write a red bounded
  CPU fixture for typed accepted/rejected/skipped/error events with exact
  disjoint guard counts. Keep CUDA acceptance open until run on hardware.

## 2026-09-29 — fixed-feature Torch completed sleep decisions

- Completed tasks: **P3.10c3a and P3.10c3b1**. Branch
  `codex/research-protocols-and-resume`, HEAD
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; cumulative tracked and
  untracked user/session changes remain in place. No commit or sweep was made.
- The red `tests/test_torch_runner_sleep_decisions.py` collection failed on
  the missing app module; the red CPU checkpoint fixture failed on the
  missing report history. Added `src/app/torch_sleep_decisions.py` and a
  read-only Torch chemistry-summary method. Periodic, adaptive, combined,
  disabled, not-due, and cooldown decisions now produce typed facts. Guarded
  results carry the inner-guard feature/label hash, measured pre/post
  accuracy and cross-entropy, selected delta/tolerance, exact completed
  two-pass exposure, attempt duration, and stable-ID proposal after rollback.
  The adapter does not mutate the head. No per-sleep time cap exists, so
  `time_limit_seconds` remains `None`; the distinct wall-time head deadline
  remains in the benchmark result (ADR-0096).
- `_train_circadian_head` now records completed decisions in the circadian
  report; baseline head histories are empty. A format-2 trusted
  fixed-feature checkpoint carries the ordered history beside model/RNG
  state. Pre-restore checks bind its length to wake/before/after-sleep
  cursors, each event's epoch and wake clock, guard role hash, and attempt,
  rollback, cooldown, guard-exposure, split, and prune counters. The CPU
  accepted/rejected/cooldown fixture matches ordinary and resumed non-timing
  histories. Disabled/not-due/warmup cases, strict public report JSON,
  sealed final-test timing, and tampered format/missing event/role/clock/
  counter rejection all pass before an update. Existing fixed-width and
  checkpoint-memory comparisons still check every non-timing event fact.
- The first full suite had 11 failures: four capacity cases compared
  variable durations or forced a rollback delta inconsistent with guard
  scores; seven retry cases replaced the guarded function with a spy that
  returned no typed telemetry. The fixtures now supply real changed guard
  scores and typed spy results, preserving the original learning/cooldown
  assertions. Targeted reruns passed, then the full suite passed.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_torch_runner_sleep_decisions.py -x  # red: missing module; then 5 passed, expanded cases included in full suite
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_fixed_feature_checkpoint_resume.py::test_fixed_feature_sleep_history_survives_guarded_after_sleep_resume -x  # red: missing sleep_events; then passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_fixed_feature_checkpoint_resume.py::test_fixed_feature_runner_records_unattempted_and_core_skips tests/test_fixed_feature_checkpoint_resume.py::test_sleep_history_tamper_rejects_before_fixed_feature_update -x  # 8 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_fixed_feature_checkpoint_resume.py::test_public_matched_route_resumes_before_final_test -x  # passed, strict report JSON and baseline-empty histories
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_fixed_feature_checkpoint_resume.py tests/test_checkpoint_memory_resume.py tests/test_cuda_fixed_feature_checkpoint.py tests/test_cuda_checkpoint_memory_resume.py tests/test_torch_runner_sleep_decisions.py -x  # 38 passed, 15 actual-CUDA skips
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q  # first full gate: 11 failed, 1,153 passed, 24 skipped; fixtures corrected
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_matched_head_capacity.py::test_fixed_width_capacity_checkpoint_preserves_guarded_sleep_and_final_test -x  # 4 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_sleep_retry_runners.py -x  # 20 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,164 passed, 24 skipped in 259.53 s
  .\.venv\Scripts\python.exe -m mypy src tests scripts  # success, 179 source files
  .\.venv\Scripts\ruff.exe check .  # all checks passed
  .\.venv\Scripts\ruff.exe format --check src/app/torch_sleep_decisions.py src/app/matched_head_benchmark.py src/app/fixed_feature_checkpoint.py src/core/resnet50_variants.py tests/test_torch_runner_sleep_decisions.py tests/test_fixed_feature_checkpoint_resume.py tests/test_checkpoint_memory_resume.py tests/test_cuda_fixed_feature_checkpoint.py tests/test_matched_head_capacity.py tests/test_sleep_retry_runners.py  # 10 files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # 2.14.0+cpu False
  ```

- Skipped tests: 24 actual-device CUDA tests in the full suite; the host has
  CPU-only Torch, so no CUDA telemetry acceptance is claimed. Experiment
  artifacts: none persistent; checkpoint files lived under pytest temporary
  paths. No baseline candidate, seed, model-order, metric, or final-test
  release policy changed. Documentation changed in README, app/core module
  notes, and ADR-0096.
- Plan changes: split P3.10c3 into adapter (c3a), completed-decision CPU
  report/checkpoint (c3b1), failed-attempt and remaining CPU modes (c3b2),
  and actual-device CUDA (c3c), while retaining the original c3 acceptance
  criteria. Checked c3a and c3b1 only after red/green, tamper, public JSON,
  full CPU, and static evidence. c3b2/c3c/c3 and later tasks remain
  unchecked; c3b2 has no external blocker. CUDA c3c awaits an actual CUDA
  host. Exact next action: add a red CPU pre/core/post guard-exception
  fixture at a `before_sleep` checkpoint, then persist an error event with
  truthful completed-pass exposure and any core proposal across explicit
  same-epoch resume. Extend preflight for repeated same-epoch attempts
  before covering wall-time, memory, fixed-width, and both guard metrics.

## 2026-09-29 — Fixed-feature failed sleep attempts and CPU mode closure

- Completed task IDs: **P3.10c3b2 and P3.10c3b**. The active checkout remains
  `codex/research-protocols-and-resume` at `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`
  with cumulative uncommitted work preserved. No reset, commit, sweep, or
  unrelated file cleanup was done.
- A red file-checkpoint fixture exposed the missing fixed-feature error
  event. Pre-guard, core, post-guard, and nonfinite paths now restore the
  head and Python/NumPy/Torch process random streams, re-raise, and save a
  typed error at the retryable `before_sleep` cursor. A returned core
  proposal stays in the error record with zero applied work. Explicit resume
  retains ordered errors, including two failures in one epoch, then resolves
  that epoch without repeating wake training. Both accuracy and
  cross-entropy guard choices are covered; strict JSON contains finite,
  typed events.
- A second red fixture showed that a partially completed guard pass was
  undercounted. The evaluator now reports completed batch sizes; nonfinite
  completed passes count all selected examples. Checkpoint preflight accepts
  only prefix sums of the selected guard batches, binds known scores to the
  failure stage, and rejects impossible exposure and eight other pending
  history tamper forms before model restore. Legacy guard-exposure counters
  still count completed two-pass decisions, keeping old report semantics.
- A deterministic wall-time fixture proves the failed attempt's 0.25 active
  seconds are charged, wake batches are not repeated, and every non-timing
  report field and resolved event matches the clean run. The public
  fixed-width route passes with and without checkpoint memory: baseline and
  circadian model hashes, capacity, validation/test metrics, final-test
  release order, and strict JSON remain aligned. ADR-0097 records the
  transaction and partial-exposure choice; README and module notes now
  describe the behavior.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_fixed_feature_checkpoint_resume.py tests/test_checkpoint_memory_resume.py tests/test_matched_head_capacity.py tests/test_sleep_event_accounting.py tests/test_sleep_telemetry_contract.py tests/test_guarded_sleep_atomicity.py  # first focused gate: 4 expectation failures after correcting nonfinite exposure, 137 passed; expectations corrected
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_fixed_feature_checkpoint_resume.py  # 54 passed, including partial-prefix persistence/tamper
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,193 passed, 24 actual-CUDA skips in 265.48 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_fixed_feature_checkpoint_resume.py::test_wall_time_resume_charges_failed_sleep_without_repeating_wake tests/test_fixed_feature_checkpoint_resume.py::test_partial_guard_batch_failure_persists_and_rejects_impossible_exposure tests/test_matched_head_capacity.py::test_fixed_width_error_history_resumes_before_final_test  # 4 passed after strengthening the wall-time assertion
  .\.venv\Scripts\python.exe -m mypy src tests scripts  # success, 179 source files
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m ruff format --check src/app/matched_head_benchmark.py src/app/fixed_feature_checkpoint.py src/app/torch_sleep_decisions.py src/core/sleep_telemetry.py tests/test_fixed_feature_checkpoint_resume.py tests/test_checkpoint_memory_resume.py tests/test_matched_head_capacity.py tests/test_sleep_telemetry_contract.py  # 8 files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Skipped tests: 24 actual-device CUDA cases in the full suite. This host
  has Torch 2.14.0+cpu; no CUDA fixed-feature telemetry acceptance is
  claimed. Experiment artifacts: none persistent; checkpoint files lived
  under pytest temporary paths. Baselines, seeds, selected metrics, and
  final-test release policy were unchanged.
- Plan changes: checked c3b2/c3b only after the CPU and static gates;
  c3c/c3/c5/c/P3.10 remain unchecked with original acceptance criteria.
  Reordered independent CPU c4 ahead of c3c while CUDA hardware is absent.
  Exact next action: inspect unmatched-vision v1/v2/v3 guard and trusted
  checkpoint routes, then write a red bounded CPU fixture for rejected
  proposal and failed guard event persistence across explicit resume. On
  CUDA hardware, add c3c restart telemetry fixtures and run
  `.\.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_cuda_fixed_feature_checkpoint.py tests/test_cuda_checkpoint_memory_resume.py`, then the full gate.

## 2026-09-29 — Unmatched-vision CPU sleep histories and failed attempts

- Completed task IDs: **P3.10c4a and P3.10c4b**. The active checkout remains
  `codex/research-protocols-and-resume` at
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`, with cumulative
  uncommitted user and development changes preserved. No reset, commit,
  baseline tuning, experiment sweep, or unrelated cleanup was done.
- The first red public fixture found no `sleep_events` in a vision report;
  another red fixture found a post-guard exception left an error-free
  `before_sleep` checkpoint. The v1/v2/v3 runner now returns a typed event
  for each completed or skipped sleep decision, retaining the core's proposed
  stable-ID changes on guard rejection and keeping applied work empty. It
  distinguishes the v1 validation guard from the disjoint v2/v3 inner
  guard, and records metric, scores, tolerance, exact two-pass exposure, and
  measured duration. Format-2 trusted active/completed files carry ordered
  histories; preflight checks clocks, nested facts, role/hash, exposure,
  resolved reasons, and counters before model restore or final-test access.
- Failed pre/core/post attempts restore the head and Python/NumPy/Torch CPU
  process random streams, append a typed error with completed guard-batch
  exposure and any returned core proposal, save a retryable `before_sleep`
  cursor, and re-raise. Explicit resume can record two errors and one
  resolution at the same epoch without repeating wake. The shared process
  RNG helper replaced duplicate fixed-feature code. Eighteen public
  protocol/stage/metric checkpoint cases compare model hashes, every
  non-timing circadian report field, baseline reports, next random draws,
  strict JSON, and sealed final-test access to uninterrupted controls.
  Four partial/nonfinite pre/post cases check completed-batch counts; an
  actual partial file rejects impossible exposure before training. Twenty-
  seven malformed pending-error histories across v1/v2/v3 reject before
  restore. An unguarded core error also resumes to the same trained state.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q tests/test_vision_checkpoint_resume.py -k unguarded_vision_core_error_is_restored_on_explicit_resume  # passed
  .\.venv\Scripts\python.exe -m pytest -q tests/test_vision_checkpoint_resume.py tests/test_guarded_sleep_atomicity.py tests/test_sleep_retry_runners.py tests/test_fixed_feature_checkpoint_resume.py tests/test_torch_runner_sleep_decisions.py  # exit 0; shared-helper regression gate
  .\.venv\Scripts\python.exe -m pytest -q tests/test_vision_checkpoint_resume.py -k failed_vision_sleep_history_tamper_rejects_before_restore  # 27 passed
  .\.venv\Scripts\python.exe -m pytest -q  # exit 0; 1,293 collected, 24 actual-CUDA skips
  .\.venv\Scripts\python.exe -m pytest -o addopts= --collect-only -q  # 1,293 collected
  .\.venv\Scripts\python.exe -m mypy src tests scripts  # success, 181 source files
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m ruff format --check src/app/resnet50_benchmark.py src/app/vision_sleep_history.py src/app/torch_sleep_decisions.py src/app/torch_sleep_transaction.py src/app/matched_head_benchmark.py src/app/vision_checkpoint.py tests/test_vision_checkpoint_resume.py tests/test_guarded_sleep_atomicity.py tests/test_sleep_retry_runners.py tests/test_resnet50_benchmark.py tests/test_cuda_vision_checkpoint_resume.py tests/test_cuda_vision_checkpoint_process.py  # 12 files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # 2.14.0+cpu False
  ```

- An initial bare `pytest` invocation used Anaconda Python 3.9.12 and failed
  during collection on `typing.Self`; all repository gates above used the
  project `.venv` Python 3.14.7. Skipped tests: 24 actual-device CUDA cases
  in that CPU-environment full suite; `.venv` has CPU-only Torch, so this
  entry does not claim P3.10c3c/c4c. A separate working `.venv-cuda` was
  found and validated in the following entry. Experiment artifacts: none
  persistent; checkpoint files lived
  under pytest temporary paths. ADR-0098/0099, README, and app module notes
  describe the CPU behavior and remaining device gate.
- Plan changes: split original c4 into completed CPU decisions (c4a), failed
  CPU transactions (c4b), and actual-device CUDA continuation (c4c), without
  weakening c4 acceptance. Checked c4a/c4b only after their tests and full
  static gate. The c3c/c3, c4c/c4, c5/c/P3.10 parents remain unchecked.
  Exact next action: add a bounded CPU P3.10c5 artifact audit that reads toy,
  continual, fixed-feature, and vision typed local JSON and checkpoint
  histories, checking field completeness, rejected proposals, and the
  distinct attempt/performed/retained counts against existing counters.
  On a CUDA host, add and run c3c/c4c guarded rejection/error restart
  fixtures and their full gate before closing those device tasks or parents.

## 2026-09-29 — Actual-device Torch sleep histories and CUDA task closure

- Completed task IDs: **P3.10c3c, P3.10c3, P3.10c4c, and P3.10c4**. The
  checkout remains `codex/research-protocols-and-resume` at
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb` with all cumulative
  uncommitted work preserved. A separate `.venv-cuda` has Torch
  2.14.0+cu130 and sees an NVIDIA GeForce RTX 3080; the default `.venv`
  being CPU-only was not a host hardware blocker. No sweep, commit, baseline
  retuning, seed selection, metric change, or unrelated cleanup occurred.
- Fixed-feature actual-device fixtures now check accepted and rejected
  checkpoint history with the inner-guard role, exact exposure, and returned
  proposal. Six pre/core/post × accuracy/cross-entropy errors disturb the
  head and Python/NumPy/Torch CPU/CUDA plus head-local split-generator
  streams, then verify the restored `before_sleep` file, explicit same-epoch
  resume, exact non-timing event/report and model state, and future random
  draws. The public route keeps final test sealed and writes a typed local
  JSON file with empty baseline histories. Two-process fixed-epoch and
  fixed-width CUDA memory reports preserve semantic event facts, trained
  hashes, metrics, and allocator segments; only measured event durations
  are excluded from cross-process state equality. Existing wall-time and
  capacity CUDA mode tests remain green.
- V1/v2/v3 unmatched-vision actual-device fixtures now check accepted and
  rejected sleep restart histories, validation versus inner-guard role/hash,
  exact two-pass exposure, retained proposals, cooldown, written local JSON,
  and exact non-timing reports and trained hashes. Nine pre/core/post CUDA
  errors preserve the restored process and head-local random streams,
  pending error facts, sealed final test, and explicit same-epoch retry with
  baseline/model/report parity. The separate-process wake restart remains
  green. ADR-0100 records the device gate and why controlled guard scores
  are correctness fixtures, not scientific comparison outcomes.
- The first full CUDA-environment suite found three stale fixture
  expectations: two cross-process memory cases compared volatile event
  durations, and one forced a rollback delta inconsistent with its recorded
  guard scores. The latter now returns consistent pre/post scores while
  exercising real CUDA scoring for completed-batch counts. Those five
  affected cases passed independently, then the combined and full gates
  passed. Commands and outcomes from the workspace root:

  ```powershell
  .\.venv-cuda\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"  # 2.14.0+cu130 True 1
  .\.venv-cuda\Scripts\python.exe -m pytest -o addopts= -q tests/test_cuda_vision_checkpoint_resume.py tests/test_cuda_vision_checkpoint_process.py  # first existing gate 8 passed; expanded gate 19 passed
  .\.venv-cuda\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # first full: 3 failed, 1,300 passed, 1 skipped; stale fixture expectations corrected
  .\.venv-cuda\Scripts\python.exe -m pytest -o addopts= -q tests/test_cuda_checkpoint_memory_resume.py::test_cuda_checkpoint_memory_keeps_two_process_allocator_segments tests/test_cuda_checkpoint_memory_resume.py::test_cuda_capacity_checkpoint_memory_keeps_fixed_width_across_processes tests/test_cuda_fixed_feature_checkpoint.py::test_cuda_file_resume_matches_learning_and_all_random_streams  # 5 passed
  .\.venv-cuda\Scripts\python.exe -m pytest -o addopts= -q tests/test_cuda_fixed_feature_checkpoint.py tests/test_cuda_checkpoint_memory_resume.py tests/test_cuda_vision_checkpoint_resume.py tests/test_cuda_vision_checkpoint_process.py  # 40 passed in 65.31 s
  .\.venv-cuda\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,309 passed, 1 skipped in 342.72 s
  .\.venv\Scripts\python.exe -m mypy src tests scripts  # success, 181 source files
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m ruff format --check tests/test_cuda_fixed_feature_checkpoint.py tests/test_cuda_checkpoint_memory_resume.py tests/test_cuda_vision_checkpoint_resume.py  # 3 files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Skipped tests: one CPU-only runtime compatibility case
  (`test_cuda_vision_checkpoint_reports_unavailable_device_before_data`)
  in the full CUDA environment; the 24 CUDA-only
  cases skipped under the default CPU environment were run through the
  focused and full CUDA gates. Experiment artifacts: none persistent;
  written JSON and checkpoint files were under pytest temporary paths.
  README, app module notes, and ADR-0100 now describe the device evidence.
- Plan changes: checked c3c/c3 and c4c/c4 only after the RTX 3080 focused
  and full gates; their original acceptance criteria remain intact. c5,
  c/P3.10, deferred P2.6a, and later phases remain unchecked. The CPU-only
  host assumption in the earlier handoff is corrected to the default
  `.venv` runtime. Exact next action: implement P3.10c5's bounded
  cross-runner audit of typed toy, continual, fixed-feature, and
  unmatched-vision local JSON and checkpoint histories; reconcile field
  completeness and attempt/performed/retained counts, then run the Phase 3
  forced split/prune/replay/no-op/rejection/resume exit gate before closing
  c5/c/P3.10.

## 2026-09-29 — Cross-runner typed sleep artifact and Phase 3 closure

- Completed task IDs: **P3.10c5, P3.10c, and P3.10**. The checkout remains
  `codex/research-protocols-and-resume` at
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; all cumulative
  uncommitted and unrelated work was preserved. This session added
  `tests/test_sleep_artifact_audit.py` and ADR-0101, then updated this log
  and `DEVELOPMENT_PLAN.md`. It changed no model, baseline, seed, metric,
  protocol ID, or final-test access rule.
- Seven bounded audit cases inspect toy, historical continual, arrived v6,
  arrived v7 candidate/trial/final selection, fixed-feature Torch, and
  unmatched-vision Torch outputs. Each strict local JSON event has exactly
  the typed schema, finite serializable values, aligned chemical counts,
  and a matching saved report/checkpoint history. Ordinary and resumed
  semantic histories agree after excluding measured durations. A rejected
  attempt retains its proposed split/replay facts but applies zero work;
  a replay-only component event performs work with no retained topology.
  Existing legacy counters remain distinct from scheduled attempts and
  retained changes. The audit identified that fixed-feature
  `guard_examples_scored` includes per-epoch stopping evaluations in
  addition to completed two-pass sleep guards; the test now reconciles both
  terms. ADR-0101 records the reason to preserve these counter scopes.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_sleep_artifact_audit.py  # final 7 passed in 4.58 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_numpy_sleep_telemetry.py tests/test_torch_sleep_telemetry.py tests/test_atomic_sleep_core.py tests/test_guarded_sleep_atomicity.py tests/test_sleep_retry_runners.py tests/test_numpy_full_snapshot.py tests/test_torch_full_snapshot.py tests/test_torch_classifier_full_snapshot.py tests/test_sleep_event_accounting.py tests/test_sleep_artifact_audit.py tests/test_toy_sleep_telemetry.py tests/test_continual_sleep_telemetry.py tests/test_continual_arrived_sleep_telemetry.py tests/test_torch_runner_sleep_decisions.py  # 194 passed, 1 skipped in 14.17 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_numpy_neuron_lineage.py tests/test_torch_neuron_lineage.py tests/test_prune_outcomes.py tests/test_prune_metadata_alignment.py tests/test_sleep_event_lineage.py tests/test_saturation_topology_boundaries.py  # 48 passed in 1.58 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,276 passed, 41 skipped in 253.21 s
  .\.venv-cuda\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,316 passed, 1 skipped in 322.27 s on RTX 3080
  .\.venv-cuda\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_sleep_artifact_audit.py  # post-edit 7 passed in 3.85 s
  .\.venv\Scripts\python.exe -m mypy  # success, 182 source files
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m ruff format --check tests/test_sleep_artifact_audit.py  # formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Skipped tests: the default CPU environment skipped 41 actual-device
  cases; the CUDA full suite skipped the CPU-only unavailable-device
  compatibility case. The 194-case focused exit gate's one skipped case
  requires a CUDA device and was exercised by the CUDA full suite.
  Experiment artifacts: no persistent new experiment output. The audit's
  JSON and trusted checkpoint files lived under pytest temporary paths.
  No sweep or scientific ranking was run.
- Plan changes: checked c5/c/P3.10 after the strict artifact, forced
  split/prune/replay-only/no-op/accept/reject/resume, snapshot alignment,
  rollback/future-continuation, full CPU/CUDA, and static gates passed.
  P4.1 is now the smallest unblocked milestone; P2.6a remains an explicit
  deferred deeper-attribution extension and later phases stay unchecked.
  Blockers: none for P4.1. Exact next action: add a deterministic P4.1
  fixture that records how many examples/bytes legacy variable-size batch
  snapshots actually retain, verifies whether saved priorities age after
  later wake updates, and measures Phase A content-ID survival after a
  Phase B shift in the opt-in bounded route; record the observed result
  before changing retention policy under P4.2.

## 2026-09-29 — P4.1 replay-retention audit

- Completed task ID: **P4.1**. This continued the same checkout and
  session after P3.10 closure. Added
  `tests/test_replay_retention_audit.py` and
  `docs/replay-retention-audit.md`; linked the audit from `README.md` and
  `docs/modules/core.md`. No production policy, model, baseline, seed,
  metric, protocol, or final-test rule changed.
- The fixed legacy two-slot A→B fixture stores one one-row A batch, then
  four- and two-row B batches. It ends with two B snapshots, six distinct
  examples, 144 input/target array bytes, and zero A rows. A separate
  three-slot fixture shows repeated A content occupies two entries for
  only two distinct IDs across three rows. Its original priority remains
  unchanged after later B wake updates change the weights. This confirms
  that `replay_memory_size` bounds batches rather than examples/bytes and
  that historical priority does not age automatically.
- The existing opt-in four-example/96-byte path retains two A and two B
  content IDs after a fixed four-row A then four-row B stream. Repeating
  one retained A row replaces its snapshot and refreshes its priority
  without changing the unique-ID set. Content-hash selection has no task
  quota; two A survivors are a property of this fixed input, not a general
  performance or fairness claim. The document states copied-array bytes
  exclude Python/object/allocator overhead and that P4.2 still must
  compare recent FIFO versus reservoir under equal budgets.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_replay_retention_audit.py  # 3 passed in 0.14 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_replay_retention_audit.py tests/test_continual_bounded_replay.py tests/test_circadian_predictive_coding.py  # 29 passed in 1.89 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,279 passed, 41 skipped in 254.66 s
  .\.venv\Scripts\python.exe -m mypy  # success, 183 source files
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m ruff format --check tests/test_replay_retention_audit.py tests/test_sleep_artifact_audit.py  # 2 files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Skipped tests: 41 actual-device cases in the default CPU environment;
  the prior P3.10 CUDA full suite ran on the RTX 3080 in this session.
  P4.1 adds only NumPy fixtures and documentation, so a second full CUDA
  suite was not run. Experiment artifacts: the fixed fixture and recorded
  counts in `docs/replay-retention-audit.md`; no external or large-sweep
  artifact. Blockers: none.
- Plan changes: checked P4.1 after the fixed retention observations,
  combined replay tests, full CPU suite, and static gates. P4.2 remains
  unchecked: the existing v4 opt-in example/byte cap is partial prior
  implementation, while recent FIFO/reservoir comparison, duplicate-ID and
  distinct replay exposure reporting, and class/task balancing policy
  remain open. P4.4 retains matched PC/backprop replay controls; deferred
  P2.6a and later phases are unchanged. Exact next action: add a
  deterministic P4.2 A→B fixture that compares recent FIFO and reservoir
  under identical example/array-byte caps and logs duplicate IDs, then
  implement those opt-in policies without changing the historical or v4
  content-hash protocol IDs.

## 2026-09-29 — P4.2a bounded core retention-policy controls

- Completed task ID: **P4.2a**. The checkout remains
  `codex/research-protocols-and-resume` at
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb` with cumulative
  uncommitted work preserved. Added `src/core/replay_retention.py`,
  `tests/test_replay_retention_policies.py`, and ADR-0102; extended the
  existing NumPy core opt-in configuration and updated `README.md`,
  `ARCHITECTURE.md`, and `docs/modules/core.md`. No dependency, baseline,
  metric, old protocol ID, or final-test access rule changed.
- The test first failed at collection because the replay policy module did
  not exist. The implemented typed policy supports `recent_fifo` and a
  declared-seed bottom-k reservoir over distinct content IDs, both under
  the existing `ReplayRetentionBudget`. The unqualified call retains the
  exact v4 smallest-content-hash rule and its snapshot field set; the
  legacy batch buffer is unchanged. Invalid policies/seeds reject before
  wake, retained duplicate IDs refresh without another slot, and restore
  rejects changed policy, seed, budget, or over-cap memory before mutating
  live state. ADR-0102 explicitly distinguishes bottom-k from classic
  Algorithm R over repeated occurrences and records why no unbounded
  seen-ID ledger was added.
- In one fixed-data A→B canary with model seed 41, policy seed 53,
  four A and four B two-feature float64 rows, and shared 4-example/96-array-
  byte caps, FIFO keeps all four latest B IDs. The seeded reservoir keeps
  two A and two B IDs and yields the same distinct-ID set when phase order
  is reversed. Independent 2-example and 48-byte cap fixtures each retain
  two rows/48 bytes. These are retention correctness observations, not
  accuracy, forgetting, or policy-selection evidence. Policy seed 53 was
  fixed for the canary; no seed search or baseline tuning was performed.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_replay_retention_policies.py  # first collection: ModuleNotFoundError for src.core.replay_retention; after implementation 10 passed; after count/byte cases 14 passed within the focused gate below
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_replay_retention_policies.py tests/test_replay_retention_audit.py tests/test_continual_bounded_replay.py tests/test_numpy_full_snapshot.py tests/test_continual_shift_benchmark.py  # final 46 passed in 2.36 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,293 passed, 41 skipped in 253.43 s
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m ruff format --check src/core/replay_retention.py tests/test_replay_retention_policies.py  # 2 files formatted
  .\.venv\Scripts\python.exe -m mypy  # success, 185 source files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Skipped tests: 41 actual-device cases in the default CPU environment.
  This increment changes NumPy core retention only; no new CUDA behavior
  is claimed or CUDA suite run. Experiment artifacts: no persistent
  experiment output, only the fixed test fixture and ADR. The repository's
  broader dirty working tree was preserved; no commit, remote write, or
  large sweep was made.
- Plan changes: split P4.2 into core policy correctness (a) and versioned
  arrived-runner comparison/exposure (b) before implementation because v4
  already owned the caps and its identity must not change. Checked a only;
  b/parent and later Phase 4 tasks remain unchecked. Inspection found that
  v6 requires an exact config type/protocol ID, its model factory always
  configures v4 hash retention, format-6 preflight reconstructs that model,
  and v7 binds v6 candidates and a separate selection digest. Therefore a
  policy field inside v6/v7 would change saved identities; b must use a
  versioned route. No blocker for b. Exact next action: add a failing
  ordinary arrived-run fixture for a separate policy-bearing config and
  predeclared FIFO/reservoir manifest on identical A→B roles/caps. Require
  public retained/duplicate/exposure facts, unchanged baseline state, and
  global final-test sealing before implementing the ordinary route; then
  introduce a distinct checkpoint format and restart preflight.

## 2026-09-29 — P4.2b1 ordinary matched replay-policy comparison

- Completed task ID: **P4.2b1**. The checkout remains
  `codex/research-protocols-and-resume` at
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; its cumulative dirty
  working tree and unrelated changes were preserved. The first new test
  failed at collection because `src.app.continual_replay_policy_comparison`
  did not exist. The new ordinary `continual_replay_policy_comparison_v8`
  module binds one exact v6 arrived-role training config, ordered seeds,
  FIFO and seeded bottom-k policies, and a manifest digest. A small optional
  policy parameter reaches the existing A-only model factory; v6/v7 default
  calls and checkpoint identities remain unchanged.
- An opt-in core snapshot ledger now reports all observed distinct content
  IDs, duplicate IDs/occurrences, distinct successfully applied replay IDs,
  and replay updates at A and B. Only explicitly selected new policies add
  those fields; v4 hash and historical batch snapshots retain their field
  sets. Internal restore validation rejects inconsistent exposed IDs before
  mutation. The ledger may grow with observed IDs and is reporting memory
  outside the copied-array retention byte cap (ADR-0103). The runner exposes
  all policy/seed scores, role IDs/hashes and access events, guard and sleep
  histories, A/B retained IDs/bytes, replay exposure, and four local baseline
  trained-state hashes. It scores final tests only after both policies and
  every seed finish training. No winner is selected.
- The fixed local run used seeds `(17, 19)`, policy seed 53, 40 source rows
  per phase, two wake epochs per phase, 4-example/96-array-byte replay caps,
  and one replay step per scheduled sleep. Every policy/seed performed four
  replay updates and retained four rows at both boundaries. FIFO and
  reservoir kept different B IDs; FIFO exposed two distinct IDs per seed,
  reservoir one. All 22 repeated wake content IDs per seed and the 22
  duplicate occurrences are recorded. Both policies had identical balanced
  scores on each seed: backprop 0.95/0.20, PC 0.75/0.50, and circadian
  0.00/0.40 for seeds 17/19. This null result is kept without tuning seeds,
  baselines, or metrics. It is a tiny synthetic correctness/comparison
  artifact, not a general policy ranking or replay-capable baseline claim.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_replay_policy_comparison.py  # first collection: ImportError for missing module; after implementation 4 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_replay_policy_comparison.py tests/test_replay_retention_policies.py tests/test_continual_arrived_runner.py tests/test_continual_arrived_checkpoint.py tests/test_continual_arrived_selection.py tests/test_continual_arrived_selection_checkpoint.py  # final 146 passed in 28.52 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,298 passed, 41 skipped in 279.14 s
  .\.venv\Scripts\python.exe -m scripts.run_continual_replay_policy_smoke --result data/continual_replay_policy_v8_smoke.json  # four fixed policy/seed rows written once
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # success, 188 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/core/replay_retention.py src/core/circadian_predictive_coding.py src/app/continual_shift_benchmark.py src/app/continual_arrived_benchmark.py src/app/continual_replay_policy_comparison.py tests/test_replay_retention_policies.py tests/test_continual_replay_policy_comparison.py scripts/run_continual_replay_policy_smoke.py  # 8 files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- The four new arrived tests prove fixed role hashes and baseline-state
  hashes across policies, actual replay-work parity, all-policy/all-seed A/B
  final-source sealing, and final-label perturbation leaving model, replay,
  guard, and development-role facts unchanged. A core test verifies
  duplicate/exposure counts, exact snapshot continuation, and rejection of
  a forged exposure ID. Existing v6/v7 runner/checkpoint tests pass in the
  focused and full gates. Skipped tests: 41 actual-device CUDA cases in the
  default CPU environment; this increment changes NumPy policy reporting,
  so no new Torch/CUDA claim is made or CUDA sweep launched.
- Experiment artifact: ignored local
  `data/continual_replay_policy_v8_smoke.json`, SHA-256
  `d28dbb2830ba2a939c1b7c13a2e2162f05e604fad52e62b88f09e37f9dbb475e`.
  The JSON parses with all four policy/seed rows, finite values, and role
  ledgers. ADR-0103, README, architecture, and app/core module docs describe
  its scope. No remote write, commit, or large sweep was made.
- Plan changes: split P4.2b into completed ordinary artifact/isolation b1
  and still-open trusted checkpoint/resume b2 because v6/v7 exact config and
  checkpoint identities cannot carry a new policy. P4.2b and P4.2 remain
  unchecked; replay-capable PC/backprop controls remain P4.4. There is no
  blocker to b2. **Exact next action:** add a red completed-policy/seed v8
  checkpoint fixture that holds unscored models and exposure records until
  every policy/seed finishes; then implement distinct format-9 identity and
  arrived-role/retained-and-exposed-ID preflight before active A/B resume.

## 2026-09-29 — P4.2b2a completed policy-trial checkpoint continuation

- Completed task ID: **P4.2b2a**. The authoritative checkout remains
  `codex/research-protocols-and-resume` at
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`, with cumulative dirty
  work and unrelated user changes preserved. The first new checkpoint test
  failed at collection because the format-9 store did not exist. Added
  `src/app/continual_replay_policy_checkpoint.py` and
  `src/app/continual_replay_policy_resume.py`, a separate trusted local
  format-9 store, and a policy-aware optional argument to the existing
  completed-state model factory and validator. V6/v7 default config,
  protocol, checkpoint, and selection identities remain unchanged.
- The new envelope saves only completed unscored policy/seed records in the
  exact predeclared policy-major order, with the full v8 manifest digest and
  a separate cumulative exposure digest. Resume rejects a changed manifest
  before source access. For every saved trial it rebuilds A/B development
  roles and checks the arrived baseline/model progress, bounded retained
  rows, guard/role events and sleep history. It then recomputes exact
  observed and duplicate content-ID counts from arrived train rows and
  declared wake epochs; exposed IDs must be observed and applied replay
  updates must match typed sleep events. A forged duplicate count remains
  rejected even when its exposure digest is recomputed. Remaining trials
  train once; no final-test field is released until all four records exist.
- Ten new checkpoint cases cover interruption after the first FIFO trial
  and after the first reservoir trial, no retraining of completed trials,
  ordinary/resumed/terminal report equality, all-policy A/B final-source
  sealing, changed policy seed/cap/seed order before source access, changed
  A training data before another update, forged exposure ID and duplicate
  count, nonprefix cursor, and separate v6 versus format-9 file magic.
  Existing v6/v7 runner and checkpoint regressions remain green. This is
  **completed-trial** continuation only: an interrupted A or B trial still
  restarts; P4.2b2b retains active transaction and exact A/B continuation.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_replay_policy_checkpoint.py  # first collection: ImportError for absent store; final 10 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_replay_policy_checkpoint.py tests/test_continual_replay_policy_comparison.py tests/test_replay_retention_policies.py tests/test_continual_arrived_runner.py tests/test_continual_arrived_checkpoint.py tests/test_continual_arrived_selection.py tests/test_continual_arrived_selection_checkpoint.py  # final 156 passed in 28.87 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,304 passed, 41 skipped in 254.00 s; later test-only parametrization/cursor additions passed the final focused gate
  .\.venv\Scripts\python.exe -m scripts.run_continual_replay_policy_smoke --checkpoint data/continual_replay_policy_v9_terminal.ckpt --result data/continual_replay_policy_v9_fresh.json  # four fixed rows and terminal checkpoint written
  .\.venv\Scripts\python.exe -m scripts.run_continual_replay_policy_smoke --checkpoint data/continual_replay_policy_v9_terminal.ckpt --resume --result data/continual_replay_policy_v9_resumed.json  # terminal resume; byte-identical result JSON
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # success, 191 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/app/continual_replay_policy_checkpoint.py src/app/continual_replay_policy_resume.py src/app/continual_replay_policy_comparison.py src/app/continual_arrived_benchmark.py src/app/continual_shift_benchmark.py src/infra/circadian_checkpoint_files.py tests/test_continual_replay_policy_checkpoint.py scripts/run_continual_replay_policy_smoke.py  # 8 files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Experiment artifacts: ignored local
  `data/continual_replay_policy_v9_terminal.ckpt` (SHA-256
  `69527ff8effe0641c3218dfc512a1fa4bdca96f87c6b57aa48c98c22bf1db4dc`),
  `data/continual_replay_policy_v9_fresh.json` and
  `data/continual_replay_policy_v9_resumed.json` (both SHA-256
  `25966ff0804c7c636b19555b192baa1f5d1aface0497cb5605761ad0691e78fb`).
  The manifest, all four rows, different FIFO/reservoir retained and exposed
  IDs, and identical balanced scores remain present; no metric, seed,
  baseline, or winner was changed to force a positive result. ADR-0104,
  README, architecture, and app/infra/evaluation module docs record the
  completed-trial boundary. No new dependency, remote write, commit, or
  large sweep was made.
- Skipped tests: 41 actual-device CUDA cases in the CPU environment. This
  NumPy runner/checkpoint increment makes no new Torch/CUDA claim; the
  relevant old CUDA gates were previously verified. Plan changes: split
  P4.2b2 into completed-trial format/preflight a and active A/B transaction
  b because reusing v6's active format would change its saved interpretation.
  Checked a only; b2b/b2/b/P4.2 and later Phase 4 remain unchecked. No
  blocker for b2b. **Exact next action:** add a red format-9 Phase A wake
  and before-sleep interruption fixture under FIFO, requiring exact model,
  ledger, and event continuation, Phase B non-arrival, and no final access;
  then extend the active cursor and policy-aware restore before covering B
  and reservoir interruption.

## 2026-09-29 — P4.2b2b active policy transactions and P4.2 closure

- Completed task IDs: **P4.2b2b, P4.2b2, P4.2b, P4.2**. The authoritative
  checkout remains `codex/research-protocols-and-resume` at
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; cumulative dirty work
  and unrelated user changes were preserved. Four initial red Phase A tests
  failed because the v8 format-9 envelope had no active cursor. Added a
  separately typed policy-bearing active transaction, nested after the
  completed policy/seed prefix, and adapted the arrived transaction engine's
  model factory and save callback without changing v6/v7 checkpoint identity
  or the v4 hash default (ADR-0105).
- Header preflight checks manifest, policy/seed order, exact phase/model
  cursor, absent final roles, and Phase A frozen-state presence before
  opening Phase B. The policy-aware arrived validator reconstructs only
  arrived development roles, checks active model/retained-ID progress,
  ordered role/guard events and typed sleep history, and rejects future-role
  replay. Additional v8 preflight recomputes observed/duplicate IDs from
  the exact wake cursor and matches applied replay updates to sleep events
  before another training update. Completed trials still rehydrate from the
  durable prefix without retraining; final fields remain sealed until all
  four fixed trials are complete.
- New active tests interrupt FIFO and reservoir in Phase A and B at wake,
  before-sleep, after-sleep, arrival, and policy boundaries in both model
  orders. They assert exact remaining wake counts, ordinary/resumed report
  equality, full circadian snapshots including RNG state, and all-policy
  final release only after the terminal checkpoint. They reject changed
  policy seed/caps/seed order before source access, a forged Phase B cursor
  before B arrival, future B replay in A, wrong saved policy, forged
  duplicate exposure, reordered sleep events, and guard role with recomputed
  digests before another update. Changing a final label leaves the training
  state, exposure, guard decisions, and development-role hashes invariant.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_replay_policy_checkpoint.py  # initial new Phase A tests: 4 failed for missing active field; final active file 35 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_replay_policy_checkpoint.py tests/test_continual_replay_policy_comparison.py tests/test_replay_retention_policies.py tests/test_continual_arrived_runner.py tests/test_continual_arrived_checkpoint.py tests/test_continual_arrived_selection.py tests/test_continual_arrived_selection_checkpoint.py tests/test_continual_arrived_sleep_telemetry.py  # 226 passed in 45.51 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # first 1,330 passed/41 skipped; an interim rerun was interrupted at 31% to include two later tests; final 1,333 passed/41 skipped in 298.44 s
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # success, 191 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/app/continual_replay_policy_checkpoint.py src/app/continual_replay_policy_resume.py src/app/continual_replay_policy_comparison.py src/app/continual_arrived_transactions.py tests/test_continual_replay_policy_checkpoint.py  # 5 files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Local bounded experiment artifacts: ignored
  `data/continual_replay_policy_v9_active_pre_resume.ckpt` is a reservoir
  Phase B before-sleep cursor after two completed trials (SHA-256
  `1c3609eb3560bda8e6dc5541fc19748b4877ca230afd7f8907649dc48fe11255`).
  Resuming an equivalent active file through
  `python -m scripts.run_continual_replay_policy_smoke --checkpoint
  data/continual_replay_policy_v9_active_reservoir.ckpt --resume --result
  data/continual_replay_policy_v9_active_resumed.json` produced the terminal
  checkpoint (SHA-256
  `740bc64ee15535511f41dbcc16e786867e1a97a7fc05f7abd2a74823c6705e54`)
  and all four policy/seed JSON rows (SHA-256
  `9fae82d48701e0540b9ac524bd516254bf5bad8c980e11a2b476bd892a8f94c5`).
  Fresh and active-resumed JSON differ only in measured sleep durations;
  after excluding those duration fields, both SHA-256 digests are
  `9d35cffbb37ae381f2225609e77430c0958395f6383a86b20a35c3dca977be50`.
  The fixed comparison still shows identical balanced policy scores on
  seeds 17/19 despite different retained/exposed IDs. No seed, metric,
  baseline, or winner was changed to make circadian win.
- Skipped tests: 41 actual-device CUDA cases in the default CPU environment.
  This increment changes the NumPy policy comparison and checkpoint path;
  it makes no new Torch/CUDA claim or large sweep. Plan changes: close
  P4.2b2b/b2/b/P4.2 after their acceptance and full quality gates, add
  **P4.4a** as a shared replay-schedule correctness gate, and prioritize it
  ahead of P4.3 side-effect policy changes as requested. Existing v8 null
  outcomes and old protocol/checkpoint formats remain untouched. P4.3,
  P4.4/P4.4a, and later phases remain unchecked. No blocker. **Exact next
  action:** write a red fixed A→B shared replay-schedule test under a new
  protocol that requires identical selected content IDs/order, cap, and
  per-sleep budget for circadian, PC, and backprop with no guard/outer/final
  role exposure; then implement the smallest shared schedule and separate
  PC/circadian inference and optimizer-work accounting.

## 2026-09-29 — P4.4a shared replay schedule before baseline training

- Completed task ID: **P4.4a**. The authoritative checkout is still
  `codex/research-protocols-and-resume` at
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; cumulative dirty work
  and unrelated user changes were preserved. The first schedule fixture
  failed at collection because `continual_matched_replay_schedule` did not
  exist. Added `src/core/shared_replay_schedule.py` and
  `src/app/continual_matched_replay_schedule.py` under a separate
  `continual_matched_replay_schedule_v9` ID, with no change to v8 or earlier
  runner/checkpoint formats (ADR-0106).
- The core buffer copies labeled float64 train rows, deduplicates content
  IDs, enforces the same example/copy-byte caps and FIFO or seeded bottom-k
  eviction as the existing circadian route, and selects newest retained
  rows without model predictions. An initial integration assertion exposed
  an important distinction: the historical public retention snapshot sorts
  IDs, while unprioritized replay uses buffer order. The new boundary
  records both explicitly. Each method can request private copies of the
  identical selected rows.
- The app session validates the manifest before source access, opens only
  arrived A training first, and opens B only after the declared A schedule.
  It rejects changed policy, seed list, caps, replay budget, non-train role,
  changed train values, or an appended train row before advancing the
  buffer. The resulting boundary assigns the same ordered IDs to
  circadian, PC, and backprop and reports **planned** examples, optimizer
  updates, and separate PC/circadian inference iterations. It does not
  perform model updates or produce scores. Eighteen new app/core cases cover
  two policies, both model orders, source-role sentinels, deterministic
  A→B selection, parity with the existing unprioritized circadian buffer,
  duplicate refresh, detached copies, and independent count/byte caps.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_matched_replay_schedule.py  # first collection: ImportError for absent app module
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_matched_replay_schedule.py tests/test_shared_replay_schedule.py  # final new app/core tests 18 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_matched_replay_schedule.py tests/test_shared_replay_schedule.py tests/test_replay_retention_policies.py tests/test_continual_replay_policy_comparison.py tests/test_continual_replay_policy_checkpoint.py  # 72 passed in 10.81 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # two interim runs interrupted after code/artifact refinements; final 1,351 passed, 41 skipped in 264.07 s
  .\.venv\Scripts\python.exe -m scripts.run_continual_matched_replay_schedule_smoke --result data/continual_matched_replay_schedule_v9_resolved.json  # fixed two-seed/two-policy artifact
  .\.venv\Scripts\python.exe -m scripts.run_continual_matched_replay_schedule_smoke --result data/continual_matched_replay_schedule_v9_resolved_repeat.json  # byte-identical repeat
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # first caught two test-sentinel type errors; corrected; final success, 196 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/core/shared_replay_schedule.py src/app/continual_matched_replay_schedule.py tests/test_shared_replay_schedule.py tests/test_continual_matched_replay_schedule.py scripts/run_continual_matched_replay_schedule_smoke.py  # 5 files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Experiment artifacts: ignored local
  `data/continual_matched_replay_schedule_v9_resolved.json` and
  `data/continual_matched_replay_schedule_v9_resolved_repeat.json`, both
  SHA-256 `2e8094081ff76a8d08a4397db64feb4472776c7c80859a4406184ab999c02f79`.
  Each has four policy/seed rows, 16 A/B periodic boundaries, resolved
  manifests with 4-example/96-byte caps and the unprioritized sampler,
  identical selected IDs across all three planned method rows, and no
  scores. This is a schedule correctness artifact, not evidence that PC or
  backprop have performed replay or that any model wins.
- Skipped tests: 41 actual-device CUDA cases in the CPU environment; this
  NumPy-only schedule increment makes no new Torch/CUDA claim. No large
  sweep, new dependency, remote write, seed selection, metric change, or
  baseline tuning was made. Plan changes: check P4.4a only, add P4.4b for
  actual replay application with truthful guard-failure accounting, and
  keep P4.4/P4.3 and later phases unchecked. No blocker. **Exact next
  action:** add a red train-only P4.4b fixture for one fixed A→B seed and
  sleep boundary that requires circadian, PC, and backprop to apply the
  same selected IDs and update count, records actual PC/circadian inference
  work, seals guard/outer/final roles, and prevents unmatched baseline
  replay when circadian guarded sleep rejects or fails; then implement the
  smallest versioned runner path without altering v8 outcomes.

## 2026-09-29 — P4.4b actual matched replay application

- Completed task ID: **P4.4b**. The authoritative checkout remains
  `codex/research-protocols-and-resume` at
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; cumulative dirty work
  and unrelated user changes were preserved. The first red fixture reached
  the system Anaconda Python (which lacks `typing.Self`); the repository
  `.venv` then showed the intended collection failure for the missing
  `continual_matched_replay_runner` module. Added that module under the
  separate `continual_matched_replay_training_v9` result protocol, with a
  read-only circadian retained-order/selected-ID preview (ADR-0107).
- The runner trains both arrived phases in the declared model order. Before
  every periodic sleep it checks manifest identity through the schedule,
  then compares circadian retained IDs/bytes, retained order, selected IDs,
  and detached row contents. The existing guarded circadian sleep is the
  commit point: accepted, fully applied replay feeds private row copies to
  PC and backprop; rollback/skips feed neither; core errors raise before
  baseline replay. Event counts and clocks prove replay did not refill the
  buffer or advance wake progress. Applied examples, optimizer calls, and
  PC/circadian fixed-loop inference iterations are recorded separately.
  No final test, outer-selection metric, or model ranking is produced.
- Fourteen new tests observe actual replay calls and copied arrays for
  seeds 17/19, FIFO/seeded bottom-k, and both model orders. They also
  verify A→B source arrival, sealed final properties, outer-role release
  without training use, forged selected IDs and retained bytes rejected
  before sleep, guard rollback/core failure without baseline replay, and
  two-epoch intervals with no off-boundary replay. Existing policy,
  checkpoint, and guarded-sleep focused regressions pass. The bounded
  train-only script uses the already frozen manifest, with its default
  guard tolerance and no seed or metric selection.
- Commands and outcomes from the workspace root:

  ```powershell
  python -m pytest -q tests/test_continual_matched_replay_runner.py  # system Anaconda 3.9: collection fails at typing.Self; corrected interpreter below
  .\.venv\Scripts\python.exe -m pytest -q tests/test_continual_matched_replay_runner.py  # red: missing runner module; final 14 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_matched_replay_runner.py tests/test_continual_matched_replay_schedule.py tests/test_shared_replay_schedule.py tests/test_replay_retention_policies.py tests/test_continual_replay_policy_comparison.py tests/test_continual_replay_policy_checkpoint.py tests/test_continual_arrived_runner.py tests/test_guarded_sleep_atomicity.py  # 111 passed in 12.63 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,365 passed, 41 skipped in 262.97 s
  .\.venv\Scripts\python.exe -m scripts.run_continual_matched_replay_training_smoke --result data/continual_matched_replay_training_v9_resolved.json  # four unscored rows
  .\.venv\Scripts\python.exe -m scripts.run_continual_matched_replay_training_smoke --result data/continual_matched_replay_training_v9_resolved_repeat.json  # byte-identical repeat
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # success, 199 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/app/continual_matched_replay_runner.py src/core/circadian_predictive_coding.py tests/test_continual_matched_replay_runner.py scripts/run_continual_matched_replay_training_smoke.py  # four files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Experiment artifacts: ignored local
  `data/continual_matched_replay_training_v9_resolved.json` and
  `data/continual_matched_replay_training_v9_resolved_repeat.json`; both
  file SHA-256 values are
  `0e39f07428f5e9ea6e494abb4a2f4867ea77a63f9f4b87413fc5f47a83ece833`.
  The writer fixes LF newlines so its printed digest matches saved bytes.
  Four policy/seed rows contain 16 A/B periodic boundaries. Their manifest
  digests and selected/retained-order IDs exactly match the earlier
  schedule artifact. All four fixed trials accepted four sleeps and applied
  eight replay rows/optimizer calls per method. Per trial, PC used 16 and
  circadian 24 replay inference iterations; backprop used zero. The trace
  has no final scores and does not establish a winner.
- Skipped tests: 41 actual-device CUDA cases in this CPU environment. This
  increment changes only NumPy matched replay; it makes no new CUDA claim.
  No large sweep, new dependency, remote write, baseline tuning, seed
  selection, or metric change was made. Plan changes: check P4.4b only and
  add **P4.4c** for a globally sealed matched outcome audit. P4.4/P4.3 and
  later phases remain unchecked; v8 and earlier identities were not
  changed. No blocker. **Exact next action:** add a red all-trials v9
  fixture that trains both fixed policies/seeds before any final source or
  label access, rejects tampered applied work or early final release, and
  scores the three methods on identical declared A/B final roles; then
  implement the smallest separate outcome runner and repeat its versioned
  artifact before considering P4.4 closure.

## 2026-09-29 — P4.4c globally sealed matched outcomes and P4.4 closure

- Completed task IDs: **P4.4c and P4.4**. The authoritative checkout is
  still `codex/research-protocols-and-resume` at
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; cumulative dirty work
  and unrelated user changes were preserved. The red all-trials fixture
  first failed at collection because `continual_matched_replay_outcomes`
  did not exist. Added that app module and a separate local outcome script
  under `continual_matched_replay_outcomes_v9` (ADR-0108).
- The fixed manifest binds FIFO and seeded bottom-k, seeds 17/19, one
  arrived-role configuration, `newest_retained_v1`, and the existing replay
  and PC inference budgets before source access. All four P4.4b trials
  train before any final source/label access. A fresh train-only schedule
  validates each unscored trial's manifest, A→B role identities, retained
  and selected IDs/order, sleep guard/event work, PC/backprop/circadian
  applied rows and updates, distinct inference iterations, circadian
  exposure IDs, and wake/replay clocks. Only after all pass are every A/B
  final role released and same-seed final IDs/content hashes compared
  across policies. The existing continual-shift scorer then evaluates all
  methods and retains every policy/seed score and aggregate. No winner or
  hyperparameter is selected from those scores.
- Eleven post-edit outcome tests cover both model orders, all-trial final
  seals, invalid manifest before source access, changed retained/selected
  IDs, forged applied work and circadian exposure, changed development
  role, final-role mismatch before any score, final-label perturbation
  without training/replay changes, and byte-repeat JSON. The JSON writer
  omits only volatile sleep durations from the deterministic artifact;
  it keeps typed sleep facts, scores, role hashes, retention, exposure,
  and applied work. A first script attempt exposed `asdict` retaining a
  tuple of sleep events; corrected before any artifact was written.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_matched_replay_outcomes.py  # red collection: missing outcome module; final post-edit 11 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_matched_replay_outcomes.py tests/test_continual_matched_replay_runner.py tests/test_continual_matched_replay_schedule.py tests/test_shared_replay_schedule.py tests/test_continual_arrived_runner.py tests/test_continual_replay_policy_comparison.py tests/test_continual_replay_policy_checkpoint.py tests/test_guarded_sleep_atomicity.py  # 105 passed in 13.75 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,374 passed, 41 skipped in 266.45 s; two added outcome tamper cases passed afterward in the 11-case focused run
  .\.venv\Scripts\python.exe -m scripts.run_continual_matched_replay_outcomes --result data/continual_matched_replay_outcomes_v9_resolved.json  # four scored rows
  .\.venv\Scripts\python.exe -m scripts.run_continual_matched_replay_outcomes --result data/continual_matched_replay_outcomes_v9_resolved_repeat.json  # byte-identical repeat
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # success, 202 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/app/continual_matched_replay_outcomes.py scripts/run_continual_matched_replay_outcomes.py tests/test_continual_matched_replay_outcomes.py  # three files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Experiment artifacts: ignored local
  `data/continual_matched_replay_outcomes_v9_resolved.json` and
  `data/continual_matched_replay_outcomes_v9_resolved_repeat.json`, both
  file SHA-256
  `2eb5b0937c992116ee18350dd2a858fb16522131ebc00018fbe5fecab149fe86`.
  The four scored rows match P4.4b's policy-specific schedule digests and
  all 16 selected/retained-order boundaries. A/B final role hashes match
  across policies for each seed. Every method applied eight replay rows
  and optimizer calls per trial, with 0 backprop, 16 PC, and 24 circadian
  latent-inference iterations. Per-policy balanced-score means were:
  FIFO backprop 0.65, PC 0.625, circadian 0.20; seeded bottom-k backprop
  0.70, PC 0.625, circadian 0.20. All individual scores remain in the
  artifact. This negative circadian result was retained without changing
  seeds, baseline access, metric, guard tolerance, or sampler. Two seeds
  are too few for a general ranking.
- Skipped tests: 41 actual-device CUDA cases in the CPU environment. This
  is a NumPy matched-control result and makes no new CUDA claim. No large
  sweep, new dependency, remote write, or retuning. Plan changes: check
  P4.4c and its P4.4 parent after the artifact and quality gates; split
  open P4.3 into a read-only side-effect audit (a) and explicit opt-in
  policy/ablation (b). Core inspection found replay currently gates neuron
  age, wake clocks, and history but updates chemistry, importance, traffic,
  reward-error baseline, and cooldowns. P4.3 and later phases remain
  unchecked; P2.6a remains deferred. No blocker. **Exact next action:**
  write a red deterministic core fixture around one wake step and
  `_run_replay_consolidation` that separates replay-only changes from
  surrounding sleep effects, records chemistry/importance/traffic/reward
  baseline/age/cooldowns/prune TTL/clocks/history/retention, and confirms
  no recursive refill or non-train example access before choosing the
  P4.3b policy.

## 2026-09-29 — P4.3a isolated NumPy replay side-effect audit

- Completed task ID: **P4.3a**. The authoritative branch and HEAD remain
  `codex/research-protocols-and-resume` and
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; cumulative dirty work
  and unrelated user changes were preserved. Added only a deterministic
  core fixture and `docs/replay-side-effect-audit.md`, with no model or
  protocol behavior change. The first test collection exposed a wrong
  `SleepEpochProgress` import, then fixture construction exposed the
  required energy-window and minimum-width invariants; corrected the
  fixture before interpreting any result.
- Two wake updates on the same two labeled train rows establish replay
  memory and scheduling history. The fixture enables dual chemistry,
  reward modulation, cooldowns, and a valid pending prune, then measures
  direct `_run_replay_consolidation()` and a replay-only component sleep
  from the same pre-sleep state. Direct replay changes combined/fast/slow
  chemistry, importance EMA, traffic sum/steps, reward-error baseline,
  split/prune cooldowns, and pending prune TTL. It leaves neuron age,
  wake batches/examples/since-sleep, energy history, and retained IDs/order
  unchanged while incrementing the replay clock and exposed train ID.
  The surrounding successful sleep independently resets since-sleep and
  increments sleep events. A deterministic threshold placed between
  pre/post chemical variances flips adaptive readiness with no wake/history
  change. That threshold is a causal unit probe, not a benchmark setting.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_replay_side_effect_audit.py  # first collection wrong import; then invalid fixture window/min width; final 3 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_replay_side_effect_audit.py tests/test_replay_retention_policies.py tests/test_replay_retention_audit.py tests/test_guarded_sleep_atomicity.py tests/test_continual_matched_replay_outcomes.py tests/test_continual_matched_replay_runner.py  # 63 passed in 2.74 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,379 passed, 41 skipped in 273.63 s
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # success, 203 source files
  .\.venv\Scripts\python.exe -m ruff format --check tests/test_replay_side_effect_audit.py  # formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Experiment artifacts: no new scored artifact or sweep. The P4.4 fixed
  outcome files and SHA-256 remain unchanged. The new audit document is
  the state matrix and proposed policy boundary; its opt-in behavior has
  not been implemented. Skipped tests: 41 actual-device CUDA cases in the
  CPU environment. This NumPy-only audit makes no CUDA claim. Plan changes:
  check P4.3a only; keep P4.3b and P4.3 parent unchecked. No new
  dependency, remote write, result-driven seed/metric change, or blocker.
  **Exact next action:** write a red core fixture for an explicit opt-in
  wake-only adaptive-state policy in which replay weight updates use the
  pre-row adaptive state while chemistry, importance/traffic, reward
  baseline, cooldown/prune TTL, age, wake clocks/history, and retained
  rows stay fixed; keep the current behavior as default, then test
  accepted/rejected/error sleep and a bounded equal-work policy ablation.

## 2026-09-29 — P4.3b opt-in replay side effects and matched ablation

- Completed task IDs: **P4.3b and P4.3 parent**. The authoritative branch
  and HEAD remain `codex/research-protocols-and-resume` and
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`. The pre-existing
  cumulative dirty checkout and unrelated user changes were preserved.
  Four new core tests first failed because the requested policy API did
  not exist. The implementation adds a pretraining-only
  `wake_only_adaptive_v1` model policy without changing `CircadianConfig`
  or default model snapshot fields. Only the opt-in model stores the
  policy name; restore rejects a mismatched policy. Historical v9 training
  and outcome identities retain their old values.
- Opt-in replay updates weights and applied exposure using the plasticity
  and supervised-error baseline present before each selected train row.
  It leaves combined/fast/slow chemistry, importance, traffic, last wake
  reward scale and baseline, split/prune cooldowns, pending-prune TTL,
  neuron age, wake clocks and diagnostic history, and retained example
  order untouched. Direct replay increments only its replay counter and
  exposure; accepted sleep still advances the sleep wrapper clock. Core
  tests cover pre-row plasticity/reward gating, default field identity,
  exact restore continuation, an accepted event, and rollback after a
  replay exception. App tests cover guard rejection and a core failure
  after sleep without baseline replay commit.
- The separate v10 ablation binds the existing v9 two-seed (17/19),
  FIFO/seeded bottom-k (seed 53), four-example/96-byte, newest-retained
  schedule and two replay updates per sleep. It trains all eight trials
  before any final release, preflights arrived roles, retained/selected
  IDs, actual work, exposure and wake/replay clocks, compares historical
  and opt-in boundary work plus PC/backprop model state, then compares all
  final A/B role IDs and hashes before scoring. A forged selected ID
  rejects before final release. All 32 boundaries accept and each trial
  applies eight rows/optimizer calls per method; PC uses two and circadian
  three inference iterations per replay row. The v10 historical rows
  exactly equal the prior v9 JSON rows. Both side-effect policies yield
  circadian mean balanced score 0.20, PC 0.625, and backprop 0.65/0.70
  for FIFO/reservoir. Seed-specific circadian scores remain 0.0 and 0.4.
  This is a null policy result with the earlier negative baseline finding;
  no seed, metric, guard tolerance, or baseline was changed to favor it.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_replay_side_effect_policy.py  # initial red: 4 failed on absent policy API; final 6 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_replay_side_effect_policy.py tests/test_replay_side_effect_ablation.py tests/test_replay_side_effect_audit.py tests/test_continual_matched_replay_outcomes.py tests/test_continual_matched_replay_runner.py tests/test_numpy_full_snapshot.py tests/test_continual_replay_policy_checkpoint.py  # 77 passed in 13.35 s
  .\.venv\Scripts\python.exe -m scripts.run_continual_replay_side_effect_ablation --result data/continual_replay_side_effect_ablation_v10_resolved.json  # eight trials; SHA-256 1e946e6a14d83fa77e696cebf103e34b765b8b323e5b22276b5c35b74a5c2f2d
  .\.venv\Scripts\python.exe -m scripts.run_continual_replay_side_effect_ablation --result data/continual_replay_side_effect_ablation_v10_repeat.json  # same SHA-256 and bytes
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,390 passed, 41 skipped in 270.12 s
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # success, 207 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/core/circadian_predictive_coding.py src/app/continual_matched_replay_runner.py src/app/continual_matched_replay_outcomes.py src/app/continual_replay_side_effect_ablation.py scripts/run_continual_replay_side_effect_ablation.py tests/test_replay_side_effect_policy.py tests/test_replay_side_effect_ablation.py  # 7 files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Experiment artifacts: ignored local
  `data/continual_replay_side_effect_ablation_v10_resolved.json` and
  `data/continual_replay_side_effect_ablation_v10_repeat.json`, both
  SHA-256 `1e946e6a14d83fa77e696cebf103e34b765b8b323e5b22276b5c35b74a5c2f2d`;
  historical v9 artifact remains SHA-256
  `2eb5b0937c992116ee18350dd2a858fb16522131ebc00018fbe5fecab149fe86`.
  The full suite skipped 41 actual-device CUDA cases in this CPU
  environment. The new policy/ablation is NumPy-only and makes no CUDA
  claim. Plan changes: check P4.3b and P4.3 after acceptance evidence,
  retain P4.4's negative control and all later work, move the active
  milestone to P4.5. ADR-0109, audit, README, module, architecture,
  evaluation-protocol, and changelog docs record the decision. No new
  dependency, large sweep, remote write, or blocker. **Exact next action:**
  inspect `src/core/neuron_adaptation.py` and NumPy/Torch structural
  score, candidate, budget, and tensor-mutation paths; write one
  deterministic test at a real proposal/budget seam before extracting
  shared logic, keeping default decisions and protocol identities fixed.

## 2026-09-29 — P4.5 structural policy seam

- Completed task ID: **P4.5**. The branch and HEAD remain
  `codex/research-protocols-and-resume` and
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; all pre-existing dirty
  work and unrelated user changes were preserved. Inspection found that
  NumPy already accepts `NeuronAdaptationPolicy.propose(LayerTraffic)` and
  typed `NeuronChangeProposal` requests, while both backends already have
  separate usage-score, candidate, budget, and tensor-mutation methods.
  Torch's prune choice is deliberately made after a noisy split on a
  detached candidate and can target a new child, so a common pre-sleep
  index-proposal framework would change its semantics.
- Added three deterministic NumPy policy tests at the phase-budget seam.
  The same one-split/one-prune proposal rejects at an early prune-zero
  boundary and a late split-zero boundary, preserving the full snapshot
  and model RNG. It succeeds in the middle phase, pruning the requested
  original neuron and splitting a different eligible source. The tests
  passed against the existing code before refactoring. The external
  proposal path was then split into typed request parsing, active cap and
  original-width eligibility validation, and score-based source ranking.
  Tensor mutation remains downstream of those validated indices. The
  default built-in selectors, Torch post-split planner, config fields,
  snapshots, checkpoint and protocol identities, and scientific metrics
  were not changed. ADR-0110 records the boundary and alternatives.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_numpy_proposal_preflight.py  # 16 passed before refactor
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_numpy_proposal_preflight.py tests/test_numpy_builtin_proposal_preflight.py tests/test_backend_parity_boundaries.py tests/test_torch_builtin_proposal_preflight.py tests/test_circadian_predictive_coding.py  # 66 passed in 2.77 s after refactor
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,393 passed, 41 skipped in 265.36 s
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # success, 207 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/core/circadian_predictive_coding.py tests/test_numpy_proposal_preflight.py  # 2 files formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Experiment artifacts: none; no scored experiment, seed selection, or
  sweep. The P4.3/P4.4 negative matched-control artifacts remain the
  previous session's evidence, not new P4.5 claims. Skipped tests: 41
  actual-device CUDA cases in the CPU environment; the cross-backend
  structural tests used CPU Torch only. Plan changes: check P4.5 after
  focused/full/static gates, retain P4.6 and later work unchecked, and
  move the handoff to the difficulty-modulation audit. No new dependency,
  remote write, or blocker. **Exact next action:** build a deterministic
  fixed-data NumPy/Torch fixture with clean rows, controlled label flips,
  and feature outliers; measure the existing reward-named
  mean-absolute-error/EMA scale and an unmodulated control, then predeclare
  clipped-error and loss-improvement comparators under the same seeds,
  work, and decision roles before selecting any new heuristic.

## 2026-09-29 — P4.6a supervised-error signal audit

- Completed task ID: **P4.6a**. P4.6b and P4.6 parent remain unchecked.
  The authoritative branch and HEAD remain
  `codex/research-protocols-and-resume` and
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; pre-existing dirty
  work and unrelated user changes were preserved. The historical
  reward-named mechanism is confirmed to be supervised mean absolute
  output error relative to an EMA, clipped to configured update bounds,
  in both NumPy and Torch. With modulation off, its scale is one and
  baseline stays absent. No model/config/checkpoint/protocol source was
  changed in this session.
- Four deterministic tests use only four synthetic train rows from a
  fixed logistic reference with clean 0.2/0.8 probabilities. One condition
  flips the first label; another moves only the first feature to 3.0,
  yielding a confident wrong probability about 0.9975. The existing
  NumPy and CPU Torch scale methods are called on the same float64 error
  arrays after clean calibration. Both yield scale 1.0 clean and the
  default 1.5 cap for either corruption, with the EMA updated after the
  scale calculation. A predeclared 0.5 per-row error clip gives ratio
  1.375 for either corruption. A constructed, equal 20% movement toward
  observed labels gives BCE improvement 0.048790 clean, 0.183539 for
  the flipped label, and 1.137313 for the feature outlier. The last
  signal requires post-update measurement and is largest on the outlier;
  it cannot be a same-step reward and this probe does not show it improves
  generalization. The unmodulated factor stays 1.0. No final role,
  outcome selection, or benchmark metric was used.
- Initial Torch fixture construction failed because its default minimum
  width exceeded the test's hidden width; a second attempt showed a
  Torch head with a module reference cannot be `deepcopy`-ed. The fixture
  now gives an explicit valid minimum width and uses the head's supported
  snapshot/restore path. Those were fixture errors, not negative model
  results. Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_difficulty_signal_audit.py  # fixture errors above, then 4 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_difficulty_signal_audit.py tests/test_circadian_predictive_coding.py tests/test_resnet50_variants.py  # 28 passed in 2.63 s
  .\.venv\Scripts\python.exe -c "import runpy; p=runpy.run_path('tests/test_difficulty_signal_audit.py'); b=p['_fixed_train_batches'](); f=p['_probabilities']; s=p['_offline_signals']; [(print(k, s(f(x),y))) for k,(x,y) in b.items()]"  # exact fixed signal values recorded in the audit document
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,397 passed, 41 skipped in 266.04 s
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # success, 208 source files
  .\.venv\Scripts\python.exe -m ruff format --check tests/test_difficulty_signal_audit.py  # formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Experiment artifact: `docs/difficulty-modulation-audit.md` records the
  train-only causal probe, exact values, and limitations; no scored local
  artifact or sweep was produced. ADR-0111 records why same-row loss
  improvement needs lookahead and why no heuristic was selected. Skipped
  tests: 41 actual-device CUDA cases on this CPU host. Torch signal
  parity is CPU-only, and the probe makes no accuracy or forgetting claim.
  Plan changes: split P4.6 into completed read-only signal audit (a) and
  open matched learning comparison (b), preserving all parent criteria.
  No dependency, remote write, or blocker. **Exact next action:** freeze a
  small versioned clean/label-flip/feature-outlier manifest for seeds 17
  and 19 with matched per-backend model/work and separated development/
  final roles; train historical-modulated and unmodulated controls, log
  train-only clipped-error/prior-step loss diagnostics, and preflight all
  trials before releasing held-out labels. Retain every outcome without
  selecting a new heuristic from this probe.

## 2026-09-29 — P4.6b matched difficulty comparison and P4.6 closure

- Completed task IDs: **P4.6b, P4.6**. Branch
  `codex/research-protocols-and-resume`, HEAD
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`. Existing dirty
  work and unrelated user changes were preserved. The reviewed plan
  commit remains `8793c49...`; this session used the current checkout
  and prior Phase 0–P4.6a evidence rather than assuming that static
  review was still current.
- The v11 manifest and acceptance boundary were written before scoring:
  seeds 17/19, clean/one B-train-label-flip/one B-train-feature-outlier,
  modulation off/on, shallow 8-wide NumPy and CPU Torch heads, two
  full-batch updates per phase at 0.02 and two inference iterations at
  0.1. Forty balanced development rows/phase were divided into disjoint
  train, inner guard, and outer selection roles; forty independent
  final rows/phase were deferred. The effective copied B train row has
  a separate content hash so clean role identities stay matched. Neither
  decision role selected settings. All 24 train-only trials and their
  A/total work, role/content hashes, scale and timing diagnostics were
  preflighted before any final source property was read. Saved A state
  and final B state yielded A-after-A, A-after-B, B-after-B accuracy and
  signed forgetting. The existing reward-named model switch alone
  differed within each backend; no core, config, snapshot, checkpoint,
  prior protocol ID, baseline, metric, or seed was changed.
- Result: all twelve backend/seed/condition matched pairs have identical
  held-out accuracy and forgetting at forty-row resolution, despite
  NumPy modulated B scales 1.0358–1.3204 and Torch CPU B scales
  1.0029–1.0068 (controls 1.0). Each trial used 4 optimizer updates,
  96 example presentations, 8 full-batch inference loops, 192
  example-inference iterations, and no replay or sleep. In Torch seed
  17, flipping one B train label reduced B accuracy .875→.700 in both
  arms. NumPy did not learn B, and seed 19 was weak on A/B; the fixed
  comparison therefore has limited power and does not establish general
  robustness. The clipped diagnostic was near its 0.5 ceiling at B
  arrival; prior-step loss improvement was logged but never applied.
  No new heuristic was selected. All rows and limitations are in
  `docs/difficulty-modulation-comparison.md` and ADR-0112.
- Commands and outcomes from the workspace root (Windows 11, Python
  3.14.7, Torch 2.14.0+cpu, CUDA unavailable):

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_difficulty_matched_benchmark.py tests\test_difficulty_signal_audit.py  # 12 passed in 1.83 s after final test edit
  .\.venv\Scripts\python.exe -m scripts.run_difficulty_matched_comparison --result data\difficulty-modulation-v11-result.json  # exit 0
  .\.venv\Scripts\python.exe -m scripts.run_difficulty_matched_comparison --result data\difficulty-modulation-v11-repeat.json  # exit 0; byte-identical SHA-256
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,405 passed, 41 skipped in 266.05 s
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # success, 213 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\app\difficulty_matched_benchmark.py src\core\difficulty_diagnostics.py src\infra\difficulty_streams.py tests\test_difficulty_matched_benchmark.py scripts\run_difficulty_matched_comparison.py  # five formatted files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Intermediate first smoke trained 24 trials successfully. Focused tests
  initially passed 10; initial targeted mypy found two local config
  variable-type errors and initial format check found two files to
  format. Those were fixed before final gates. Eight new tests cover
  one-row train-only corruption, equal within-backend initial state,
  all-trial final seal, forged total/A work and train hash rejection
  before release, changed manifest before source access, and exact
  repeat equality. No relevant CPU/Torch tests were skipped. The 41
  full-suite skips were actual-device CUDA cases on this CPU host.
- Local ignored experiment artifacts:
  `data/difficulty-modulation-v11-result.json` and
  `data/difficulty-modulation-v11-repeat.json`, both SHA-256
  `caf939687d54ba4b480e60b7d9593005093a479968989c42789de230b978b9b5`;
  manifest digest
  `8a274c3b7dd6c747284489f83e918c55cca4ebcfc5f7fa80b8abca2cbe5520d1`.
  Neither artifact overwrote historical results.
- Files added: `src/infra/difficulty_streams.py`,
  `src/core/difficulty_diagnostics.py`,
  `src/app/difficulty_matched_benchmark.py`,
  `scripts/run_difficulty_matched_comparison.py`,
  `tests/test_difficulty_matched_benchmark.py`, comparison document, and
  ADR-0112. Plan/README/architecture/changelog/module docs and this log
  were updated. Plan change: close P4.6b and parent with a valid null
  result, preserving the fixed budget and limitations; P4.7 remains
  unchecked. No external dependency, network run, large sweep, or
  blocker. **Exact next action:** inspect existing NumPy and Torch
  structural importance/ranking inputs to test whether a reward-aware
  term adds information beyond wake modulation and current importance
  weighting; predeclare a fixed-cap matched control probe before
  altering ranking or opening held-out outcomes.

## 2026-09-29 — P4.7a existing reward-to-structure rank audit

- Completed task ID: **P4.7a**. P4.7b and P4.7 parent remain unchecked.
  Branch `codex/research-protocols-and-resume`, HEAD
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; the same extensive
  pre-existing dirty tree and unrelated changes were preserved. The
  previous goal turn was progress: it completed the fixed v11 P4.6
  comparison and closed that parent. This turn rechecked AGENTS.md,
  current plan/handoff/log, status, ADR-0110–0112, and the current
  NumPy/Torch ranking implementations before the new audit.
- Inspection found that both cores already multiply per-neuron
  hidden-output gradient magnitude by the supervised-error reward scale
  inside `_update_importance_ema`. Both split and prune scores already
  mix that EMA with chemistry and output-weight norm. The fixed P4.7a
  protocol was documented before rank results: four neurons, two
  eligible original IDs, equal output norms, one split/prune slot,
  importance EMA decay 0.5, gradients `[6,0,0,0]` then
  `[0,2.5,0,0]`, and reward factors 1.0 then 1.5. It compares
  no-importance scoring, unweighted gradient importance, and existing
  reward-weighted importance on matched state in NumPy and CPU Torch.
- Both backends produced plain importance EMA `[1.5,1.25,0,0]` and
  historical weighted EMA `[1.5,1.875,0,0]`. The no-importance/plain/
  historical one-slot split choices were 0/0/1; prune choices were
  0/1/0. Applying the same positive factor 1.5 to both steps scaled
  the EMA but left min-max-normalized scores equal. Exact score pairs
  are recorded in `docs/structural-reward-ranking-audit.md` and
  ADR-0113. The model made zero wake updates, zero sleep events, and
  zero applied structural changes; all original IDs remained active.
  This establishes distinct *ranking* information, not measured
  learning benefit. No new heuristic, core config/serialization change,
  baseline retuning, final-test access, or remote experiment occurred.
- Commands and outcomes from the workspace root (Windows CPU host,
  Torch 2.14.0+cpu):

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_structural_reward_rank_audit.py  # 4 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_structural_reward_rank_audit.py tests\test_numpy_proposal_preflight.py tests\test_resnet50_variants.py  # 31 passed in 2.50 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,409 passed, 41 skipped in 266.74 s
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # success, 214 source files
  .\.venv\Scripts\python.exe -m ruff format --check tests\test_structural_reward_rank_audit.py  # formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Initial targeted mypy found only fixture variable-name type reuse;
  initial format check requested formatting. Those were fixed before
  focused/full/static gates. Four new backend-parametrized tests cover
  exact EMA, candidate scores/IDs, constant-scale invariance, and no
  wake/sleep/structure mutation. No relevant CPU/Torch test was skipped;
  41 full-suite skips were actual-device CUDA cases. Experiment artifact
  is the versioned read-only audit document and test fixture; no scored
  JSON or sweep was produced.
- Plan changes: split P4.7 into completed rank-only audit (a) and open
  matched outcome gate (b) because the supposedly prospective reward
  signal already exists in both backends. Parent acceptance is unchanged:
  wake scale, history weighting, and score importance must be separated
  under fixed global work/change caps, and complexity without measured
  benefit must be rejected. No blocker. **Exact next action:** freeze
  a small v12 train/final manifest prospectively, then implement a
  train-only feasibility/preflight runner varying wake modulation,
  reward weighting of importance history, and ranking importance mix
  independently. Verify equal model/data/work/change caps and actual
  candidate IDs before any final-role release; retain a null if the
  fixed trials do not separate decisions.

## 2026-09-29 — P4.7b fixed structural factor outcomes

- Completed task IDs: **P4.7b1, P4.7b2, P4.7b, P4.7**. Branch
  `codex/research-protocols-and-resume`, HEAD
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; the extensive
  existing dirty tree, unrelated user changes, and older ignored result
  files were preserved. This session reread AGENTS.md, the plan/handoff,
  current log, P4.7a/earlier comparison evidence, actual NumPy/Torch
  structural implementations, and checkout state before work.
- Before v12 training or final scoring, `docs/structural-ranking-comparison.md`
  froze a clean A→B source, new seeds 23/29, both CPU backends, three
  independent binary factors (wake scaling, reward weighting the existing
  importance EMA increment, and importance score mix), 8+8 full-batch
  updates, one A-boundary sleep, one split/prune cap, role splits, metrics,
  and no selection. `src/app/structural_rank_trial.py` applies post-step
  counterfactual corrections for the first two factors only in this
  experiment. `src/app/structural_rank_comparison.py` checks all 32
  unscored cells before shared final roles open. The script adapter
  writes exclusive new local JSON. A parity test checks both corrections
  against the disabled core after a nonunit reward update. ADR-0114
  records why these controls remain outside historical core/config/
  checkpoint identities.
- Every trial applied one split and one prune at the fixed sleep event,
  retained width 8 and 33 NumPy or 42 Torch parameters, and completed
  16 wake updates, 384 row presentations, 32 inference loops, 768
  example-inference loops, and zero replay. Train-only preflight checks
  role hashes, actual model clocks and parameter/lineage state, matched
  initialization, same-wake pre-sleep states and A-phase reward traces,
  actual stable structural IDs, and final-role seals. Tests reject
  changed manifest and forged work, role, structure, or pre-sleep facts
  before final data; source sentinels prove no train-only final read and
  that all 32 cells freeze before final release.
- Scored results: all sixteen reward-weighted-versus-plain importance-
  history pairs have identical chosen IDs and held-out metrics. Existing
  score mix changed the prune choice for Torch seed 23 and NumPy seed 29,
  reducing A-after-B accuracy in those cells. The comparison document
  retains all 32 outcomes and explicit within-seed deltas, including
  wake-factor effects and signed forgetting. Small 40-row final roles,
  weak A learning for seed 23, and failed NumPy B learning limit wider
  inference. The prospective bounded null rejects an *additional*
  reward-ranking term; it does not erase the historical importance EMA.
- Commands and outcomes from the workspace root (Windows CPU host):

  ```powershell
  .\.venv\Scripts\python.exe -m scripts.run_structural_rank_comparison --train-only --result data\structural-ranking-v12-train.json  # exit 0, 32 unscored cells
  .\.venv\Scripts\python.exe -m scripts.run_structural_rank_comparison --train-only --result data\structural-ranking-v12-train-repeat.json  # exit 0, byte-identical
  .\.venv\Scripts\python.exe -m scripts.run_structural_rank_comparison --result data\structural-ranking-v12-result.json  # exit 0, 32 scored cells
  .\.venv\Scripts\python.exe -m scripts.run_structural_rank_comparison --result data\structural-ranking-v12-result-repeat.json  # exit 0, byte-identical
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_structural_rank_comparison.py tests\test_structural_reward_rank_audit.py tests\test_difficulty_matched_benchmark.py  # 22 passed in 4.11 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,419 passed, 41 skipped in 268.28 s
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # success, 218 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\app\structural_rank_trial.py src\app\structural_rank_comparison.py scripts\run_structural_rank_comparison.py tests\test_structural_rank_comparison.py  # four formatted files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- No relevant NumPy/Torch CPU tests were skipped. The 41 full-suite skips
  were actual-device CUDA tests on this CPU host. Ten new v12 tests cover
  factor parity, global seal and sentinels, tampered facts, manifest
  rejection, and exact result repeat. An initial smoke found the typed
  sleep telemetry outcome is `applied`, not `accepted`; this was fixed.
  Initial preflight incorrectly required post-sleep B reward traces to
  match across rank factors; only the pre-sleep A prefix can be equal,
  and that assertion was corrected before release. A targeted mypy
  complaint about a heterogeneous group check was fixed. No result was
  used to change a seed, metric, threshold, budget, or stopping point.
- Ignored local experiment artifacts:
  `data/structural-ranking-v12-train.json` and its `-repeat.json` have
  SHA-256 `f4137041aa152c9fca17d8d990a0f89b7329a25a9cfe9b907e2e5e54d29c0379`;
  `data/structural-ranking-v12-result.json` and its `-repeat.json` have
  SHA-256 `1f4bb8017f799fe5f3572ebb5830a51a65d654bde6e23764365121e49844a309`.
  Manifest digest is
  `7ef8e0cf5b27074fcae4425538bdb981dce89b824eed6799b6751635ce2e5491`.
  No network experiment or large sweep ran.
- Plan amendment and rationale: before outcomes, split P4.7b into b1
  train-only feasibility and b2 final-release gates without weakening
  parent acceptance. After both gates and full checks passed, close
  P4.7b1/b2, P4.7b, and P4.7 with an explicit bounded negative result.
  README, architecture, changelog, module docs, comparison, ADR-0114,
  and handoff were updated. **Blockers:** none. **Exact next action:**
  inspect the existing periodic/adaptive trigger decision paths and
  A/B phase sources for P4.8, then prospectively freeze a bounded
  stationary-noise and actual-shift comparison of periodic, current
  adaptive, and no-sleep controls with equal data/wake/inference work,
  sleep opportunity accounting, and sealed evaluation roles before
  any new trigger heuristic or held-out scoring.

## 2026-09-29 — P4.8a fixed-capacity sleep-trigger timing control

- Completed task ID: **P4.8a**. P4.8b and P4.8 parent remain unchecked.
  Branch `codex/research-protocols-and-resume`, HEAD
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`; the extensive
  existing dirty tree and unrelated user changes were preserved. The
  previous goal turn was progress (P4.7b1/b2 and parent completed).
  This turn rechecked AGENTS.md, plan/handoff/log, checkout state,
  NumPy core trigger/sleep semantics, existing schedule and matched role
  boundaries before implementation.
- Before v13 training or scoring, the plan split P4.8 into fixed-capacity
  timing isolation (a) and broader structural/replay/guarded sleep (b),
  retaining the parent comparison. `docs/sleep-trigger-comparison.md`
  prospectively froze seeds 41/43, stationary noisy A→B and axis-shift
  A→B, periodic/current adaptive/no-sleep arms, 16+16 full-batch updates
  with shared train-only jitter, role fractions, fixed width, four
  possible forced periodic events, unchanged adaptive thresholds, and
  accuracy/BCE/forgetting metrics. Why this: core adaptive readiness
  combines a ten-wake-batch spacing, eight-energy window, plateau
  `≤0.001`, and chemical variance `≥0.02`; topology/replay differences
  would obscure this first timing comparison. Only the existing chemical
  reset was active during component sleep. ADR-0115 records the scope.
- New `src/infra/trigger_streams.py` supplies paired noisy/axis-shifted
  development rows with independent final properties.
  `src/app/sleep_trigger_trial.py` uses existing schedule/core paths and
  records exact effective train hashes, per-epoch trigger facts, weight
  invariance across sleep, chemistry reset, and actual work.
  `src/app/sleep_trigger_comparison.py` preflights the complete twelve-cell
  Cartesian before releasing common final roles once per seed/condition.
  `scripts/run_sleep_trigger_comparison.py` writes exclusive local JSON;
  `tests/test_sleep_trigger_comparison.py` verifies role/data isolation,
  matched work and capacity, global final seal, forged role/batch/work/
  trace/model rejection, changed-manifest rejection, final-label state
  isolation, adaptive/no-sleep parity, and exact result repeat.
- Every trial used 32 decision opportunities and wake updates, 768 row
  presentations, 64 inference loops, 1,536 example-inference loops,
  zero replay, width eight, and 33 parameters. Periodic performed four
  chemical-reset-only events per trial; adaptive/no-sleep performed none.
  Among 23 spacing/window-eligible adaptive opportunities per trial,
  chemical variance crossed the unchanged 0.02 threshold zero times.
  Axis-shift seed 41 had seven plateau windows, but still no variance
  crossing. No adaptive threshold was tuned to force an event.
- The final study retains all twelve scored rows and paired deltas.
  Adaptive equals no-sleep exactly in saved states, A/B accuracy, BCE,
  and signed forgetting. Periodic lowers B BCE in all four cells but
  worsens A-after-B BCE by 0.002638 in axis-shift seed 41 and lowers
  B accuracy by 0.025 in axis-shift seed 43. Forty-row final roles and
  only two seeds limit interpretation. This is a valid narrow negative
  for unchanged adaptive timing here, with mixed periodic effects; no
  new trigger heuristic, seed, threshold, metric, or baseline was chosen
  from final outcomes. The full sleep-stack question remains P4.8b.
- Commands and outcomes from the workspace root (Windows CPU host):

  ```powershell
  .\.venv\Scripts\python.exe -m scripts.run_sleep_trigger_comparison --train-only --result data\sleep-trigger-v13-train.json  # exit 0, 12 unscored cells
  .\.venv\Scripts\python.exe -m scripts.run_sleep_trigger_comparison --train-only --result data\sleep-trigger-v13-train-repeat.json  # exit 0, byte-identical
  .\.venv\Scripts\python.exe -m scripts.run_sleep_trigger_comparison --result data\sleep-trigger-v13-result.json  # exit 0, 12 scored cells
  .\.venv\Scripts\python.exe -m scripts.run_sleep_trigger_comparison --result data\sleep-trigger-v13-result-repeat.json  # exit 0, byte-identical
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_sleep_trigger_comparison.py  # 12 passed in 1.79 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # 1,431 passed, 41 skipped in 270.75 s
  .\.venv\Scripts\python.exe -m ruff check .  # all checks passed
  .\.venv\Scripts\python.exe -m mypy  # success, 223 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\infra\trigger_streams.py src\app\sleep_trigger_trial.py src\app\sleep_trigger_comparison.py scripts\run_sleep_trigger_comparison.py tests\test_sleep_trigger_comparison.py  # five formatted files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- Initial local smoke exposed that `train_epoch` returns a typed result,
  not a bare energy; the trial now records its `.energy`. An exact-equality
  source test hit floating subtraction roundoff and was corrected to
  numeric tolerance without changing the protocol. Formatting changes
  were applied before the final gates. No relevant CPU tests were
  skipped; the 41 full-suite skips were actual-device CUDA cases.
- Ignored local artifacts: `data/sleep-trigger-v13-train.json` and its
  repeat both SHA-256
  `5b1ae508b074f54198518bc908b733fbae8eff6f17c15868be9df428369c279f`;
  `data/sleep-trigger-v13-result.json` and its repeat both SHA-256
  `e8dc920c1ca6944d5f0c48cc87023aaed0548cd830ac07a6c05746fefa6a9821`.
  Manifest digest:
  `b361d99e1354d3de3a4099859177678a77bbb162ac28a94422d0162024ebb183`.
  No network experiment, large sweep, new dependency, or historical
  config/checkpoint/protocol identity change occurred.
- Plan changes: mark P4.8a complete only after its outcome and quality
  gates; keep P4.8b and parent unchecked because chemical-reset timing
  does not test structural/replay/guarded sleep. README, architecture,
  changelog, module docs, comparison, ADR-0115, and handoff were updated.
  **Blockers:** none. **Exact next action:** inspect the guarded
  continual runner, v9 matched replay schedule, and disjoint shifted
  source; prospectively freeze a bounded P4.8b periodic/adaptive/no-
  sleep protocol with explicit wake/replay exposure, structural/capacity
  caps, guard decisions, selection roles, and final-role release before
  any full-stack outcome or new trigger heuristic.

## 2026-09-29 — P4.8b1 train-only full-stack trigger replay supply

- Completed task ID: **P4.8b1**. P4.8b2, P4.8b, and P4.8 remain
  unchecked. The session started on branch
  `codex/research-protocols-and-resume`, HEAD
  `ec7634d174e98a83b6623a5cdb3c3960f215c4cb`, with an extensive
  dirty tree. During validation the checkout changed to `master`, HEAD
  `5134a17db04d94d4afa26150dfae1939e724a6f4`; commits
  `7ff2f55`, `5a262e2`, `6f06d86`, and `5134a17` captured the
  previously dirty work and new schedule files. I did not reset,
  amend, or overwrite those commits. The remaining local changes are
  this plan, the evaluation-protocol note, and this log; older ignored
  artifacts remain. This session reread AGENTS.md, the full plan/current
  log and handoff, inspected the actual checkout and v9 guarded runner,
  matched schedule, shared buffer, arrived four-role source, core
  configuration, and prior ADRs. The reviewed commit remains older than
  the active checkout; earlier Phase 0 evidence is retained.
- Before any v14 training or scoring, `docs/full-stack-trigger-comparison.md`
  froze the shifted arrived A→B source, new seeds 47/53, 12+12 wake
  epochs, three trigger arms, full component sleep settings, phase-local
  interval four, unchanged adaptive defaults, FIFO 8-row/192-byte
  newest-two replay, guard tolerance zero, two replay updates per
  accepted event per method, and global width/split/prune caps. The
  v14 schedule then offered prediction-independent potential replay at
  **every** train-only wake epoch. Each seed had 24 opportunities;
  phase A had 72 train-role rows per epoch and B had 36. Its six
  periodic opportunities per phase pair matched the v9 schedule's
  train hash, sorted retention, retention order, selected IDs, and
  three method-work records exactly. The buffer retained eight
  distinct rows/192 bytes and selected two detached copies at each
  opportunity. No model trained, no decision/final role was opened, and
  no held-out outcome was observed.
- Fourteen focused tests cover both seeds, exact v9 parity, B-arrival
  ordering, final and guard/outer access sentinels, train role and
  manifest mutation rejection before observation, detached copies,
  budgets, exact payload repeat, actual file-byte hash, and exclusive
  output. No relevant CPU tests were skipped. The full suite's 41 skips
  were actual-device CUDA tests on this CPU host.
- Commands and outcomes from the workspace root (Windows CPU host):

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_continual_trigger_replay_schedule.py  # exit 0, 14 passed
  .\.venv\Scripts\python.exe -m scripts.run_continual_trigger_replay_schedule --result data\trigger-replay-v14-opportunities-verified.json  # exit 0
  .\.venv\Scripts\python.exe -m scripts.run_continual_trigger_replay_schedule --result data\trigger-replay-v14-opportunities-verified-repeat.json  # exit 0, byte-identical
  Get-FileHash data\trigger-replay-v14-opportunities-verified.json -Algorithm SHA256  # 1C30E960...5D168, matches adapter output
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0, 1,445 passed, 41 skipped in 283.47 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0, 226 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\app\continual_trigger_replay_schedule.py scripts\run_continual_trigger_replay_schedule.py tests\test_continual_trigger_replay_schedule.py  # exit 0, three formatted files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- The first writer smoke used Windows text newline translation. Its two
  ignored preliminary files were byte-identical to each other at on-disk
  SHA-256 `6f619a13f200bbfdbd6b9e259b102398fa5088a2f6db26091d8fee489bd6a80b`,
  but the adapter reported the LF payload hash. The writer now pins LF
  bytes, and a regression checks actual disk bytes and refuses overwrite.
  The canonical ignored files are
  `data/trigger-replay-v14-opportunities-verified.json` and its
  `-repeat.json`, both on-disk SHA-256
  `1c30e960aa2dee65a862434fb584e12eaa31bd14a71cb73b3eb919f9dfa5d168`.
  Manifest digest is
  `22e90b3b3b5312ea52abc956f8ac997a457126bb07a9e03ef0f404079a5a87d0`.
  The full suite was also run before that one-line writer correction
  (1,444 passed/41 skipped); the final 1,445-pass run reflects the
  exact handed-off code and added regression.
- Files added: `src/app/continual_trigger_replay_schedule.py`,
  `scripts/run_continual_trigger_replay_schedule.py`,
  `tests/test_continual_trigger_replay_schedule.py`, comparison document,
  and ADR-0116. README, architecture, changelog, evaluation protocol,
  module docs, plan, and this log were updated. Plan amendment split
  P4.8b into b1 replay-supply feasibility and b2 actual guarded outcomes
  because v9 explicitly rejects adaptive triggering and only emits
  periodic boundaries. Original full-stack criteria remain under b2 and
  parent; no seed, baseline, threshold, metric, or stopping point was
  selected from results. No new dependency, network experiment, large
  sweep, or blocker. **Exact next action:** implement a v14 train-only
  runner above `TriggerReplayScheduleSession`; after each arrived wake
  update, check core retention/selection against the offered rows,
  invoke the existing guarded sleep decision for the fixed arm, and
  replay detached rows to PC/backprop only on accepted events. First
  test rejected/skipped/no-sleep zero-replay and periodic parity with
  final-role sentinels; then build the six-trial global preflight and
  scorer required by P4.8b2.

## 2026-09-29 — P4.8b2 guarded full-stack outcomes and P4.9 backend scope

- Completed task IDs: **P4.8b2a, P4.8b2b, P4.8b2, P4.8b, P4.8,
  P4.9**. Phase 5 remains open. The checkout stayed on `master` at
  `5134a17db04d94d4afa26150dfae1939e724a6f4`; tracked v14 docs
  and untracked v14 code/tests/ADRs are preserved in the working tree.
  The prior P4.8b1 log records the earlier branch/commit transition.
  AGENTS.md, the full plan, current log, active checkout, v9 runner,
  v14 schedule, NumPy/Torch core capabilities, and current artifacts
  were inspected before the handoff was changed.
- Before v14 model training or final access, P4.8b2 was split into
  train-only guard-committed replay (b2a) and complete global final
  release/outcomes (b2b). The split isolates rollback and no-replay
  correctness from final scoring; the original b2/b/parent acceptance
  criteria and frozen seeds, arms, metrics, budgets, and stopping point
  were retained. New `continual_trigger_replay_runner.py` trains all
  three NumPy models on each arrived wake epoch, preflights the offered
  retention/order/selection, and gives detached replay to PC/backprop
  only after an accepted circadian guarded event. No baseline replay
  occurs after rollback, skip, no-sleep, or core failure.
- All six fixed seed/arm trials completed 24 wake updates and 1,296
  arrived row presentations per method; both PC methods used 48 wake
  inference loops and 2,592 example-inference iterations. Within a
  seed, all arms saw the same 24 offered replay selections. Periodic
  accepted six guarded sleeps per seed, applied twelve exact selected
  replay rows per method, and pruned two (seed 47) or three (seed 53)
  neurons. Adaptive and no-sleep attempted zero sleeps and applied zero
  replay at the unchanged thresholds. All widths stayed within 4–32
  and the fixed total structural caps; there were no splits or
  rollbacks. The six-trial train-only preflight rederived role and
  opportunity identity, event/guard/work/capacity/clock facts, and
  structural stable-ID lineage without final values.
- The outcome gate reran that preflight before any final read, released
  all twelve A/B final roles, compared common within-seed IDs/hashes,
  and only then scored all 18 method cells and 18 signed contrasts.
  Forged train state rejects before any final read, and mismatched
  final hashes reject before a score. All outcomes remain in
  `docs/full-stack-trigger-comparison.md` and the scored JSON. Adaptive
  and no-sleep are exactly equal; periodic effects vary by seed and
  method. Seed-53 ordinary PC has zero A-after-A accuracy in every arm,
  so its large periodic gain is an underlearning observation. The
  40-row final roles have 0.025 accuracy resolution. This bounded
  synthetic study does not establish a general circadian advantage or
  justify a new adaptive trigger.
- Local ignored artifacts and repeats, each pair byte-identical:
  `data/trigger-replay-v14-training.json` and `-repeat.json`, on-disk
  SHA-256 `174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324`;
  `data/trigger-replay-v14-outcomes.json` and `-repeat.json`, on-disk
  SHA-256 `ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f`.
  Both writers use exclusive new paths and LF bytes. No large sweep,
  new dependency, network experiment, or historical protocol change.
- P4.9 inspected `docs/feature-inventory.md`, current NumPy/Torch
  config/core/runner capabilities, and the v14 JSON. The new
  `docs/backend-capability-matrix.md` records shared and distinct
  features, clocks, structural planners, snapshots, and protocol
  scope. `docs/result-backend-metadata.json` binds the original v14
  protocol IDs and file hashes to `array_backend: numpy`, the three
  method names, and `torch_included: false`. Both local SHA-256 and
  protocol-ID checks matched the sidecar. The next P5.1 task defines
  an artifact schema, so no Torch consolidation was required. ADR-0119
  records why the scored v14 bytes remain unchanged; a future Torch
  replay experiment needs its own prospective protocol.
- Commands and outcomes from the workspace root (Windows CPU host):

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_continual_trigger_replay_schedule.py tests\test_continual_trigger_replay_runner.py tests\test_continual_trigger_replay_training_study.py tests\test_continual_trigger_replay_outcomes.py tests\test_continual_matched_replay_runner.py tests\test_continual_matched_replay_outcomes.py  # exit 0, 63 passed in 8.72 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0, 1,469 passed, 41 skipped in 283.12 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0, 234 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\app\continual_trigger_replay_runner.py src\app\continual_trigger_replay_training_study.py src\app\continual_trigger_replay_outcomes.py scripts\run_continual_trigger_replay_training.py scripts\run_continual_trigger_replay_outcomes.py tests\test_continual_trigger_replay_runner.py tests\test_continual_trigger_replay_training_study.py tests\test_continual_trigger_replay_outcomes.py  # exit 0, eight files already formatted
  .\.venv\Scripts\python.exe -m scripts.run_continual_trigger_replay_training --result data\trigger-replay-v14-training.json  # exit 0; repeat to a new -repeat.json matched bytes
  .\.venv\Scripts\python.exe -m scripts.run_continual_trigger_replay_outcomes --result data\trigger-replay-v14-outcomes.json  # exit 0; repeat to a new -repeat.json matched bytes
  Get-FileHash data\trigger-replay-v14-training.json -Algorithm SHA256  # 174ee794...28b324; sidecar matched
  Get-FileHash data\trigger-replay-v14-outcomes.json -Algorithm SHA256  # ea11fc7c...2501f; sidecar matched
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- No relevant CPU tests were skipped. The 41 full-suite skips require
  actual-device CUDA on this CPU host. The full suite was run after
  the final v14 code/lineage preflight changes; later changes were
  documentation and the metadata sidecar only. Plan changes marked
  P4.8b2a/b and their parents complete with exact evidence, closed
  P4.9 on the verified capability record, and moved the handoff to
  P5.1. README, architecture, changelog, evaluation protocol, module
  docs, comparison, inventory, ADRs 0117–0119, and plan were updated.
  **Blockers:** none. **Exact next action:** inspect existing v14 JSON
  adapters, trusted checkpoint formats, and historical provenance
  fields; draft the smallest versioned P5.1 run-schema contract with
  explicit required versus unavailable provenance, then implement and
  test one opt-in local artifact writer without changing v14 or earlier
  protocol bytes. Keep P5.1 unchecked until all required schema fields
  and its validation gate pass.

## 2026-09-29 — P5.1 versioned execution manifest and opt-in v14 bundle

- Completed task IDs: **P5.1a, P5.1b, P5.1**. P5.2–P5.7 remain open.
  Continued on `master` at HEAD
  `5134a17db04d94d4afa26150dfae1939e724a6f4`, preserving the
  existing dirty v14 work and all ignored local results. Rechecked
  AGENTS.md, the P5.1 plan/handoff and current log, actual checkout,
  existing v14 adapters, trusted checkpoint store, algorithm IDs,
  four-role source hashes, and historical provenance gaps. The prior
  goal turn made progress by completing fixed v14/P4.9 work; this
  continuation executed the next open task rather than only rewriting
  the plan.
- Split P5.1 before implementation into pure schema validation (a)
  and one opt-in producer/reader (b). The fixed v14 JSON already had
  protocol/results but lacked execution provenance; trusted pickle
  checkpoints have a different purpose. The split retained every P5.1
  field and left P5.2 observation streams and P5.3 interrupted-write
  recovery open. A red schema test initially failed collection because
  `src/core/run_manifest.py` did not yet exist; eight schema cases and
  two environment cases later passed. The schema requires a safe run
  ID, status, protocol/algorithm IDs, commit and coherent dirty state
  (or explicit Git unavailability), full resolved config/digest,
  numeric seed map, source-role hashes, pretrained identity state,
  Python/NumPy versions, CPU model or reason, float64 precision,
  determinism, timing scope, and hash-bound output files. It rejects
  nonfinite/malformed/omitted fields and false completion.
- `src/infra/run_environment.py` captures Git HEAD, porcelain status,
  tracked binary diff, and hashes nonignored untracked source contents
  before training. It records runtime and CPU facts; a fixture verified
  that changing an untracked file changes the workspace digest even
  when the status listing stays the same. The v14 adapter captures
  source again after scoring and refuses publication on drift. The
  scored study uses the same six train-only trials after repeating the
  global preflight; train-only serialization occurs before final-role
  release. Existing v14 train-only and outcome adapter `build_payload()`
  bytes remain exactly at their earlier SHA-256 values.
- The new local bundle writer writes `training.json`, `outcomes.json`,
  and then `manifest.json` under a new ignored
  `artifacts/runs/<run-id>/` directory. Its reader verifies schema,
  status, protocol/algorithm IDs, resolved config/digest, file SHA-256,
  all six ordered seed/arm cells, three method rows, 18 contrasts,
  development/final role hashes, and directory identity. Tests cover
  exclusive output before training, final file tamper, missing
  manifest, a partial cell grid with an updated checksum, and source
  drift with no published directory. No resume or atomic directory
  replacement is claimed; those remain P5.3.
- Two fresh bounded local runs, `artifacts/runs/p51-v14-schema-c/` and
  `artifacts/runs/p51-v14-schema-d/`, both passed the CLI verifier.
  Their train-only files repeat SHA-256
  `174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324`;
  their scored files repeat SHA-256
  `ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f`.
  Manifest SHA-256 values differ by run ID:
  `702f5f75c86467a980f2410c77f58976dc428526c89bc7a6453612d95085024c`
  and `604bee7973fa341645d4d9baf6faa2d426039fe4b273f1f69f5650d4486bde29`.
  Source provenance and all role hashes are identical across the two
  fresh runs. The recorded dirty workspace digest is
  `0bee07214fc072cf32c0ae67fb00466b770c46aa010629b7d52cd882e3aaea2a`;
  runtime Python 3.14.7, NumPy 2.4.6, Windows 11 CPU with Intel64
  Family 6 Model 151, float64, no pretrained weights, and no measured
  timing. Earlier ignored preliminary `p51-v14-schema-a/b` bundles
  predate the CPU-model field and are not canonical P5.1 evidence.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_run_manifest_schema.py tests\test_run_environment.py tests\test_versioned_v14_run.py tests\test_continual_trigger_replay_outcomes.py tests\test_continual_trigger_replay_training_study.py  # exit 0, 34 passed
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id p51-v14-schema-c  # exit 0
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id p51-v14-schema-d  # exit 0
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --verify-run artifacts/runs/p51-v14-schema-c  # exit 0, completed
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --verify-run artifacts/runs/p51-v14-schema-d  # exit 0, completed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0, 1,486 passed, 41 skipped in 287.39 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0, 242 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\core\run_manifest.py src\infra\run_environment.py src\infra\versioned_run_files.py src\app\versioned_v14_run.py scripts\run_versioned_v14_bundle.py tests\test_run_manifest_schema.py tests\test_run_environment.py tests\test_versioned_v14_run.py src\app\continual_trigger_replay_outcomes.py scripts\run_continual_trigger_replay_training.py scripts\run_continual_trigger_replay_outcomes.py  # exit 0, 11 files already formatted
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- The 41 full-suite skips require actual-device CUDA on this CPU host;
  no relevant CPU test was skipped. Added `src/core/run_manifest.py`,
  `src/app/versioned_v14_run.py`, two `src/infra/` modules, one script,
  three test files, `docs/versioned-run-manifest.md`, ADR-0120, and a
  narrow `/artifacts/runs/` ignore rule. The existing v14 scorer and
  adapters gained reusable one-study/serialization functions; their
  historical payload bytes and protocol IDs stayed exact. README,
  architecture, module docs, evaluation protocol, changelog, plan, and
  this log were updated. No network experiment, dependency addition,
  algorithm change, result selection, or large sweep. **Blockers:**
  none. **Exact next action:** inspect the actual v14 per-epoch
  opportunity/event fields and scored seed rows; define a versioned
  JSONL projection from a verified P5.1 bundle for sleep, topology,
  replay, validation, and final records while retaining raw seed data.
  Mark missing per-epoch metrics as unavailable until a separately
  versioned instrumentation path records them; add derived CSV only
  after the raw-stream and role/hash gates pass. Keep P5.2 unchecked
  until all its observation types and output formats are verified.

## 2026-09-29 — P5.2a observed v14 streams

- Completed task ID: **P5.2a**. P5.2b and P5.2 parent remain open.
  Continued on `master` at HEAD
  `5134a17db04d94d4afa26150dfae1939e724a6f4`, preserving the
  pre-existing dirty working tree and ignored P5.1 bundles. Re-read
  AGENTS.md, the full living plan and latest log, then inspected the
  actual checkout and both completed P5.1 local bundles. This session
  implemented and validated the next unblocked task after amending
  the plan; it did not run a new sweep or alter an algorithm.
- The v14 train-only JSON has six seed/arm rows and 144 ordered
  opportunities with typed sleep, topology, retention, offered IDs,
  applied matched replay work, 12 inner-guard decisions, and 516 role
  accesses. The scored JSON has 18 full method rows. Inspection of
  `base._train_named_model_epoch` shows it discards each core
  `train_epoch` return, so no genuine per-epoch loss or energy exists
  in the saved v14 source. P5.2a wake rows explicitly say
  `unavailable_not_recorded`; final/guard scores were not repurposed.
- Added pure `src/app/v14_observation_projection.py` and
  `src/infra/observation_projection_files.py`, plus a local CLI and
  tests. The application transform checks all six cells, per-trial
  epoch ordering, train-role hashes, clock/width/work identity, guard
  alignment, and the train-only final/outer role seal. It derives
  deterministic JSONL for wake, sleep, topology, replay, validation,
  role access, and final method rows, plus final-row CSV. The file
  boundary first verifies the P5.1 complete source, writes an
  exclusive `observations-v1/` directory with source/output hash and
  count metadata last, and verifies exact regeneration. Tests reject
  a missing file, rehashed forged stream, wrong source directory,
  and reordered source epochs after their checksum is updated. The
  raw seed JSON files and frozen v14 train/result SHA-256 values remain
  unchanged.
- Local projections of
  `artifacts/runs/p51-v14-schema-c/observations-v1/` and
  `artifacts/runs/p51-v14-schema-d/observations-v1/` each verify.
  Their eight data-file hash maps are identical, while the
  run-specific source-manifest hashes differ. Counts: 144 wake,
  144 sleep (12 accepted/132 skipped), 144 topology, 144 replay,
  12 guard, 516 role-access, 18 final JSONL, and 18 final CSV data
  rows. Per-opportunity applied replay work totals 72 optimizer
  updates across methods. Representative SHA-256 values:
  `wake-epochs.jsonl`
  `1de24cebd3b04fb1d17482e0dca12fe2fafab5dde57dc3c5f00e7c80efe4a0ae`,
  `final-results.jsonl`
  `dbe9b98ea72ee576c04653bf54162e883c26b18f3a1b24bde94a507d06602006`,
  and `summary.csv`
  `75c4ac48de3b2e58346e290f81578c2956dd82cfc0040f860797bac640682b3e`.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q tests/test_v14_observation_projection.py  # red before implementation: missing module; then 5 passed
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --run artifacts/runs/p51-v14-schema-c  # exit 0
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --run artifacts/runs/p51-v14-schema-d  # exit 0
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --verify-run artifacts/runs/p51-v14-schema-c  # exit 0
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --verify-run artifacts/runs/p51-v14-schema-d  # exit 0
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0, 1,491 passed, 41 skipped in 283.79 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_v14_observation_projection.py tests/test_versioned_v14_run.py tests/test_continual_trigger_replay_training_study.py tests/test_continual_trigger_replay_outcomes.py  # exit 0, 29 passed after missing-file test
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0, 246 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/app/v14_observation_projection.py src/infra/observation_projection_files.py scripts/project_v14_observations.py tests/test_v14_observation_projection.py  # exit 0, four formatted files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- The 41 suite skips require actual-device CUDA on this CPU host;
  no relevant CPU test was skipped. Added
  `docs/structured-observation-audit.md`, ADR-0121, and narrow
  README/architecture/module/evaluation/changelog notes. The plan
  split P5.2 into a projection (a) and separate genuine metric capture
  (b) because recorded v14 returns are absent; this preserves the
  original P5.2 per-epoch acceptance criterion. No baseline, seed,
  score, metric, or selection rule changed. **Blockers:** none.
  **Exact next action:** inspect the Backprop, PC, and circadian
  `train_epoch` result types and all callers of
  `base._train_named_model_epoch`; write a red test for opt-in capture
  of one typed pre-update diagnostic per method/phase/epoch under a
  new observation identity, with matched update counts and the global
  final-role seal. Keep the historical v14 JSON bytes and P5.2 parent
  unchanged until measured rows, repeatability, and full gates pass.

## 2026-09-29 — P5.2b real wake diagnostics and measured projection

- Completed task IDs: **P5.2b1, P5.2b2, P5.2b, P5.2**. P5.3–P5.7
  remain unchecked. Previous goal turn was progress: P5.2a was
  implemented, fully tested, logged, and checked. This continuation
  re-read AGENTS.md, the living plan, latest log, and ADR-0121, and
  inspected the actual checkout before coding. Continued on `master`
  at HEAD `5134a17db04d94d4afa26150dfae1939e724a6f4` with the
  pre-existing tracked/untracked changes and ignored local artifacts
  preserved. No commit or dependency change was made.
- Inspection found that Backprop returns pre-gradient binary BCE;
  ordinary PC returns its all-hidden-error energy after latent
  inference and before the gradient update; circadian PC returns its
  final-hidden-error energy at the same stage, after possible pending
  prune decay. The shared `base._train_named_model_epoch` discarded
  these results at six call sites. It now returns the already-created
  core result; all historical callers ignore it. Only the fixed v14
  runner opts in to copying 432 successful train-only returns across
  seeds 47/53, periodic/adaptive/no-sleep arms, A/B phases, 24 epochs,
  and three NumPy methods. No extra model update, guard/final read,
  criterion, seed, or algorithm change was introduced. Study preflight
  rejects incomplete/misordered/nonfinite/incorrectly defined metric
  rows before global final release. A direct comparison of default
  and captured studies passed equal serialized training, model
  digests, wake work, role audit, and fixed v14 output hashes.
- Split P5.2b into capture/preflight (b1) and completed-run
  publication/projection (b2) before implementation; the split
  preserved all original criteria. `v14_wake_diagnostics_v1` writes
  canonical 432-row JSONL only after the original v14 scorer has
  preflighted all six trials, released common final roles, scored
  them, and published a verified completed P5.1 bundle. The sidecar
  manifest binds source run ID, source manifest and two raw payload
  hashes, diagnostic hash/count, and its own observation ID. A
  distinct `v14_measured_observation_projection_v1` retains all eight
  P5.2a streams and adds 432-row wake JSONL and direct CSV. The
  final-only 18-row CSV remains separate. Tests reject missing and
  changed sidecar files, a rehashed forged projection, nonfinite or
  wrong-role diagnostics, and a forced global scoring failure before
  any run directory is published. The manifests are local integrity
  records, not signed attestation. Atomic replacement and interrupted
  write recovery remain P5.3.
- Two bounded local invocations, `artifacts/runs/p52-measured-a/` and
  `artifacts/runs/p52-measured-b/`, used Python 3.14.7, NumPy 2.4.6,
  Windows 11 CPU (Intel64 Family 6 Model 151), the fixed v14 seeds
  47/53 and three arms, with the same source digest
  `0df280d0305d088e9124ab62b717dc752a1c3437c23b0adf4918a117c5e16039`,
  configuration, roles, and resolved work. Both raw training files
  repeat SHA-256
  `174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324`;
  both scored files repeat
  `ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f`.
  Both 432-row diagnostic files repeat
  `2d6b6e4b0bc86a3dc292e7f04ad82816ae698dd43b9e63a5fceb41fe85f49f7c`.
  Their ten measured projection data-file hash maps are identical;
  `wake-metrics.csv` is
  `6eb17df53d18cf48bf48157170fa929bcd8d183744546283640439bdc9760386`.
  Run-bound P5.1 manifest hashes differ by ID:
  `41b4f933ded7ec3b7602bf353139a47b21973e45dab6f6b854f3bc7115035d7d`
  and `36397f8fec580df6a3e56cd6259d2dfd99ce18935fb598d59938d2b53c6d5aad`.
  Both sidecars and projections verify; the existing negative/mixed
  method outcomes remain unchanged. These are same-environment
  repeatability checks, not cross-platform bitwise claims.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_v14_wake_diagnostics.py  # initial red import; then 3 passed after replacing a duration-sensitive assertion with exact serialized comparison
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_v14_measured_observations.py  # initial red import; then 3 passed before final missing-manifest case
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id p52-measured-a --capture-wake-diagnostics  # exit 0
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id p52-measured-b --capture-wake-diagnostics  # exit 0
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --run-measured artifacts/runs/p52-measured-a  # exit 0
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --run-measured artifacts/runs/p52-measured-b  # exit 0
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --verify-measured-run artifacts/runs/p52-measured-a  # exit 0
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --verify-measured-run artifacts/runs/p52-measured-b  # exit 0
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_v14_wake_diagnostics.py tests/test_v14_measured_observations.py tests/test_v14_observation_projection.py tests/test_versioned_v14_run.py tests/test_continual_trigger_replay_training_study.py tests/test_continual_trigger_replay_outcomes.py  # exit 0, 36 passed after missing-manifest case
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0, 1,498 passed, 41 skipped in 303.47 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0, 251 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/app/wake_diagnostic.py src/app/v14_measured_observations.py src/app/continual_shift_benchmark.py src/app/continual_trigger_replay_runner.py src/app/continual_trigger_replay_training_study.py src/infra/measured_observation_files.py scripts/run_versioned_v14_bundle.py scripts/project_v14_observations.py tests/test_v14_wake_diagnostics.py tests/test_v14_measured_observations.py  # exit 0, ten formatted files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- The 41 full-suite skips require actual-device CUDA on this CPU
  host; no relevant NumPy test was skipped. Added two app modules,
  one infra module, two tests, `docs/measured-wake-observations.md`,
  ADR-0122, and narrow README/architecture/module/evaluation/
  changelog notes; extended the two existing CLIs only with opt-in
  flags. The plan now marks P5.2b1/b2/b and P5.2 complete on their
  declared NumPy v14 artifact scope. **Blockers:** none. **Exact next
  action:** inspect P5.1 bundle and P5.2 sidecar/projection write
  order alongside existing trusted checkpoint stores; split P5.3
  into interrupted-write state/atomic publication and checked-resume
  gates before code. Write a failing test for a partially written
  measured artifact that cannot be mistaken for a completed run,
  then implement the smallest compatible boundary without altering
  fixed v14 bytes or experiment settings.

## 2026-09-29 — P5.3a atomic artifact publication

- Completed task ID: **P5.3a**. P5.3b/P5.3 parent and P5.4–P5.7
  remain open. Previous goal turn was progress: P5.2b1/b2/b and
  P5.2 parent completed on a fully tested, logged measured route.
  This continuation re-read AGENTS.md, the full living plan/latest
  log, ADRs 0120/0122, and inspected the actual checkout. Continued
  on `master` at HEAD
  `5134a17db04d94d4afa26150dfae1939e724a6f4`, preserving all
  prior tracked/untracked work and ignored local artifacts; no commit,
  dependency, algorithm, metric, baseline, or seed change was made.
- Audit found that the P5.1 bundle and P5.2 sidecar/projection writers
  created public directories before all files finished. A missing
  last manifest was rejected by verifiers but could leave a visible
  partial path and block the ID. The trusted local checkpoint store
  already writes checksummed files via temporary file plus
  `os.replace`, but it does not publish whole artifact directories.
  Split P5.3 before coding into atomic publication (a) and checked
  unscored trial-prefix resume (b). The six independent fixed v14
  trials justify a completed trial as b's persisted unit; an
  interrupted trial will restart from its beginning. The parent
  retains checkpoint, hash, run-status, and parity requirements.
- Added `src/infra/atomic_artifact_directory.py`. It validates local
  byte-file names, acquires an exclusive target-specific lock,
  creates a hidden same-parent `.pending.` stage, fsyncs each file
  and state update, compares staged bytes, removes staging-only
  state, and renames the complete directory into public view. A
  caught failure/cancellation leaves the hidden stage with
  `failed`/`canceled` status and completed-file list; the stage says
  `incomplete` while writing. The P5.1 bundle and all three P5.2
  derived writers use it. Public payloads, manifests, protocol IDs,
  and verifiers remain unchanged. Same-volume rename gives atomic
  visibility; crash durability of directory metadata and training
  continuation are not claimed by a.
- Red test collection initially failed because the helper was
  missing. Fault injection now covers a second-file write,
  KeyboardInterrupt, an occupied writer lock, last-manifest writes
  for the bundle, measured sidecar, measured projection, and original
  P5.2a projection, plus a final rename failure after all bundle
  files exist. In every case there is no public completed directory;
  hidden status identifies failure/cancellation. A fully staged
  hidden bundle is rejected by the existing P5.1 verifier because
  its directory ID differs. Retrying from the same validated
  in-memory inputs succeeds; no failed stage was deleted by code.
- One bounded local fixed NumPy v14 measured run,
  `artifacts/runs/p53-atomic-a/`, used Python 3.14.7/NumPy 2.4.6
  on Windows 11 CPU with seeds 47/53 and the three declared arms.
  Its source workspace digest was
  `7bc7e4f765db855318343194f5e6a16996bb1e9da8b07578aa627d1cfd5955a0`,
  resolved config digest
  `22e90b3b3b5312ea52abc956f8ac997a457126bb07a9e03ef0f404079a5a87d0`,
  and completed manifest SHA-256
  `5c86aac5f90d1a740f8f4c6c10ee4cdb968982a5264103f84d4e3d9223fff22d`.
  The original train/outcome hashes repeat
  `174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324`/
  `ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f`;
  the 432-row wake diagnostic hash repeats
  `2d6b6e4b0bc86a3dc292e7f04ad82816ae698dd43b9e63a5fceb41fe85f49f7c`.
  Both eight-file and ten-file projections verify with prior data
  hashes. No pending stage or lock remained after success. Older
  `p51-v14-schema-c/d` completed bundles still verify.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_atomic_artifact_publication.py  # red import before implementation; then 4 tests passed before final rename case
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_atomic_artifact_publication.py tests/test_versioned_v14_run.py tests/test_v14_observation_projection.py tests/test_v14_measured_observations.py  # exit 0, 21 passed after final rename case
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id p53-atomic-a --capture-wake-diagnostics  # exit 0
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --run-measured artifacts/runs/p53-atomic-a  # exit 0
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --run artifacts/runs/p53-atomic-a  # exit 0
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --verify-run artifacts/runs/p53-atomic-a  # exit 0
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --verify-measured-run artifacts/runs/p53-atomic-a  # exit 0
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --verify-run artifacts/runs/p53-atomic-a  # exit 0
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --verify-run artifacts/runs/p51-v14-schema-c  # exit 0
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --verify-run artifacts/runs/p51-v14-schema-d  # exit 0
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0, 1,502 passed, 41 skipped in 290.70 s; final rename test added afterward with focused pass
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0, 253 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/infra/atomic_artifact_directory.py src/infra/versioned_run_files.py src/infra/measured_observation_files.py src/infra/observation_projection_files.py tests/test_atomic_artifact_publication.py  # exit 0, five formatted files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- The 41 full-suite skips require actual-device CUDA on this CPU
  host; no relevant NumPy test was skipped. Added ADR-0123 and
  `docs/atomic-artifact-publication.md`, with narrow README,
  architecture, module, evaluation, manifest, observation, and
  changelog updates. **Blockers:** none. **Exact next action:**
  define a typed format-10 v14 unscored trial-prefix checkpoint and
  run-state identity with source/config/protocol/capture hashes;
  write a red interruption test after trial 1 and trial 3, then
  validate checkpoint role/replay/work/cell facts before any resume
  training or final-role release. Prove fresh/resumed raw, measured,
  and scored bytes match before checking P5.3b/parent.

## 2026-09-29 — P5.3b checked trial-prefix resume

- Completed task IDs: **P5.3b and P5.3 parent**. Re-read repository
  instructions, the plan and latest log, inspected the dirty checkout,
  fixed v14 runner/study/scorer, prior checkpoint store, and P5.3a
  publication path. Continued on `master` at HEAD
  `5134a17db04d94d4afa26150dfae1939e724a6f4`. All prior
  tracked/untracked work and ignored local artifacts were preserved;
  no commit, dependency, algorithm, seed, baseline, trigger, metric,
  or scored protocol change was made.
- Added `src/app/v14_trial_checkpoint.py` for a typed format-10
  completed Cartesian prefix. Each stored trial passes the existing
  role, opportunity, replay, work, capacity, structural-lineage, and
  matched-arm preflight. A partial seed checks its available matched
  arms. The first file-backed test exposed two real details: arrived
  roles use unpicklable mapping proxies, and their deferred source
  references would copy unopened final fields into a checkpoint.
  The format-10 store now has a local mapping-proxy reducer, and the
  app strips `_source` from both phases before persistence. Only a
  complete six-trial prefix reconstructs deterministic source
  references; the original global preflight still runs before final
  release. Original runner and default CLI behavior remain available.
- Added `src/infra/v14_resume_files.py` for an atomically replaced
  hidden run-state file, immutable checksummed checkpoint files and
  exact file SHA-256 references, an OS file lock released on process
  exit, and incomplete/failed/canceled/completed status. The state
  binds the full captured source/runtime environment, fixed config
  and protocol hashes, wake-capture mode, and next seed/arm cell.
  `scripts/run_versioned_v14_bundle.py` now has opt-in `--resumable`
  and `--resume`; the latter validates state and every stored trial
  before another training update. It retrains only the interrupted
  trial or missing suffix, scores only after all six preflight, and
  refuses to overwrite changed public bytes. If a bundle was already
  published before a sidecar failure, resume verifies that bundle
  byte for byte and finishes the missing sidecar. A default fresh
  run also refuses an ID with an unfinished hidden cursor.
- New tests stop after trials 1 and 3, assert no public output before
  six-cell preflight, count only resumed suffix trials, and compare
  fresh/resumed training, outcome, and measured sidecar bytes. They
  reject changed source, config, protocol, capture mode, next cell,
  altered checkpoint bytes, a stale checkpoint reference, and
  rehashed forged role/replay/work records before training. They
  exercise incomplete/failed/canceled status, sidecar interruption,
  and unsafe public overwrite. The first file-backed test was red on
  `mappingproxy` serialization; its local reducer and final-source
  removal resolved that actual boundary issue. No final-role score
  or model selection was used to choose settings.
- One bounded local fixed NumPy v14 measured CLI run is saved at
  `artifacts/runs/p53-resume-cli-a/`, with cursor at
  `artifacts/runs/.p53-resume-cli-a.resume/run-state.json`. It used
  Windows 11 CPU, Python 3.14.7/NumPy 2.4.6, seeds 47/53, and
  periodic/adaptive/no-sleep arms. Its recorded workspace/config/
  protocol SHA-256 values are
  `41b9914abe981f28098ac33f3502e24886b174f5646ab3c2032b99388281c097`,
  `22e90b3b3b5312ea52abc956f8ac997a457126bb07a9e03ef0f404079a5a87d0`,
  and `f9596e86acd7fa5783bacbdafeffadf9ec1682d54b5b14f056d4ffdc9bcb9c12`.
  Completed manifest SHA-256 is
  `e0b9abb6425f727975928dffd1bffdfe0a4c1cb4c8298703cfb378c9fb89d775`.
  Training/outcome/432-row wake metric SHA-256 values repeat the fixed
  `174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324`/
  `ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f`/
  `2d6b6e4b0bc86a3dc292e7f04ad82816ae698dd43b9e63a5fceb41fe85f49f7c`.
  The ten-file measured projection was created and verified. A first
  projection *verify* command failed because that optional derived
  projection had not yet been created; the create command then exited
  zero and verify passed. This was command order, not a producer or
  verifier defect. The completed `--resume` invocation returned the
  same manifest hash without retraining.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_v14_checked_resume.py tests/test_v14_trial_checkpoint.py  # initial file-backed run red on mappingproxy pickle; final exit 0, 24 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_v14_checked_resume.py tests/test_v14_trial_checkpoint.py tests/test_continual_trigger_replay_training_study.py tests/test_continual_trigger_replay_outcomes.py tests/test_versioned_v14_run.py tests/test_v14_measured_observations.py tests/test_atomic_artifact_publication.py  # exit 0, 57 passed
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id p53-resume-cli-a --resumable --capture-wake-diagnostics  # exit 0
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id p53-resume-cli-a --resume --capture-wake-diagnostics  # exit 0, same manifest hash
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --verify-run artifacts/runs/p53-resume-cli-a  # exit 0
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --verify-measured-run artifacts/runs/p53-resume-cli-a  # first exit 1, projection not yet created; later exit 0
  .\.venv\Scripts\python.exe -m scripts.project_v14_observations --run-measured artifacts/runs/p53-resume-cli-a  # exit 0
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0, 1,527 passed, 41 skipped in 495.61 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0, 257 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/app/v14_trial_checkpoint.py src/app/continual_trigger_replay_training_study.py src/infra/v14_resume_files.py src/infra/circadian_checkpoint_files.py scripts/run_versioned_v14_bundle.py tests/test_v14_checked_resume.py tests/test_v14_trial_checkpoint.py  # exit 0, seven formatted files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- The 41 full-suite skips require an actual CUDA device; relevant
  NumPy resume tests ran. Added ADR-0124 and
  `docs/v14-checked-resume.md` plus narrow README, architecture,
  module, protocol, manifest, atomic-publication, and changelog notes.
  P5.3b and parent were checked only after exact-byte, CLI, focused,
  full, and static evidence. Trusted pickle/checksum remains local
  integrity, not signed attestation; an interrupted trial restarts.
  **Plan changes:** P5.3 now complete at its declared v14 scope; P5.4
  becomes active, with its original unknown-key and resolved-config
  acceptance still open. **Blockers:** none. **Exact next action:**
  inspect fixed v14 manifest construction, CLI flags, config digest,
  and existing preset patterns; split P5.4 before code if needed,
  then add a red unknown-override-key test and a saved resolved-config
  assertion without changing the default v14 baseline, seeds, or
  raw result hashes.

## 2026-09-29 — P5.4a/b typed continual configuration

- Completed task IDs: **P5.4a and P5.4b**. P5.4 parent and new P5.4c
  remain unchecked. The preceding P5.3b/P5.3 turn was progress:
  checked trial-prefix resume, atomic artifact publication, exact
  byte parity, and full gates were completed. This continuation
  re-read AGENTS.md, the full living plan/latest log, and ADR-0124,
  then inspected the actual checkout and relevant configuration
  constructors, CLIs, and tests. Continued on `master` at HEAD
  `5134a17db04d94d4afa26150dfae1939e724a6f4`, with prior dirty
  tracked/untracked work and ignored artifacts preserved. No commit,
  dependency, fixed v14 algorithm, seed, baseline, trigger, metric,
  or scored-protocol change was made.
- Inspection found `_validate_manifest` intentionally requires exact
  `fixed_trigger_replay_manifest()` equality, while the v14 bundle
  binds its original raw hashes. Opening that ID to setting overrides
  would weaken the completed matched result. The older
  `run_continual_shift_benchmark` CLI is actually configurable: it
  already has typed `ContinualShiftConfig` subclasses, three named
  presets, many optional flags, and a JSON result containing the
  resolved config. Split P5.4 before code into strict resolver (a)
  and CLI/artifact wiring (b); after implementation, added c to audit
  other active entrypoints because a/b alone cannot prove the broad
  P5.4 parent. This retains every original criterion and leaves the
  fixed v14 route unchanged.
- Added `src/app/continual_experiment_config.py`. Its allowlist covers
  existing data/phase, shared width, noise/transform,
  validation-fraction, and sleep-interval fields. It rejects unknown
  and type-invalid values, Python booleans in numeric fields,
  nonfinite numbers, whole nested circadian config, protocol ID,
  model order, and baseline learning-rate changes. It applies typed
  values to the existing dataclass and reruns `_validate_config`.
  It also builds `continual_resolved_config_v1`, checking that
  explicit override values match the final config and that the
  complete record is finite JSON before any training.
- Extended `scripts/run_continual_shift_benchmark.py` with repeatable
  `--override FIELD=JSON` and optional `--resolved-config`. Overrides
  apply after the named profile and old individual flags; duplicate,
  malformed, and unknown inputs reject before the runner. An explicit
  override requires `--json-result` or `--resolved-config`. Config,
  result, and text output paths must be new/distinct. The exclusive
  resolved record saves preset, seeds, raw explicit overrides, and
  every actual config field; the existing result JSON embeds the same
  config. The default no-override CLI output path and old JSON result
  schema remain intact.
- New tests initially failed collection on the absent resolver.
  They now cover empty/default and all three profile defaults,
  typed successful overrides, unknown/protocol/model-order/baseline
  keys, bool/nonfinite/wrong-type/range failures, duplicate and
  malformed CLI input, missing artifact, occupied path, legacy
  nonfinite flag, and exact resolved-config/result equality in a tiny
  actual run. Existing continual and v14 focused tests passed.
- One bounded local descriptive smoke used the `baseline` preset,
  seed 13, 80 A/B source rows, 0.5 B training fraction, 2+2 epochs,
  width 4, and explicit 2/1 sleep intervals on Windows 11 CPU,
  Python 3.14.7, NumPy 2.4.6. Its artifacts are
  `artifacts/runs/p54-config-smoke-result.json` (SHA-256
  `3cdc1db5cab4334391ac1983780b9ce604532279109d4f1f9393552db7820d18`)
  and `artifacts/runs/p54-config-smoke-resolved.json` (SHA-256
  `de9de7bbc90742f0fc56d5fdef4621e9218fac4f067d4f584996be615bbdc2c0`).
  A direct disk comparison found identical `config` and `seeds` in
  both. The result labels itself `continual_validation_v1` and
  `numpy_shallow_descriptive_v1`; its model scores were not used for
  selection or a matched-circadian claim. This is one local smoke,
  not a new hypothesis test or a sweep.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_experiment_config.py  # initial red ModuleNotFoundError; final exit 0, 21 passed after three test-only profile cases
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_continual_experiment_config.py tests/test_continual_shift_benchmark.py tests/test_continual_sleep_telemetry.py tests/test_versioned_v14_run.py tests/test_v14_checked_resume.py  # exit 0, 81 passed before final three profile cases
  .\.venv\Scripts\python.exe -m scripts.run_continual_shift_benchmark --profile baseline --seeds 13 --sample-count-phase-a 80 --sample-count-phase-b 80 --phase-b-train-fraction 0.5 --override phase_a_epochs=2 --override phase_b_epochs=2 --override hidden_dim=4 --override circadian_sleep_interval_phase_a=2 --override circadian_sleep_interval_phase_b=1 --json-result artifacts/runs/p54-config-smoke-result.json --resolved-config artifacts/runs/p54-config-smoke-resolved.json  # exit 0
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0, 1,545 passed, 41 skipped in 495.35 s; final three test-only profile cases added afterward and passed focused
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0, 259 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/app/continual_experiment_config.py scripts/run_continual_shift_benchmark.py tests/test_continual_experiment_config.py  # exit 0, three formatted files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- The 41 full-suite skips require an actual CUDA device; no relevant
  NumPy configuration or v14 test was skipped. Added ADR-0125 and
  `docs/configured-continual-experiments.md`, with README,
  architecture, app/adapter module, evaluation protocol, and changelog
  notes. **Plan changes:** P5.4a/b checked after their evidence;
  P5.4c added and parent left open for broad active-entrypoint audit,
  with no original acceptance criterion removed. **Blockers:** none.
  **Exact next action:** inventory actively documented experiment
  CLIs and their preset, override, and resolved-config contracts;
  begin with `scripts/run_versioned_v14_bundle.py`, explicitly declare
  and test its sole fixed-v14 preset/unknown-input rejection and
  manifest `resolved_config` identity without opening its frozen
  settings or changing the original raw hashes.

## 2026-09-29 — P5.4c1 fixed-v14 preset and entrypoint audit

- Completed task ID: **P5.4c1**. P5.4c2, P5.4c, and P5.4 parent remain
  unchecked. Read repository AGENTS.md, the full living plan, and the
  latest development log, then inspected the actual checkout. The
  checkout had changed since the preceding handoff: `master` HEAD was
  `f4ae40214b5f136e26be63ecd51349223c56c05d` (the prior P5.1–P5.4b
  work is in that commit), and `git status --short` was clean. The plan's
  reviewed `8793c49...` remains a historical review boundary. This
  session's new tracked/untracked changes and ignored artifacts were
  preserved without a commit or dependency change.
- The fixed v14 bundle already serialized `asdict(study.manifest)` through
  canonical JSON into `resolved_config`; its writer and disk verifier
  bound that to both old raw payloads and `config_sha256`. Added pure
  `src/app/v14_experiment_config.py` with the typed sole ID `fixed-v14`.
  The versioned runner resolves it before source capture, fresh training,
  or checked resume; the CLI accepts optional `--preset fixed-v14` and
  argparse rejects unknown presets and settings. The direct app call
  rejects an unknown ID before training. No v14 setting, protocol, run
  manifest schema, seed, baseline, metric, or raw output serializer changed.
- Tests first failed collection because the resolver did not exist.
  The first implementation run then exposed a test expectation that
  compared dataclass tuples to canonical JSON lists; the assertions now
  compare the exact canonical saved form. One concurrent focused run
  hit the intended v14 source-drift guard when the runner file changed
  during its resume test; rerunning with an untouched checkout passed.
  Focused tests cover unknown CLI/direct inputs before training,
  default and explicit preset runs, full canonical `resolved_config`,
  original raw hashes, and checked resume.
- `docs/configuration-entrypoint-audit.md` inventories documented fixed
  producers, frozen selection/confirmation routes, the completed
  configurable continual route, and the actively documented unmatched
  ResNet multi-seed CLI. That ResNet CLI's parser holds many defaults;
  its JSON saves only a dataset/runtime subset of the typed
  `ResNet50BenchmarkConfig`, omitting inherited model settings. Split
  P5.4c into c1 and new c2; the original P5.4c and parent criteria stay
  open. ADR-0126 explains why the fixed v14 schema and study identity
  were left intact. README, architecture, module docs, manifest guide,
  changelog, and plan/handoff were updated.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_versioned_v14_run.py tests/test_v14_checked_resume.py  # final exit 0: 25 passed in 29.94 s; initial red import, tuple/list assertion, and concurrent source-drift rerun retained above
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0: 1,551 passed, 41 skipped in 482.96 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0: 260 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/app/v14_experiment_config.py scripts/run_versioned_v14_bundle.py tests/test_versioned_v14_run.py  # exit 0: three formatted files
  git -c core.safecrlf=false diff --check  # exit 0
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --run-id p54-fixed-v14-preset --preset fixed-v14  # exit 0, one local fixed six-trial run
  .\.venv\Scripts\python.exe -m scripts.run_versioned_v14_bundle --verify-run artifacts/runs/p54-fixed-v14-preset  # exit 0, completed
  ```

- The ignored local bundle `artifacts/runs/p54-fixed-v14-preset/` has
  manifest SHA-256
  `d27aa6cec2f9e474768ee6c1c1a5156a3d04aa19f6082f64f463fe51db9c87d1`.
  Its training and outcomes retain SHA-256
  `174ee7941c0b1e2489783f43b0b481db11f402c4ea55001888c7998cfb28b324`
  and `ea11fc7cc0ac8113eec2fc5512bb28044b99d0885627813c92aad80cf0f2501f`.
  A direct disk comparison against old default bundle
  `artifacts/runs/p51-v14-schema-c/` found equal `resolved_config`,
  `config_sha256`, training bytes, and outcome bytes. This is an identity
  check, not a new hypothesis test or model selection. The 41 full-suite
  skips require CUDA; no fixed-v14/configuration case was skipped.
- **Plan changes:** P5.4c1 checked only after focused, full, static,
  and actual-artifact evidence; P5.4c2 added for the unmatched ResNet
  CLI. P5.4c and P5.4 remain unchecked. **Blockers:** none.
  **Exact next action:** inspect `scripts/run_multiseed_resnet_benchmark.py`
  parser and `build_base_config`, the typed `ResNet50BenchmarkConfig`,
  and its output tests. Add red no-flag/flag parity, complete resolved
  config/seeds, and pretraining unknown/invalid-input tests. Move the
  existing defaults into a named typed app preset and save the actual
  resolved record without changing historical unmatched status,
  baseline rates, validation-only winner logic, or old result files;
  then run focused/full/static gates and assess P5.4c/P5.4 acceptance.

## 2026-09-29 — P5.4c2 descriptive ResNet multi-seed configuration

- Completed task ID: **P5.4c2**. P5.4c3/c4, P5.4c, and P5.4 parent
  remain unchecked. Re-read AGENTS.md and the current plan/log, then
  inspected actual `master` HEAD
  `f4ae40214b5f136e26be63ecd51349223c56c05d`. Previous P5.4c1
  tracked/untracked edits were present and preserved. HEAD later
  advanced to `68f9dd01a1eb789f7129b421732c7dba1c8c9392`, a
  commit containing prior c1 and initial c2 changes;
  this agent did not run `git commit`. The final CIFAR compatibility
  repair and handoff edits remain unstaged. No dependency changed.
  The reviewed `8793c49...` is still the original plan boundary.
- Before editing, the current no-flag multi-seed CLI built a complete
  `ResNet50BenchmarkConfig` with canonical JSON SHA-256
  `8639c5d9a43fde3f3b84d70921e364b43822d2ab22203213317682338565666f`.
  The representative existing flags (synthetic, classes 10, epochs 2,
  seed 5, no download, trainable backbone, target 0.8) gave
  `ac82520c19c15f63c1589d17c72c4d9cc248147efc4dc7b3197f730934c8d029`.
  Tests now lock both whole-config identities, rather than a subset
  of fields. The existing JSON only saved selected dataset/runtime
  fields, omitting inherited model settings and learning rates.
- Added `src/app/resnet_experiment_config.py` with typed sole preset
  `historical-unmatched`, the original defaults, strict type/finite/range
  checks for the fields already exposed by this CLI, and a complete
  `resnet_multiseed_resolved_config_v1` record. Repeatable JSON overrides
  apply after old flags, reject duplicate/unknown/nested or bad values,
  and cannot change model learning rates. The result records the full
  base config, ordered exact per-seed configs, preset, seeds, exact input
  tokens, and overrides. The CLI checks runner-returned config identity
  before writing a result. Its unmatched status, validation-only winner
  calculation, existing dataset/runtime/summary fields, and CSV names
  are unchanged. Old artifacts were not rewritten.
- New test collection first failed for the missing module, then a patch
  hunk temporarily introduced a syntax error in the script; both were
  repaired before focused and full gates. The mocked one-seed CLI test
  writes temporary JSON/CSV, verifies the exact config sent to the fake
  runner, and checks override precedence. Rejection tests cover
  baseline-rate/unknown keys, duplicate fields, bool for integer,
  nonfinite noise, invalid epoch ranges, and runner config drift before
  publication. Existing output-protection and validation-selection
  tests remain green. After the first full pass, a parity review found
  that CIFAR historically ignores the synthetic-only `train_samples`
  and `test_samples` flags; an explicit red test showed the new
  validator rejected their valid zero values. Validation now permits
  zero for CIFAR and requires positive values for synthetic sources.
  The final focused/static/full gates passed after that repair.
  ADR-0127, the configuration guide, README,
  architecture, module docs, audit, changelog, and plan were updated.
- Commands and outcomes from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_multiseed_resnet_config.py  # initial red ModuleNotFoundError; later passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_multiseed_resnet_config.py tests/test_multiseed_output_protection.py tests/test_resnet50_benchmark.py  # exit 0: 50 passed in 80.74 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # first exit 0: 1,564 passed, 41 skipped in 479.89 s
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_multiseed_resnet_config.py::test_should_keep_unused_cifar_synthetic_sample_flags_valid  # red before repair, green after
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests/test_multiseed_resnet_config.py tests/test_multiseed_output_protection.py  # final exit 0: 19 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # final exit 0: 1,565 passed, 41 skipped in 484.70 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0: 262 source files
  .\.venv\Scripts\python.exe -m ruff format --check src/app/resnet_experiment_config.py scripts/run_multiseed_resnet_benchmark.py tests/test_multiseed_resnet_config.py  # exit 0: three formatted files
  git -c core.safecrlf=false diff --cached --check  # exit 0
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- **Experiment artifacts:** the small mocked result/CSV files lived in
  pytest temporary directories and were not retained as scientific
  evidence. No actual Torch run, large sweep, seed selection, or
  score-driven retuning was launched. The prior ignored fixed-v14
  `artifacts/runs/p54-fixed-v14-preset/` and existing historical
  results remain untouched. The 41 skips require CUDA; no new config
  or multi-seed output test was skipped.
- **Plan changes:** checked c2 after default/flag parity, mocked exact
  record/rejection tests, full suite, and static gates. A subsequent
  README scan found two root wrappers omitted by the first scripts-only
  audit: `predictive_coding_experiment.py` delegates to a configurable
  toy CLI whose JSON omits `ExperimentConfig`, and `resnet50_benchmark.py`
  delegates to a configurable single-run Torch CLI that prints only a
  report. Added c3/c4 and kept c/P5.4 parent unchecked without changing
  their acceptance criteria. **Blockers:** none. **Exact next action:**
  capture the root toy CLI's no-flag and representative flagged resolved
  configs under fixed environment defaults; add red tests for named
  typed preset, pretraining input rejection, complete baseline JSON
  config, and ordered indepth seed/noise artifact. Implement the smallest
  app/adapter change and run a tiny local CPU check plus gates before
  starting the root single-run ResNet c4 slice.

## 2026-09-29 — P5.4c3 root toy CLI configuration

- **Completed task ID:** P5.4c3. Re-read `AGENTS.md`, the full plan/current
  handoff, the log, and the actual checkout. `master` remained at
  `68f9dd01a1eb789f7129b421732c7dba1c8c9392`; the prior P5.4c2
  compatibility and handoff edits were unstaged and preserved. The plan's
  reviewed `8793c49...` remains a preparation boundary, not this session's
  code state. No unrelated or ignored file was removed.
- Before editing, mocked CLI capture under `PC_BASE_SEED=7`,
  `PC_DATASET_SIZE=400`, and `PC_EPOCHS=160` gave complete canonical
  config SHA-256 `b9c678b3ded07dad2969412d8f595d75ac5dc25aa987cff86bca80e51ad42801`
  for no flags and `96941126d2990a7a16ce2e3e4084e0357844b48f195dac98e10f192e4db8a19c`
  for the fixed 80-sample/4-epoch/seed-13/deep/noise/reward flag set. The
  80-sample/2-epoch indepth request with ordered seeds `[13,7]` and noise
  `[0.7,1.1]` hashed to
  `097f8551dc083593539903c3a47aa1369cfa3c91b7b81a9af44fc200a2b684f4`.
  New tests assert all three unchanged identities.
- Added `src/app/toy_experiment_config.py`: the `historical-toy` preset
  owns inherited model defaults and environment-backed sample/epoch/seed
  values. Legacy flags retain precedence, with repeatable typed overrides
  afterward only for their existing `ExperimentConfig` fields. Unknown,
  duplicate, wrong-type, nonfinite, range-invalid, occupied-path, and
  unsaved override requests reject before a runner call. The validator
  retains valid legacy behavior for a validation fraction unused by the
  legacy protocol and replay settings unused when replay is disabled.
  `indepth_comparison.py` now exposes the same per-cell config constructor
  to its runner and the artifact builder, in noise-level then seed order.
- The baseline CLI's new JSON embeds `toy_resolved_config_v1` beside the
  old top-level report fields. `--resolved-config` writes the same full
  record for baseline or indepth after success, exclusively. A bounded
  actual 80-sample/one-epoch test asserted the saved config equals the
  exact object passed to training and that the separate artifact matches.
  A mocked indepth test asserted all four ordered trial configs without
  reading scores or choosing a winner. Direct calls to the existing
  two-argument JSON writer retain their old shape. ADR-0128, README,
  architecture, module docs, audit, and changelog describe the boundary.
- **Commands and outcomes** from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_experiment_config.py  # initial red import; later green
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_experiment_config.py tests\test_toy_sleep_telemetry.py tests\test_indepth_comparison.py tests\test_sleep_artifact_audit.py  # final exit 0: 38 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # first exit 0: 1,582 passed, 41 skipped; final exit 0: 1,585 passed, 41 skipped in 453.50 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0: 264 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\app\toy_experiment_config.py src\app\indepth_comparison.py src\adapters\cli.py src\infra\local_result_json.py src\infra\toy_result_files.py tests\test_toy_experiment_config.py  # exit 0: six formatted files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

  The first green focused run found two test assertion mismatches because
  JSON encodes tuple `model_order` as a list; the assertions were corrected
  to compare canonical JSON. Three compatibility/grid cases were added
  after the first full run, then the final focused and full suites passed.
- **Skipped tests and artifacts:** the 41 full-suite skips require CUDA;
  no toy config test was skipped. The actual bounded result and mocked
  indepth artifact lived only in pytest temporary directories. No
  scientific result file, fixed-v14 bundle, historical output, or large
  sweep was rewritten or launched. No Torch training, seed selection,
  baseline-rate tuning, metric change, or score-driven choice occurred.
- **Plan changes and next work:** checked P5.4c3 only, after parity,
  artifact, rejection, and final gates. P5.4c4, P5.4c, and P5.4 remain
  unchecked; P5.5–P5.7 and later tasks remain open. Preliminary read-only
  inspection of `resnet50_benchmark.py` and its adapter found a 110-field
  config assembled in the CLI. Mocked capture gave default config SHA-256
  `f7a4b9664fc73e2260c60a7dcb02929d254f9390723e1093270807364b593775`
  and README flag-set SHA-256
  `2ef3ac40d8cbe27aee7f292cf26d2188e4fc4680608801f614058f608391f349`
  without invoking the Torch runner. **Blockers:** none. **Exact next
  action:** add red P5.4c4 tests locking those two full-config identities,
  rejecting bad settings before the runner, and saving the exact complete
  config in an exclusive artifact. Then implement the smallest compatible
  app preset/adapter change and run focused/full/static gates without a
  Torch sweep before checking c4, c, or P5.4.

## 2026-09-29 — P5.4c4 and P5.4 configuration audit completion

- **Completed task IDs:** P5.4c4, P5.4c, and P5.4 parent. Re-read
  `AGENTS.md`, the plan handoff, current log, and checkout before work.
  `master` stayed at `68f9dd01a1eb789f7129b421732c7dba1c8c9392`;
  the prior unstaged c2/c3 changes and ignored fixed-v14 artifacts were
  preserved. The reviewed `8793c49...` commit remains the preparation
  boundary, not the current checkout.
- Pre-change mocked CLI capture had fixed the complete 110-field no-flag
  SHA-256 as
  `f7a4b9664fc73e2260c60a7dcb02929d254f9390723e1093270807364b593775`
  and the README CIFAR-100 flag-set SHA-256 as
  `2ef3ac40d8cbe27aee7f292cf26d2188e4fc4680608801f614058f608391f349`.
  Tests now lock both and explicit preset parity. The parser's 110 action
  destinations were mapped to the 110 dataclass fields before editing;
  92 scalar and six boolean parser-held defaults moved to the typed
  `historical-single-unmatched` preset, with 98 leftover default
  placeholders removed. The old `--classes` dataset-dependent omission
  and `--target-accuracy -1` sentinel remain.
- Added `src/app/single_resnet_experiment_config.py` for strict finite
  scalar validation, existing-field typed overrides, and a complete
  `resnet_single_resolved_config_v1` record. The root adapter checks
  unknown, duplicate, malformed, type/range-invalid, nonfinite, and
  occupied/same-path inputs before `run_resnet50_benchmark`. Existing
  broad flags and stdout formatting remain. A compatibility case preserved
  CIFAR's ignored synthetic sample flags (including
  negative values), an empty dataset root, and zero backprop learning
  rate; the route remains descriptive/unmatched. The completed result
  JSON retains its report fields and embeds the same record saved by
  `--resolved-config`. A mocked runner test verifies every requested
  config field and override, with no Torch training.
- A final review found that the fixed CLI model order was outside the
  config dataclass. A red artifact/order-drift test exposed the omission;
  `VISION_DEFAULT_MODEL_ORDER` now drives the runner and record, and
  runner order drift rejects before either artifact is published. The
  record also contains preset, seed, unmatched track, exact input
  tokens, and overrides. ADR-0129, README, architecture, module docs,
  audit, and changelog describe the change.
- **Commands and outcomes** from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_single_resnet_config.py  # initial red missing-module collection, then 17 passed; final 18 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_single_resnet_config.py tests\test_resnet50_benchmark.py tests\test_sleep_retry_runners.py tests\test_sleep_event_accounting.py tests\test_multiseed_resnet_config.py tests\test_multiseed_output_protection.py  # exit 0: 104 passed before order addition
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_single_resnet_config.py::test_should_save_complete_exact_config_and_descriptive_result tests\test_single_resnet_config.py::test_should_reject_runner_order_drift_before_artifact_publication  # red 2, then repaired
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_single_resnet_config.py tests\test_resnet50_benchmark.py tests\test_sleep_retry_runners.py tests\test_sleep_event_accounting.py  # final exit 0: 87 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # first exit 0: 1,602 passed, 41 skipped; final exit 0: 1,603 passed, 41 skipped in 362.00 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0: 266 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\app\resnet50_benchmark.py src\app\single_resnet_experiment_config.py src\adapters\resnet_benchmark_cli.py tests\test_single_resnet_config.py  # exit 0: four formatted files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- **Skipped tests and experiment artifacts:** 41 full-suite skips
  require CUDA; no new single-run config test was skipped. Mocked JSON
  result/config files were in pytest temporary directories and were
  not retained as scientific outcomes. No real Torch benchmark, large
  sweep, baseline/seed/metric tuning, test-informed choice, fixed-v14
  rerun, or rewrite of a historical result occurred.
- **Plan changes:** checked c4, then c and P5.4 only after the active
  CLI audit covered both root wrappers, fixed studies stayed closed,
  exact defaults and artifacts were verified, and final gates passed.
  P5.5–P5.7 and later tasks remain unchecked. **Blockers:** none.
  **Exact next action:** inspect current runner work counters,
  replay/capacity limits, fixed-v14 manifests, and sweep entrypoints
  for P5.5. Split the broad budget task where code inspection justifies
  it, add red tests for a small route with maximum updates, wall time,
  replay examples, capacity/memory bounds, explicit stop reasons, and
  prelaunch sweep-size estimation. Implement one bounded increment
  before any large cross-product experiment.

## 2026-09-29 — P5.5a legacy policy sweep preflight

- **Completed task:** P5.5a only. P5.5 parent and new b–d subtasks
  remain unchecked. Checkout `master` at
  `68f9dd01a1eb789f7129b421732c7dba1c8c9392`; earlier P5.4 work
  and this increment remain unstaged. Preserved prior changes and the
  ignored fixed-v14 artifact.
- **Inspection and plan change:** Re-read AGENTS.md, the plan, and the
  current log, then checked the fixed v14 manifest, runner work counters,
  synthetic DataLoader, and both legacy tuning scripts. V14 already
  fixes replay retention at eight examples/192 bytes and width at 4–32.
  The policy script has 18 candidates, one seed, 20 epochs, 2,500 rows,
  and batch 64: 14,400 possible wake training updates. Pareto has
  10+12+12 candidates × three seeds under the same dimensions: 81,600
  possible updates. Both scripts previously reached Torch/data setup
  before any size preflight. Split P5.5 into policy preflight (a), Pareto
  preflight (b), runtime update/time stops (c), and replay/capacity/memory
  limits (d). A launch estimate does not meet the runtime criteria.
- **Implementation:** Added pure
  `src/app/sweep_work_estimate.py` with strict positive synthetic config
  and seed checks, short-final-batch arithmetic, example exposures, and
  a planned-update limit predicate. The existing policy script now
  resolves its actual candidate list and rejects non-circadian overrides
  that could diverge from its shared loader, prints the estimate, and
  refuses work over an automatic 1,000-update ceiling before
  `require_torch`, weights, or dataset construction. `--estimate-only`
  prints without launching; a higher `--max-planned-training-updates`
  is explicit. New result JSON records estimate and launch ceiling.
  README, architecture/module docs, changelog, ADR-0130, and the plan
  describe the launch-only scope. Candidate settings, seed, guard and
  validation roles, selection metric, and historical files were not changed.
- **Commands and outcomes** from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_sweep_work_estimate.py  # initial red import: missing module, exit 1
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_sweep_work_estimate.py::test_should_reject_candidate_that_changes_shared_loader_before_torch tests\test_tuning_selection.py::test_policy_sweep_passes_training_guard_and_outer_validation_only  # red 1 failure/1 pass, then repaired
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_sweep_work_estimate.py tests\test_tuning_selection.py  # exit 0: 24 passed
  .\.venv\Scripts\python.exe scripts\run_circadian_policy_sweep.py --estimate-only  # exit 0: 18 candidates, 14,400 updates, 900,000 row exposures; no output file
  .\.venv\Scripts\python.exe scripts\run_circadian_policy_sweep.py  # expected exit 1: estimated 14,400 exceeds launch ceiling 1,000; no output file
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0: 1,619 passed, 41 skipped in 388.10 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0: 268 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\app\sweep_work_estimate.py scripts\run_circadian_policy_sweep.py tests\test_sweep_work_estimate.py  # exit 0: three formatted files after formatting
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- **Skipped tests and artifacts:** The 41 full-suite skips require CUDA;
  no new P5.5a test was skipped. The estimate-only and refusal commands
  opened no Torch runtime or dataset and launched no sweep; the full suite
  still ran its existing CPU Torch fixtures. No new scientific output,
  changed baseline, selected seed/metric, fixed-v14 rerun, or historical
  artifact rewrite occurred. The CLI paths created no output file; pytest
  used temporary paths.
- **Blockers:** none for the next increment. Runtime stop reasons and
  update/wall/replay/capacity/memory enforcement remain open by design.
  **Exact next action:** begin P5.5b by extracting Pareto's actual
  10/12/12 candidate lists into side-effect-free builders. Add red tests
  asserting 34 candidates × three seeds = 81,600 planned updates and
  pre-Torch refusal, then reuse the estimator for `--estimate-only` and
  an explicit launch ceiling. Do not launch the Pareto cross-product.

## 2026-09-29 — P5.5b multi-seed Pareto sweep preflight

- **Completed task:** P5.5b only. P5.5 parent and c/d remain unchecked.
  Re-read AGENTS.md, the plan, latest log, ADR-0130, and current script;
  checkout remains `master` at
  `68f9dd01a1eb789f7129b421732c7dba1c8c9392`. Earlier unstaged
  changes and ignored fixed-v14 artifacts were preserved.
- **Pre-change evidence and implementation:** The script held candidate
  lists inside its three training functions and initialized Torch before
  enumerating them. Before editing, parsed the literal lists and fixed
  ordered JSON SHA-256 values: backprop 10 cells
  `425e85d723384f3c1fbfbd43f7a1364cba1d29252c75f9741eb4651c24f62588`,
  predictive 12 cells
  `50858f59efa6b96077a589230ffd0f0b5ca4893a2cbdf14fc439fa1fa7d29356`,
  and circadian 12 cells
  `701e8f5874b7ef9b4921c36bf4f8208feb47bab321ecabf199d9edec21f6633a`.
  Extracted side-effect-free builders in
  `scripts/run_pareto_hard_tuning.py`; tests pin all three hashes. Main
  now estimates from those exact lists and the existing seeds (7, 13,
  29), passes the same lists to the unchanged training calls, and
  rejects wrong-family candidate fields before Torch. `--estimate-only`
  prints 34 candidates/102 trials, 81,600 possible wake optimizer
  updates and 5,100,000 row exposures without launch. An unqualified
  command refuses at the 1,000-update ceiling; a sufficient explicit
  `--max-planned-training-updates` reaches a mocked launch boundary.
  New completed JSON records the prelaunch estimate and ceiling; a
  mocked result confirms exact candidate counts/seeds, validation-only
  selection, and pending final test. README, app module docs,
  changelog, and ADR-0131 were updated.
- **Commands and outcomes** from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_sweep_work_estimate.py  # red exit 1: 7 expected missing-builder/preflight failures, 16 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_sweep_work_estimate.py tests\test_tuning_selection.py  # exit 0: 31 passed after implementation, final 32 passed after mocked artifact case
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_sweep_work_estimate.py::test_should_pass_preflighted_pareto_candidates_and_save_estimate  # first exit 1 on a mock keyword typo; corrected, final focused suite passed
  .\.venv\Scripts\python.exe scripts\run_pareto_hard_tuning.py --estimate-only  # exit 0: 34/102, 81,600 updates, 5,100,000 exposures; no output file
  .\.venv\Scripts\python.exe scripts\run_pareto_hard_tuning.py  # expected exit 1: planned 81,600 exceeds automatic 1,000; no output file
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0: 1,627 passed, 41 skipped in 366.62 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0: 268 source files
  .\.venv\Scripts\python.exe -m ruff format --check scripts\run_pareto_hard_tuning.py tests\test_sweep_work_estimate.py  # exit 0: two formatted files after formatting
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- **Skipped tests and artifacts:** The 41 full-suite skips require
  CUDA; no new P5.5b test was skipped. The direct estimate-only and
  refusal commands opened no Torch runtime or datasets. Existing CPU
  Torch tests still ran in the full suite. Only a mocked JSON result in
  a pytest temporary directory was written; no Pareto sweep, new
  scientific outcome, baseline/seed/metric tuning, fixed-v14 rerun,
  or historical output rewrite occurred.
- **Plan changes and blockers:** checked b after exact candidates,
  pre-resource refusal, CLI estimate, mocked artifact, full/static gates,
  and documentation passed. No blocker for c. P5.5 parent, c, and d
  retain runtime maximum updates, wall time, replay examples,
  capacity/memory, and explicit stop-reason criteria. **Exact next
  action:** inspect `src/app/experiment_runner.py` and
  `src/app/toy_checkpoint.py` at the per-model wake update and final-test
  release boundaries; add deterministic-clock red tests for an opt-in
  maximum-update/wall-time stop before the next update with an explicit
  incomplete reason and checked resume or precise non-resumability.

## 2026-09-29 — P5.5c1 checked toy runtime limits

- **Completed task:** P5.5c1 only. P5.5c2, P5.5c parent, P5.5d, and
  P5.5 parent remain unchecked. Re-read AGENTS.md, the plan and latest
  log, then inspected the current toy update, trusted checkpoint, CLI,
  and final-release boundaries. Checkout remains `master` at
  `68f9dd01a1eb789f7129b421732c7dba1c8c9392`; earlier unstaged
  work and ignored fixed-v14 artifacts were preserved.
- **Plan amendment before implementation:** The existing
  `_train_toy_models` saves a checked cursor after intermediate model
  updates and before/after sleep. `run_experiment` scores final-test
  arrays only after that helper returns. The CLI separately owns full
  resolved-config and completed-result writers. Split P5.5c into app
  stop/resume contract c1 and CLI completed/incomplete/error lifecycle
  artifact c2; the original runtime, stop-reason, and no-leakage criteria
  remain. ADR-0132 records why execution limits stay outside the
  scientific `ExperimentConfig` and old checkpoint identity.
- **Implementation:** Added `src/app/toy_execution_budget.py` with
  strictly validated optional total wake-update and per-invocation
  finite wall-time ceilings, an injectable monotonic clock, and a typed
  `incomplete` stop with exact reason, committed updates, elapsed time,
  and last successfully saved cursor or explicit non-resumability.
  `src/app/experiment_runner.py` checks before each complete model
  update, before sleep, and before final release. On resume, the existing
  checkpoint validator runs first and the three saved loss lengths
  supply the total update count. An exact full-run cap completes;
  stopped runs neither return `ExperimentResult` nor read final labels.
  The first focused run exposed a strict old `_train_toy_models` keyword
  boundary, so unbudgeted calls retain exactly their old arguments.
  README, architecture, app module docs, changelog, and ADR-0132 describe
  the bounded scope and soft wall-time boundary.
- **Commands and outcomes** from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_execution_budget.py  # initial red missing-module collection, exit 1
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_execution_budget.py tests\test_toy_checkpoint_resume.py tests\test_experiment_runner.py  # first 31 pass/1 strict-default-call failure, then 32 passed; after reversed order 33 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_execution_budget.py tests\test_toy_checkpoint_resume.py tests\test_experiment_runner.py tests\test_toy_experiment_config.py tests\test_toy_sleep_telemetry.py  # final exit 0: 68 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0: 1,648 passed, 41 skipped in 378.77 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0: 270 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\app\toy_execution_budget.py src\app\experiment_runner.py tests\test_toy_execution_budget.py  # exit 0: three formatted files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- **Skipped tests and artifacts:** All 41 full-suite skips are CUDA-only;
  no new budget test was skipped. The small real toy runs used pytest
  temporary trusted checkpoints; no retained scientific result or new
  Torch experiment, sweep, seed/metric/baseline tuning, fixed-v14 rerun,
  or historical file rewrite occurred. Update ceilings count completed
  per-model wake calls, not replay/sleep work; wall time is checked at
  boundaries and a single operation may overrun before the next check.
- **Blockers:** none for c2. **Exact next action:** inspect the toy CLI
  path preflight and `local_result_json`/trusted checkpoint stores, then
  add red opt-in CLI tests for budget flags, exclusive lifecycle artifact
  with `completed`/`incomplete`/`error` reason and actual work, checked
  resume, no completed JSON on limit/error, and no final-role access
  before completion. Implement that adapter/infra increment before
  checking P5.5c; keep fixed v14 and default toy stdout/output stable.

## 2026-09-29 — P5.5c2 and P5.5c toy CLI lifecycle

- **Completed task IDs:** P5.5c2 and P5.5c parent. P5.5d and P5.5 parent
  remain unchecked. Re-read `AGENTS.md`, the plan, the latest log, ADR-0132,
  and the current checkout before work. Branch `master` stayed at
  `68f9dd01a1eb789f7129b421732c7dba1c8c9392`; all earlier unstaged
  user/session changes and ignored fixed-v14 artifacts were preserved.
- **Implementation and decision:** Added opt-in toy baseline CLI flags
  `--max-training-updates`, `--max-wall-seconds`, `--run-state`, `--checkpoint`,
  and `--resume` without changing the no-budget runner call, scientific
  `ExperimentConfig`, baseline settings, protocol, or output schema. The
  `src/adapters/toy_budget_cli.py` boundary claims distinct new paths and
  records complete original resolved config plus every attempted budget and
  input token. `src/infra/toy_run_state_files.py` exclusively creates the
  versioned state, holds a sidecar lock, and atomically replaces only its
  observed bytes. Fresh checkpoint names are reserved before state claim;
  a failed claim cleans up only the exact owned reservation. Checked resume
  requires an incomplete/error state, identical scientific config/artifact
  paths, and the exact trusted checkpoint byte SHA-256, cursor, and durable
  update count. A config artifact left by a failed result publication is
  reused only when its content matches the state. `ToyExecutionProgress`
  exposes committed update count even if a later checkpoint write fails;
  error records distinguish observed work from durable work. An unreadable
  checkpoint records a null durable count and cannot resume. Results are
  written only after full final scoring; budget stops exit 3, while other
  exceptions retain their type after recording `error`. ADR-0133, README,
  architecture, module docs, changelog, and the plan describe these choices.
- **Observed local CLI smoke:** `data/p55c2-cli-smoke/` is ignored and retained.
  With `--samples 80 --epochs 1 --sleep-interval 0`, a fresh invocation
  using `--max-training-updates 2` exited 3: state `incomplete`, reason
  `max_training_updates`, observed/durable two updates, no result. A second
  process with identical scientific/artifact args, `--resume`, and limit
  three exited 0: state `completed`, observed/durable three updates, result
  and resolved config present. It was a tiny descriptive toy run, not a new
  selected scientific outcome; its scores were not used to tune anything.
- **Commands and outcomes** from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_cli_run_state.py  # red exit 1: 9 expected failures, 2 passes (missing flags/state)
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_cli_run_state.py tests\test_toy_execution_budget.py tests\test_toy_experiment_config.py  # interim exit 1: Windows lock-close and JSON tuple/list resume issues; repaired to 52 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_cli_run_state.py tests\test_toy_execution_budget.py tests\test_toy_experiment_config.py tests\test_toy_checkpoint_resume.py  # final exit 0: 68 passed
  $artifactRoot = Join-Path (Get-Location) 'data/p55c2-cli-smoke'
  New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
  $common = @('--samples','80','--epochs','1','--sleep-interval','0','--run-state',(Join-Path $artifactRoot 'state.json'),'--checkpoint',(Join-Path $artifactRoot 'checkpoint.bin'),'--json-result',(Join-Path $artifactRoot 'result.json'),'--resolved-config',(Join-Path $artifactRoot 'config.json'))
  .\.venv\Scripts\python.exe predictive_coding_experiment.py @common --max-training-updates 2  # direct process exit 3, incomplete/no result; @common defined with the artifact paths above
  .\.venv\Scripts\python.exe predictive_coding_experiment.py @common --resume --max-training-updates 3  # direct process exit 0, completed/result present
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # pre-parent-path refinement exit 0: 1,664 passed/41 skipped; final exit 0: 1,666 passed/41 skipped in 392.16 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0: 273 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\adapters\cli.py src\adapters\toy_budget_cli.py src\app\toy_execution_budget.py src\app\experiment_runner.py src\infra\toy_run_state_files.py tests\test_toy_cli_run_state.py  # exit 0 after formatting: six files
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- **Skipped tests and artifacts:** The 41 full-suite skips require CUDA;
  no new CLI lifecycle test was skipped. Pytest used disposable temporary
  trusted checkpoints/results. The direct ignored smoke artifacts above
  are the only retained new run files. No Torch benchmark, large sweep,
  fixed-v14 rerun, baseline/seed/metric tuning, historical result rewrite,
  or external service was used. Wall time remains checked between complete
  operations, so one update/sleep/final score can pass the ceiling before
  another check. A killed process can leave `running` and a stale lock;
  the CLI refuses automatic resume until manually audited.
- **Plan changes and blockers:** Checked c2 and then c only after real
  stop/resume, lifecycle/error and isolation tests, and final full/static
  gates. P5.5d still owns replay-example, capacity, and measured-memory
  limits; P5.5 parent stays open. No blocker for the next increment.
  **Exact next action:** inspect `src/core/circadian_predictive_coding.py`
  replay retention and potential sleep work, its structural width caps,
  `src/app/experiment_runner.py` sleep boundary, and existing RSS sampler.
  Split P5.5d before coding if the mechanisms require separate acceptance
  gates. Add red toy runner/CLI tests for strict run-level replay-example
  and capacity refusal with final-test labels sealed, then implement the
  smallest bounded route; leave measured memory open until evidenced.

## 2026-09-29 — P5.5d1 exact toy replay-example ceiling

- **Completed task IDs:** P5.5d1. P5.5d2/d3 and their P5.5d/P5.5 parents
  remain unchecked. Re-read `AGENTS.md`, the full living plan and latest
  development log; inspected `master` at
  `68f9dd01a1eb789f7129b421732c7dba1c8c9392` and the dirty checkout.
  Earlier unstaged work and ignored fixed-v14 artifacts were preserved.
- **Inspection and decision:** Toy replay uses retained whole-batch snapshots;
  selected batch length, not `replay_steps` or retained row count, is the
  exact work. A rejected sleep needs to stop before structural mutation.
  Split P5.5d into d1 exact total replay exposure, d2 transient hidden width,
  and d3 observed process RSS. These have different preflight/measurement
  boundaries; all original parent acceptance remains open. ADR-0134 records
  the replay decision and alternatives.
- **Implementation:** An opt-in `max_replay_examples` stays outside the
  scientific config. The NumPy core selects the actual replay batches once
  before mutation and raises `SleepReplayLimitExceeded` if their total rows
  exceed the caller's remaining allowance. The toy app counts only applied
  telemetry, restores the cumulative count from validated checkpoint sleep
  history, and raises a distinct incomplete stop at `before_sleep`. It
  rejects a resumed cap below already durable replay work. The budgeted
  baseline CLI records observed and durable replay work, cap, reason, and
  checked identity, while allowing older v1 states without the additive
  replay fields to resume against their original checkpoint hash/cursor.
  Checkpoint validation now rejects malformed replay usage before restoration.
  Unbudgeted core/app call shapes and fixed-v14 scientific identity remain.
- **Acceptance evidence:** A 52-row selected training batch stops at cap 0,
  completes at the exact cap 52, and two sleeps complete at 104; stopping
  before sleep 2 with cap 52 then resuming at 104 reproduces unbounded toy
  scores, loss histories, and replay work. Direct core tests cover 2+3-row
  retained batches, priority selection of a nonlatest 3-row batch,
  pre-mutation rejection, zero/invalid caps, component replay off, disabled
  sleep, and legacy zero-structural skip. A sealed final role is never read
  on the stop. A forged negative checkpoint replay count rejects before new
  work; a lowered resume cap leaves the durable file untouched. After an
  injected post-replay checkpoint failure the CLI records 52 observed and
  zero durable examples, then resumes to an exact 52 total. Older v1 run
  state resumes. No baseline, seed, metric, or outcome was selected.
- **Direct local process artifact:** `data/p55d1-cli-smoke/` is ignored and
  retained. With 80 samples, two epochs, sleep interval 1, replay steps 1,
  and replay memory 2, a fresh CLI process at cap 0 exited 3 with
  `incomplete/max_replay_examples`, observed/durable replay 0, a checked
  `before_sleep` checkpoint, and no result. A second process using identical
  scientific/artifact args plus `--resume --max-replay-examples 104` exited
  0 with `completed`, 104 observed/durable examples, and a result whose
  two sleep events sum to 104. This was a tiny budget smoke, not a new
  comparative experiment.
- **Commands and outcomes** from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_replay_budget.py tests\test_toy_cli_run_state.py::test_should_record_replay_limit_and_resume_same_cli_result  # red exit 1: missing SleepReplayLimitExceeded
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_replay_budget.py tests\test_toy_cli_run_state.py::test_should_record_replay_limit_and_resume_same_cli_result  # green: 11 passed after correcting assumed train rows 64 -> observed 52
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_replay_budget.py tests\test_toy_cli_run_state.py tests\test_toy_execution_budget.py tests\test_toy_checkpoint_resume.py tests\test_toy_sleep_telemetry.py tests\test_replay_side_effect_audit.py tests\test_replay_side_effect_policy.py  # existing default-call regression first failed; repaired; final 83 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_cli_run_state.py::test_should_distinguish_observed_replay_from_durable_after_sleep_save_error tests\test_toy_replay_budget.py  # 15 passed
  $artifactRoot = Join-Path (Get-Location) 'data/p55d1-cli-smoke'
  $common = @('--samples','80','--epochs','2','--sleep-interval','1','--replay-steps','1','--replay-memory-size','2','--run-state',(Join-Path $artifactRoot 'state.json'),'--checkpoint',(Join-Path $artifactRoot 'checkpoint.bin'),'--json-result',(Join-Path $artifactRoot 'result.json'),'--resolved-config',(Join-Path $artifactRoot 'config.json'))
  .\.venv\Scripts\python.exe predictive_coding_experiment.py @common --max-replay-examples 0  # exit 3: incomplete, no result
  .\.venv\Scripts\python.exe predictive_coding_experiment.py @common --resume --max-replay-examples 104  # exit 0: completed, 104 applied examples
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # final exit 0: 1,684 passed, 41 skipped in 382.83 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0: 274 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\core\circadian_predictive_coding.py src\app\toy_checkpoint.py src\app\toy_execution_budget.py src\app\experiment_runner.py src\adapters\cli.py src\adapters\toy_budget_cli.py tests\test_toy_replay_budget.py tests\test_toy_cli_run_state.py  # exit 0
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- **Skipped tests and experiment scope:** The 41 full-suite skips require
  CUDA. No d1 test was skipped. No Torch benchmark, large sweep, fixed-v14
  rerun, baseline/seed/metric tuning, historical file rewrite, or external
  service ran. The replay ceiling counts completed example presentations;
  it does not bound stored replay memory, transient topology, or process RSS.
- **Plan changes and blockers:** Checked d1 only after final focused/full
  CPU and static gates and direct CLI stop/resume. Kept d2/d3 and d/P5.5
  unchecked; no blocker for the next increment. **Exact next action:** add
  red toy core/runner/CLI tests for an opt-in transient hidden-width cap:
  reject initial/resumed over-cap width and actual proposed split growth
  before mutation, preserve the checked cursor/final-role seal, show
  exact-cap completion and raised-cap resume, and record observed/durable
  width in CLI state. Implement the pre-mutation gate without changing
  scientific config, split/prune decisions, default toy behavior, or fixed
  v14. Leave measured RSS for d3.

## 2026-09-29 — P5.5d2 transient toy hidden-width ceiling

- **Completed task IDs:** P5.5d2. P5.5d3 and P5.5d/P5.5 parents stay
  unchecked. The previous goal turn made verified progress on d1; this turn
  re-read `AGENTS.md`, the plan handoff/acceptance, current log, and the
  actual dirty checkout. Branch `master` remained at
  `68f9dd01a1eb789f7129b421732c7dba1c8c9392`; earlier unstaged work
  and ignored fixed-v14 artifacts were preserved.
- **Inspection and decision:** Existing intrinsic NumPy `max_hidden_dim`
  constrains model selection but is scientific configuration, not a new run
  ceiling. Sleep splits before it prunes. A small fixed fixture with
  `--split-threshold 0` selected two splits and two prunes, entering and
  finishing its first sleep at width 12 while transiently reaching 14.
  Checking only final width or clamping split count would miss/change the
  actual decision. Final external circadian proposals can also grow width.
  ADR-0135 fixes an opt-in absolute cap on the adaptive circadian layer,
  pre-mutation selected-width checks, and a historical peak restored from
  trusted checkpoint sleep facts. This is separate from replay retention
  and measured process RSS.
- **Implementation:** Added positive `max_hidden_width` to the execution
  budget and root toy baseline CLI, with typed `max_hidden_width` incomplete
  stops. The core checks initial/current width and `current + selected
  splits` after existing decisions but before sleep or final proposal
  mutation. The app checks initial width before any wake update, restores
  current/peak width from the validated checkpoint snapshot and sleep
  history, counts actual transient peaks, and stops at the checked
  `before_sleep` cursor when proposed sleep growth exceeds the cap. The
  lifecycle JSON separates observed versus durable current/peak width and
  labels a rejected proposal separately. Older version-1 states without
  additive width fields still resume only against their original checked
  checkpoint identity. Unbudgeted model calls keep their original shape.
- **Acceptance evidence:** Direct core sleep at width 5 with one split and
  one prune rejects cap 5 before changing weights/lineage/clocks, although
  the final width would also be 5; cap 6 applies the unchanged proposal.
  A final external proposal similarly stops before its split. The toy
  runner rejects cap below initial width at zero updates and a lower cap
  below restored current or historical transient peak without changing
  checkpoint bytes. Exact-cap completion and raised-cap checked resume
  match unbudgeted scores, loss histories, and structural event work;
  sealed final labels stay unread on a stop. No-growth sleep completes at
  the initial cap. CLI tests cover invalid flag preflight, non-resumable
  initial refusal, completed/incomplete states, and observed peak 14 versus
  durable peak 12 after an injected `after_sleep` checkpoint failure,
  followed by checked resume. A small combined replay/width API call
  completed at width 5 with 104 applied replay examples. These are test
  fixtures, not model selection or a new comparative result.
- **Direct local process artifact:** `data/p55d2-cli-smoke/` is ignored and
  retained. An 80-sample, two-epoch baseline process at interval 1,
  split threshold 0, and cap 12 exited 3: `incomplete/max_hidden_width`,
  observed/current peak 12, rejected proposed width 14, checked
  `before_sleep`, no result. A second process with identical scientific
  and artifact args plus `--resume --max-hidden-width 14` exited 0 with
  completed JSON, current width 10, observed/durable peak 14. The first
  sleep's telemetry records before/final width 12 and two applied splits.
- **Commands and outcomes** from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_width_budget.py tests\test_toy_cli_run_state.py::test_should_record_transient_width_stop_and_checked_resume tests\test_toy_cli_run_state.py::test_should_reject_invalid_width_flag_before_artifacts  # red exit 1: missing HiddenWidthLimitExceeded
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_width_budget.py tests\test_toy_cli_run_state.py tests\test_toy_execution_budget.py tests\test_toy_replay_budget.py  # green: 73 passed after extra error/final-proposal tests
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_width_budget.py tests\test_toy_cli_run_state.py tests\test_toy_execution_budget.py tests\test_toy_replay_budget.py tests\test_toy_checkpoint_resume.py tests\test_toy_sleep_telemetry.py  # final focused: 93 passed
  $artifactRoot = Join-Path (Get-Location) 'data/p55d2-cli-smoke'
  $common = @('--samples','80','--epochs','2','--sleep-interval','1','--split-threshold','0','--run-state',(Join-Path $artifactRoot 'state.json'),'--checkpoint',(Join-Path $artifactRoot 'checkpoint.bin'),'--json-result',(Join-Path $artifactRoot 'result.json'),'--resolved-config',(Join-Path $artifactRoot 'config.json'))
  .\.venv\Scripts\python.exe predictive_coding_experiment.py @common --max-hidden-width 12  # exit 3, incomplete/no result
  .\.venv\Scripts\python.exe predictive_coding_experiment.py @common --resume --max-hidden-width 14  # exit 0, completed/result
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0: 1,701 passed/41 CUDA skips in 385.33 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0: 275 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\core\circadian_predictive_coding.py src\app\toy_checkpoint.py src\app\toy_execution_budget.py src\app\experiment_runner.py src\adapters\cli.py src\adapters\toy_budget_cli.py tests\test_toy_width_budget.py tests\test_toy_cli_run_state.py  # exit 0
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- **Skipped tests and experiment scope:** The 41 full-suite skips need
  CUDA; no d2 test was skipped. No Torch benchmark, large sweep, fixed-v14
  rerun, baseline/seed/metric tuning, historical result rewrite, or
  external service ran. The cap bounds the circadian adaptive hidden
  layer's selected transient width, not baseline widths or process RSS.
- **Plan changes and blockers:** Checked d2 only after the focused/full
  CPU and static gates and direct process smoke. Kept d3 and both parents
  open; no blocker. **Exact next action:** inspect
  `src/shared/process_memory.py` and the toy runner/CLI pre-resource
  boundary. Add red injected-reader tests for a per-invocation absolute
  process-RSS high-water limit checked before resources and at wake,
  sleep, and final boundaries; verify checked resume, truthful CLI
  start/peak/sample work, and an unsupported-host error. Implement the
  observed-memory limit and one tiny real local smoke. Document that
  between-sample peaks may be observed by the sampler but cannot be
  prevented; do not alter fixed v14 or historical baselines.

## 2026-09-29 — P5.5d3 measured toy process RSS and P5.5 closure

- **Completed task IDs:** P5.5d3, P5.5d, P5.5. P5.6 and P5.7 remain
  unchecked. Reconciled the current `master` checkout at
  `68f9dd01a1eb789f7129b421732c7dba1c8c9392` with the plan/log
  handoff and preserved all earlier dirty changes and ignored artifacts.
- **Inspection and decision:** `src/shared/process_memory.py` already
  supplies a Windows/Linux absolute current-process RSS sampler with a
  5 ms background interval. The toy budget session is constructed before
  data, while its existing wake/sleep/final checks provide safe stop
  boundaries. A process-RSS limit therefore has a fresh per-invocation
  baseline and cannot be treated as durable model/checkpoint work. ADR-0136
  records this scope, the soft sampling limit, and why model-parameter
  estimates or baseline subtraction would answer a different question.
- **Implementation:** Added positive `max_process_rss_bytes` and typed
  `max_process_rss_bytes` stop facts to the app budget, plus the root toy
  CLI flag. The sampler starts before toy dataset/model construction,
  explicitly samples at each checked wake/sleep/final boundary, and checks
  its observed high-water after its final sample before a result can be
  published. `ToyExecutionProgress` keeps measured attempt work even on
  a later exception. The CLI's additive `work.process_rss` object records
  absolute current-process scope, PID, start/peak bytes, sample count, and
  interval per attempt. A checked resume starts a fresh segment, leaving
  its training checkpoint identity unchanged. A typed unsupported-host
  failure records `error/process_rss_unavailable` before toy resources.
  Existing scientific config, baseline rules, default unbudgeted output,
  fixed v14 bytes, seeds, and metrics were not changed.
- **Acceptance evidence:** A missing-field red test failed before the
  implementation. Deterministic fake-reader tests cover refusal before
  dataset construction; checked wake, before-sleep, and after-sleep/final
  cursors; observed peak retained after current RSS falls; sealed final
  labels at the before-final stop; result withheld when the exit sample
  exceeds the cap; unsupported-host state; and exact-cap completion on a
  checked CLI resume with a new process segment. The unsupported-host
  artifact initially had only generic `RuntimeError`, so the first full
  suite was stopped at about 61% without a test failure while adding the
  typed state reason. Focused and final full gates then passed.
- **Direct local process artifacts:** Ignored `data/p55d3-cli-smoke/`
  contains `refusal/` and `completion/` state/checkpoint/result paths.
  A one-byte cap refused at zero updates before a checkpoint: start
  40,542,208 bytes, peak 40,652,800 bytes, three samples, no result.
  A separate 1 GiB cap completed one 80-sample epoch: start 40,288,256
  bytes, peak 44,552,192 bytes, 11 samples, three updates, checked
  checkpoint and result. A direct subprocess check confirmed child exit
  code 3 for the refusal (the PowerShell tool wrapper itself reported 1
  for the initial nonzero native command); completion exited 0. These
  are budget smoke runs, not a comparative result or model selection.
- **Commands and outcomes** from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_rss_budget.py  # red: 8 missing-budget-field failures
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_toy_rss_budget.py tests\test_toy_rss_cli.py tests\test_toy_cli_run_state.py tests\test_toy_execution_budget.py tests\test_process_memory.py tests\test_toy_checkpoint_resume.py  # final focused: 71 passed
  .\.venv\Scripts\python.exe predictive_coding_experiment.py --samples 80 --epochs 1 --sleep-interval 0 --max-process-rss-bytes 1 --run-state data\p55d3-cli-smoke\refusal\state.json --checkpoint data\p55d3-cli-smoke\refusal\checkpoint.bin --json-result data\p55d3-cli-smoke\refusal\result.json --resolved-config data\p55d3-cli-smoke\refusal\config.json  # incomplete, no result
  .\.venv\Scripts\python.exe predictive_coding_experiment.py --samples 80 --epochs 1 --sleep-interval 0 --max-process-rss-bytes 1073741824 --run-state data\p55d3-cli-smoke\completion\state.json --checkpoint data\p55d3-cli-smoke\completion\checkpoint.bin --json-result data\p55d3-cli-smoke\completion\result.json --resolved-config data\p55d3-cli-smoke\completion\config.json  # completed, exit 0
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # first attempt interrupted without failure at ~61% to repair unsupported-host artifact; final exit 0: 1,713 passed/41 CUDA skips in 361.93 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0: 277 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\app\toy_execution_budget.py src\app\experiment_runner.py src\adapters\cli.py src\adapters\toy_budget_cli.py tests\test_toy_rss_budget.py tests\test_toy_rss_cli.py tests\test_toy_cli_run_state.py  # exit 0
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- **Skipped tests and experiment scope:** The 41 full-suite skips require
  CUDA; no d3 test was skipped. No Torch benchmark, large sweep, fixed-v14
  rerun, baseline/seed/metric tuning, historical result rewrite, or
  external service ran. The sampler can observe transient peaks between
  checked boundaries but cannot prevent them; a brief unseen peak may be
  missed. Its absolute RSS includes Python, data, and all three toy models.
- **Plan changes and blockers:** Checked d3 only after focused/full/static
  gates and the real local smoke. Checked d/P5.5 after verifying d1/d2
  replay/width, c update/time, and a/b sweep preflights already satisfy
  the original four bound classes and stop/estimate criteria. Kept P5.6,
  P5.7, and deferred P2.6a open. No blocker. **Exact next action:** inspect
  the validated P5.1/P5.2/v14 bundle schemas and existing dashboard/report
  code. Define a small report contract for seed count, spread, failures,
  protocol, commit, and track; add red tests rejecting incomplete or
  unvalidated artifacts and producing deterministic table data from one
  checked local bundle. Then implement that artifact-only increment without
  training or altering fixed v14/historical results.

## 2026-09-29 — P5.6a source-verified v14 report table

- **Completed task IDs:** P5.6a. P5.6b/P5.6 parent, P5.7, and deferred
  P2.6a stay unchecked. Re-read `AGENTS.md`, the plan handoff, latest log,
  actual dirty checkout, validated P5.1/P5.2 file boundaries, and the
  historical `docs/index.html`. Branch `master` remained at
  `68f9dd01a1eb789f7129b421732c7dba1c8c9392`; earlier user/worktree
  changes were preserved.
- **Decision and plan change:** Split P5.6 into source-verified table work
  (a) and chart/dashboard rendering (b), retaining the parent criteria.
  P5.1 verifies exact v14 training/outcome bytes and source roles, while
  P5.2 emits raw rows; the existing dashboard is historical. The new
  table must include all declared cells in fixed order without selecting
  a seed, arm, method, or favorable metric. A completed bundle proves
  zero missing cells inside it but has no external attempt-failure log;
  the report explicitly marks that history unavailable (ADR-0137).
- **Implementation:** `src/app/v14_artifact_report.py` derives deterministic
  `summary.json` and `summary.csv` with each of three arms × three methods,
  the two configured seeds, final balanced score/signed forgetting/A-after-B
  accuracy/B-after-B accuracy mean/minimum/maximum/range, descriptive
  interpretation, protocol, fixed NumPy synthetic continual track, source
  commit/dirty/workspace identity, and narrow failure scope.
  `src/infra/v14_artifact_report_files.py` calls `verify_run_bundle` before
  reading, rechecks source bytes against the manifest, atomically publishes
  an exclusive `summary-report-v1` directory, and verifies exact re-derived
  bytes and source/output hashes. `scripts/build_v14_artifact_report.py`
  exposes create/verify CLI commands. Source v14 payloads and historical
  chart/dashboard files were not modified.
- **Acceptance evidence:** A missing-module red test preceded code. Nine
  new tests cover deterministic all-cell aggregation and observed spread,
  missing/reordered/nonfinite/incomplete records, unavailable Git identity,
  invalid source refusal before writing, changed source SHA, exclusive
  write, and hand-edited report refusal. Existing P5.1/P5.2 boundary tests
  pass with them (focused 24). The final full CPU suite passed 1,722 with
  41 CUDA skips; Ruff, mypy (281 source files), format, and diff checks
  passed. No score-based sorting, baseline change, or metric tuning occurs.
- **Experiment/report artifacts:** Ignored
  `artifacts/runs/p51-v14-schema-d/summary-report-v1/` was created and
  re-verified from the existing checked bundle. It reports seeds 47/53,
  18/18 method cells, nine rows, zero failed cells only within the
  published bundle, external attempts `not_recorded_in_bundle`, outcome
  protocol `continual_trigger_replay_outcomes_v14`, track
  `numpy_synthetic_continual_v14`, and dirty source commit
  `5134a17db04d94d4afa26150dfae1939e724a6f4`. The source training/
  outcome SHA-256 values remain `174ee794...b324`/`ea11fc7c...501f`;
  report JSON/CSV values are `a0a3f13b...9a752`/`e879aa84...2f901`.
  An earlier ignored draft under `p51-v14-schema-c/summary-report-v1`
  became stale when explicit failure/interpretation scope was added; its
  verifier rejects it. Automatic approval review rejected removal of
  that generated directory, so it was left untouched and is not accepted
  evidence. The fresh `p51-v14-schema-d` artifact is the verified result.
- **Commands and outcomes** from the workspace root:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_v14_artifact_report.py  # red collection error: missing app module; final 9 passed
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short tests\test_v14_artifact_report.py tests\test_v14_observation_projection.py tests\test_versioned_v14_run.py  # 24 passed
  .\.venv\Scripts\python.exe -m scripts.build_v14_artifact_report --run artifacts\runs\p51-v14-schema-d  # wrote exclusive derived report
  .\.venv\Scripts\python.exe -m scripts.build_v14_artifact_report --verify-run artifacts\runs\p51-v14-schema-d  # exit 0, source and report re-derived
  .\.venv\Scripts\python.exe -m scripts.build_v14_artifact_report --verify-run artifacts\runs\p51-v14-schema-c  # exit 1, expected stale draft rejection
  .\.venv\Scripts\python.exe -m pytest -o addopts= -q --tb=short  # exit 0: 1,722 passed/41 CUDA skips in 368.08 s
  .\.venv\Scripts\python.exe -m ruff check .  # exit 0
  .\.venv\Scripts\python.exe -m mypy  # exit 0: 281 source files
  .\.venv\Scripts\python.exe -m ruff format --check src\app\v14_artifact_report.py src\infra\v14_artifact_report_files.py scripts\build_v14_artifact_report.py tests\test_v14_artifact_report.py  # exit 0
  git -c core.safecrlf=false diff --check  # exit 0
  ```

- **Skipped tests and experiment scope:** All 41 full-suite skips require
  CUDA; no P5.6a test skipped. The suite includes existing bounded
  fixtures, but this task launched no new comparative training, large
  sweep, Torch benchmark, fixed-v14 rerun, or external service. The
  output is a descriptive table, not evidence that circadian wins.
- **Plan changes and blockers:** Checked P5.6a only after its focused,
  full, static, and checked local artifact gates. Kept P5.6b/P5.6 and
  P5.7 open. No blocker to the next implementation; the stale ignored
  draft is explicitly excluded from evidence and verification rejects it.
  **Exact next action:** inspect the verified
  `p51-v14-schema-d/summary-report-v1` table and historical
  `docs/index.html`; add red tests for a renderer that first verifies
  the report, includes all nine rows plus seed count, observed range,
  failure scope, protocol, original commit, and track, and rejects
  changed source/table/chart bytes. Publish charts/dashboard content in
  a new derived directory without overwriting the historical dashboard
  or training.

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

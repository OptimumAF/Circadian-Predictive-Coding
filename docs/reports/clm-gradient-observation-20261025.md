# R2.1-CLM.G — existing native gradient observation

Issue [#156](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/156),
2026-10-25. Implemented and executed scoped software behavior, single-agent
source/procedure/results **self-review**, not independent acceptance. Full goal
and R2/R5.1/R5.3/G5/R7/release parents remain open. Previous #155 package evidence
is reused for its original scope; that earlier wheel does not contain this change.

## Delivered behavior

The shared native engine exposes its existing `clip_grad_norm_` return through
`train_batch_observed`. Immutable scalar records contain the original loss,
committed update, query rows, optimized gradient-tensor count, pre-clip L2 norm,
original unit clip limit and fixed definition. Paired/catalog/state-only and
disposable probe dispatch preserve their original objectives and optimized sets.
Default diagnostic, model/snapshot/work schemas and all update mathematics remain.

The existing delayed workflow accepts `record_gradient_norm=True` through a thin
TRAIN-only port, retaining the original inbox, authority, receipts, clocks and
budget. Its bounded <=32 scalar list binds every observation to a returned native
commit and original loss receipt. Prior value/list/row/returned-owner replacement,
features/budget corruption and save failure refuse further work and preserve the
trusted paid prefix. A lost native return keeps its physical charges and produces
no invented norm/receipt. No native observation history, tensor reference, extra
forward/backward/norm reduction, budget renewal or new orchestration framework.
Scalar conversion, validation and reporting still cost time/storage.

Files: new `src/core/clm_training_observation.py`; shared native engine and existing
catalog/state-only/probe forwarding; `src/app/clm_workflow_training.py`;
`tests/test_clm_gradient_observation.py`; [API/usage](../clm-gradient-observation.md)
and README. No dependency, CLI, environment variable, training recipe or final-data
access change. Inner modules import no native backend. Existing public CLI wiring
and disposable-probe app reports are unchanged; the native probe port is available
to explicitly budgeted callers.

## Executed evidence

Stage `artifacts/runs/clm-gradient-observation-20261025/` contains the prospective
PROTOCOL, six pre-edit source snapshots, measured literal harness/supervisor,
source delta, XML/stdout/work/parent receipts, statics failures and final passes,
current-source SHA256 pins, rollup and cache-scope supplement. No old scientific
actual/execution/ledger or detached review was read by this increment.

| Run | Cases | Body seconds | Parent seconds | Peak RSS bytes |
| --- | ---: | ---: | ---: | ---: |
| Windows1 | 59 | 20.792842400 | 21.386826500 | 329412608 |
| Linux1 | 59 | 29.735664851 | 43.077659400 | 346140672 |
| Windows2 final | 60 | 14.527061200 | 15.202220800 | 329273344 |
| Linux2 final | 60 | 31.547587932 | 53.961697000 | 346193920 |

All four EXIT0, zero failures/errors/skips/timeouts. Final60 includes unchanged
workflow23 plus three selected catalog/state-only math regressions and34 new
cases. Positive and zero norm, independent Python scalar L2 (fixed f32
rtol1e-6/atol1e-7), exact original clip return, observed/default paired/catalog/
state-only complete native state/loss/work/RNG, and all three disposable-probe
paths pass. Snapshot comparisons include gradients, optimizer, scheduler, modes
and native RNG. Existing before-workflow report is exact against its saved source
using the same engine; it is not a separate before-engine experimental run.

After first59 checks, self-review strengthened explicit positive-norm assertions,
paired disposable-probe parity and ordinary/observed catalog-probe parity. Final
source was then qualified on both platforms; no production or test change followed.

Per-platform **cumulative**, including both runs:320 native TRAIN attempts and322
backwards (two separate direct math-reference backwards),3240 query/1468candidate/
15936logit exposures,103 scheduler-constructor initial steps,1032 head objects,
648 head forwards/4728 projected rows,722 source capture returns totaling3671944
tensor elements plus4009408 caller-RNG elements. Literal width<=4; physical random,
backbone, encoder, tokenization, cached typed-decode, dispatch and Transformers
imports all0. The combined two-platform totals double these counts; they are not
free tests. Native physical work accounting remains distinct from app receipts.

The original per-platform384-update/backward,4608query/2304candidate/27648logit,
2048head/forward,16384projected-row,128initial-step,48Mcopy-element+4MiBother-copy/
65536literal reserve,120sbody/150sparent/1GiBsampledRSS/CPU1 sequential limits hold.
Windows cumulative35.319903600sbody/36.589047300sparent; Linux61.283252783sbody/
97.039356400sparent. RSS sampling starts after Torch import; returned-copy counts
are not total allocations/FLOPs. Torch2.14.0+cpu on both cached environments.
Source signatures and the sole clip call were inspected against saved pre-edit
sources; original losses, parameter choices and atomic commit sequence preserved.
Instrumentation wraps optimizer steps, producing the retained scheduler warning;
this is fixture instrumentation, not an observed production ordering failure.

## Failures, checks and resource reconciliation

Initial Ruff F401 unused test import was removed before fixtures; original
statics-1 retained. statics-2 found the TRAIN-only wrapper incompatible with the
broader inbox protocol annotation. An explicit local `Any` boundary documents the
same train-only behavior without granting fake snapshot/restore methods.
statics-3 found seven new test annotation/heterogeneous-dict typing errors; exact
dict annotations corrected them. statics-4: Ruff, format, default-target mypy and
Linux-target mypy all EXIT0 on eight affected source/test files. No tolerances,
math, native allowances or scientific limits changed to repair typing.

Measured four statics attempts total116.393464500s including failures; one extra
two-file format check has a conservative parent remainder<=0.7871497s. Bounded
stdlib rollup/cache/source-delta work adds0.790632s body, with a scoped whitespace
check also passing, within the120s engineering static allowance. Actual Linux
Ruff/mypy was unavailable and not installed; Linux-target typing ran on Windows.

Stage before rollup148145B, stdout14223B, below8MiB/512KiB. The rollup's whole
repository cache inventory is596739144B, including unrelated inherited caches;
**that global total is not within128MiB**. The task-scoped cache supplement charges
the *entire*109690284B of cache files modified since issue creation (109543424B
mypy plus146860B Ruff), below the original128MiB task reservation. Untouched
inherited files are487048860B. Modification-time attribution is a conservative
current-file reservation, not a measurement of cumulative IO or allocation. No
cache was deleted, moved, refunded or renewed; original global and scoped figures
are both preserved. No original engineering bound was retrospectively increased.

Two read-only path guesses failed (absent prior statics helper; invalid wildcard/
nonexistent CLM path). Ordinary explicit-file reads replaced them; these were not
test or native failures. Some large tool outputs were display-truncated; final
test results were read from compact retained receipts, without rerunning work.

Commands, from repository root:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-gradient-observation-20261025/prepare.py
.venv/Scripts/python.exe -m ruff format src/core/clm_training_observation.py src/adapters/clm_head_learner.py src/adapters/clm_catalog_learner.py src/adapters/clm_state_only_catalog_learner.py src/adapters/clm_catalog_probe.py src/adapters/clm_head_probe.py src/app/clm_workflow_training.py tests/test_clm_gradient_observation.py artifacts/runs/clm-gradient-observation-20261025/prepare.py artifacts/runs/clm-gradient-observation-20261025/focused.py artifacts/runs/clm-gradient-observation-20261025/launch.py
.venv/Scripts/python.exe -m ruff check src/core/clm_training_observation.py src/adapters/clm_head_learner.py src/adapters/clm_catalog_learner.py src/adapters/clm_state_only_catalog_learner.py src/adapters/clm_catalog_probe.py src/adapters/clm_head_probe.py src/app/clm_workflow_training.py tests/test_clm_gradient_observation.py artifacts/runs/clm-gradient-observation-20261025/prepare.py artifacts/runs/clm-gradient-observation-20261025/focused.py artifacts/runs/clm-gradient-observation-20261025/launch.py
.venv/Scripts/python.exe -B artifacts/runs/clm-gradient-observation-20261025/launch.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/clm-gradient-observation-20261025/launch.py linux-1
.venv/Scripts/python.exe -B artifacts/runs/clm-gradient-observation-20261025/statics.py statics-2
.venv/Scripts/python.exe -m ruff format --check src/app/clm_workflow_training.py tests/test_clm_gradient_observation.py
.venv/Scripts/python.exe -B artifacts/runs/clm-gradient-observation-20261025/statics.py statics-3
.venv/Scripts/python.exe -B artifacts/runs/clm-gradient-observation-20261025/statics.py statics-4
.venv/Scripts/python.exe -B artifacts/runs/clm-gradient-observation-20261025/launch.py windows-2
.venv/Scripts/python.exe -B artifacts/runs/clm-gradient-observation-20261025/launch.py linux-2
.venv/Scripts/python.exe -B artifacts/runs/clm-gradient-observation-20261025/rollup.py
git diff --check -- README.md src/core/clm_training_observation.py src/app/clm_workflow_training.py src/adapters/clm_head_learner.py src/adapters/clm_catalog_learner.py src/adapters/clm_state_only_catalog_learner.py src/adapters/clm_catalog_probe.py src/adapters/clm_head_probe.py tests/test_clm_gradient_observation.py docs/clm-gradient-observation.md
```

The statics helper records the exact scoped Ruff/format/mypy commands and outputs.
The launcher records the cached Linux invocation and all limits. No full unchanged
suite, clean clone/install/package rebuild, GPU, remote CI, native pretrained
experiment, browser, independent review or release qualification. These are
deliberate scoped skips; package/UI and broad native gates are not inherited.
No installs/downloads/commits/pushes/deletions/unrelated resets. All historical
plans/log, dirty user work, final isolation, prior negative results and permanent
scientific charges remain unchanged.

## Interpretation and next action

This closes the discarded-scalar software gap. It does not measure gradients in
old runs or explain #151's partial F acquisition and old-task interference.
#153 remains OPEN FAILED-SPENT before model load, #154's manual browser criterion
remains open, and no CPC advantage or generalization is established.

The next useful question is whether frozen CLM features support a simple direct
categorical fit, separating representational fit from warm nonlinear-head update
behavior. This should be a conventional TRAIN-only learnability control through
existing feature/stream boundaries, not another observation/capture sequel or a
renamed retry of #153. Inspection found an existing fixed minimum-norm categorical
fit (`src/core/categorical_linear_fit.py`) capped at296 dimensions; released CLM
features are4096, so it cannot simply be called or silently widened.

**One next implementation action:** design and implement a separately bounded
wide-feature categorical TRAIN-fit control using that existing solver/readout
convention, preserving the old296-width contract, role isolation and original
learner. Start with scalar/literal fixtures and explicit row/width/LAPACK/copy
limits. No native asset load or old-source reconstruction is authorized by this
software handoff. Before a new native question, freeze its fresh owner, complete
training information, fixed metrics, no-final/no-selection rule and total costs;
preserve every earlier failed/spent result. This is a diagnostic control, not
matched compute or deployed retention/recovery by itself.

Live scope/status/evidence/handoff belongs in issue156/tracker18 under AGENTS.md;
historical plans and development log remain frozen rather than duplicating status.

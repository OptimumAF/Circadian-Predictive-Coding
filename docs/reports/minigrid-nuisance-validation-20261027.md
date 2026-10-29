# Separate-source nuisance TRAIN validation

[#235 / R7.5-MG.N](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/235).
The existing controller runner now supports `nuisance_redwall_delay2` in its
Python API. This is executed and verified **software behavior with scripted
learners/worlds**, not a native nuisance experiment. Source, procedure and
results were self-reviewed, not independently accepted. Full repository,
generalization, reliable adaptation and release requirements remain open.

## Behavior delivered

The original grey A3/A4 warmup and four fitting trials are unchanged. The chooser
freezes before collecting one fixed red-wall A3 blue/red TRAIN pair. Original
grey A3 sources and new red A3 sources form a separate validation study. Both
domains have branch3/rotation0/unswapped keys; seed17/max16, nine rows/296features,
seven classes, explicit TRAIN roles and distinct source identities. No F/probe
or final sources are relabelled or accessed.

Five existing policies receive red-blue/grey-blue/red-red/grey-red, label-delay2,
six opportunities and four original update credits. No new policy or learning
algorithm is introduced. New examples and validation endpoints cannot enter
chooser fitting, tuning, promotion or probe/final selection. Original source
episodes, warm model, actors, chooser fit, budgets, clock origins, RSS sampler
and RNG are guarded. A new input domain here does not itself mean a new task.

After **all five** training windows finish, the runner reads a fresh warm-model
baseline in the grey/red A3 domain, then candidate endpoints. It cannot substitute
the grey A3/A4 fitting baseline. Collection and endpoint layout/seed/cue/role
bindings agree; result groups are `old_A3` and `shifted_A3`. The original known
recipe, fitting baseline and CLI keep their behavior.

Readout reuse stays local and bounded to warm plus five complete states, with
the existing full-state/argument/source/paid-evidence checks. Reused nuisance
endpoints have unavailable utility and measures, including baseline reuse when
the optional state cache is disabled. Physical calls and logical reuse are
separate. Fresh baseline time is recorded separately; it is not invented as
cheap per-arm work. Bad layouts or lost reads fail before cache admission.

Collection uses the original cumulative zero-update/zero-replay150s/512MiB owner;
candidate windows retain their own original4-update/90s/512MiB limits. No budget
is renewed. Two collection records and their charges enter the **existing**
study evidence/link store within its unchanged26-paid/34-link bounds. Corruption
or replacement refuses further work. Normal failed closeout recovers the last
paid records rather than silently continuing with edited evidence. Partial
construction/cleanup/returns remain explicit; no unknown endpoint is inferred.

## Source and architecture

- `scripts/minigrid_controller_learning.py`: separate validation source assembly
  after freeze, guarded study domain and matching readout dispatch. A typed
  explicit ACK keyword also removes an inherited `**dict[str,bool]` ambiguity
  against the current callback signature, preserving False-default behavior.
- `scripts/minigrid_controller_validation.py`: fixed recipe, guarded preparation,
  deferred domain baseline, honest reuse and semantic endpoint groups.
- `src/adapters/minigrid_policy_evaluation.py`: exact opt-in nuisance reference
  readout and shared phase-schema check; old admission/defaults remain intact.
- `src/adapters/minigrid_train_collection.py`: original G/A4 teacher/cleanup loop
  factored for reuse, with only the additional fixed red A3 profile and strict
  native-record/packet equality. The original workflow wrapper remains.
- `tests/test_minigrid_nuisance_validation.py`:24 new cases. README, architecture,
  [usage](../minigrid-controller-learning.md) and
  [ADR-0640](../adr/ADR-0640-nuisance-train-validation-separation.md) document the path.

The collector factor avoids a new cycle among script orchestrators. It imports
core/infrastructure, not scripts. No dependency, runner, cache framework, source
loader, learner math, final/probe route or global budget protocol was added.

## Executed evidence

Stage: `artifacts/runs/minigrid-nuisance-validation-20261027/`.
Original dirty sources are in `preedit/`, actual changes in `combined-source.patch`,
scope in `PROTOCOL.md`, final source review/pins in `review-final.json` and final
delta in `combined-source-final.patch` (earlier review/delta retained). No Git mutation or
unrelated source reset occurred.

**46 distinct checks/platform qualify**:24 new cases plus22 existing known-source
validation cases, including the original A4 collector wrapper regression among
the new cases. Final16 affected cases/platform plus6 domain-binding cases pass.

Windows:76 instances,75PASS/1FAIL/0errors/0skips. The first six-instance cohort
failed a test-only assumption that correct acknowledgement implies0optimizer
updates for every policy. Periodic still chooses two fits; acknowledgements
are paid separately. The corrected test verifies applied receipt/update equality
and processed=fitted+acknowledged counts, with2periodic fits/2ACKs. No policy or
acceptance criterion was weakened. That original failure remains in
`win-initial.{json,xml,txt}`. Subsequent39combined,9preservation,16final and6domain cases pass.

Linux:63 instances/63PASS,41combined plus16final and6domain,0errors/failures/skips. These
fixtures exercise actual constructor, bank, stream, chooser, source witness,
budget, readout admission and failure-closeout paths. Scripted success booleans
are not evidence a native policy solves the nuisance task.

Passing boundaries include post-freeze collection, original fitting/source/RNG
preservation, new-source packet order, delayed label consumption, all-training
endpoint release, fresh domain baseline, shared/unshared/ACK variants, wrong
layout refusal, constructor failure and cleanup, corrupted/replaced paid records,
late collection corruption with prefix recovery, changed chooser and RNG refusal,
undeclared recipes/flags before model/IO, and legacy default preservation.
Final domain review also pins recipe to the study domain, cross-checks the
mutable mode against immutable source identity, and validates nuisance layout
bindings again before reducing counts. Changing both mode and its copied pin,
or supplying the fitting A3/A4 games as the nuisance baseline, is refused.

| All attempts | Windows | Linux |
| --- | ---: | ---: |
| Body seconds | 126.9327815999 | 114.8706233390 |
| Parent seconds | 134.6971813997 | 136.7291887002 |
| Sampled peak RSS, bytes | 101,654,528 | 83,472,384 |
| Literal owners / snapshots | 1,804 /38,365 | 1,444 /30,405 |
| Scripted fits / rows | 423 /6,930 | 346 /5,670 |
| Scripted predictions / rows | 1,127 /19,602 | 890 /15,462 |
| Scripted four-game readout calls | 221 | 179 |
| Counterfactual calls | 152 | 124 |
| Fake collection environments | 58 | 43 |
| Conservative scripted teacher-step charges | 453 | 338 |

All240s-body/300s-parent/512MiB and original work caps held. The affected rerun
reserve was prospectively increased16→32 instances/platform because final
source/baseline/RNG checks followed9Windows preservation repeats. The measured
remaining resources justified it; failed work was preserved, no scientific
allowance changed. Amendment receipt:
[comment6433593798](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/235#issuecomment-6433593798).
Source `PROTOCOL.md` retains the original allowance; GitHub holds the amendment.

Native constructors/fit/predict, solver and real environment construction are
patched to refuse. Forbidden entries/imports are0. Genuine learned numeric work,
physics, real teacher execution, model random draws, payload/source downloads,
old scientific raw reads and actual final/probe data access are0. One negative
fixture deliberately changes NumPy RNG, verifies refusal and restores test state.
RSS is sampled after imports, not a process-tree or exhaustive allocation peak.

Initial Windows command:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/minigrid-nuisance-validation-20261027/supervise.py win-initial .venv/Scripts/python.exe -B artifacts/runs/minigrid-nuisance-validation-20261027/qualify.py win-initial
```

Linux uses the same qualifier through WSL Ubuntu/cached Python and CPU1 variables.
All exact commands, selectors, timeouts, terminal outputs, per-case outcomes and
cumulative costs are in `*-parent.json`, XML/logs and `software-costs.json`.
Exclusive/raw receipts must not be overwritten or rerun under the same names.

All eight changed Python/helper files pass final Ruff format/check and
default/Linux-target mypy. Initial mypy's7annotation/typed-kwargs diagnostics are
retained in `statics-initial.json`; explicit list/fixture annotations and direct
ACK keyword fixed them without ignores. Total formatting/static time over all
attempts is recorded in the final postflight receipt, below180s. One inherited111,280,128B cache file was mtime-selected,
below160MiB; this is touched storage, not growth. Exact static commands and
outputs are in `statics-final.json` and `statics-domain.json`. Ordinary focused test entry:
`python -B -m pytest -q tests/test_minigrid_nuisance_validation.py
tests/test_minigrid_controller_validation.py`.

## Limits and next action

No native nuisance experiment, complete-suite, clean-clone, current-wheel/install,
UI, GPU/remote CI, independent review, CPC comparison or scientific release check
ran. G default collection was preserved by source inspection of the moved path;
the original A4 path was also executed with its existing fake producer. No new
scientific results, nuisance benefit, reliability, generalization, recovery,
efficiency or release readiness are established. #214's missing fitted endpoint,
#231/#227 and every earlier failure/negative retain their original meaning.
Existing useful conventional learning, bank retention, R2.1 diagnostics and
matched BP/PC/neutral/active evidence were reused, not reconstructed.

Next precise action: freeze and self-review a small native nuisance protocol
using this existing Python path, original chooser fit, fixed red-wall A3 source,
all five unchanged controls, and explicitly declared ACK/readout-sharing settings.
Budget original fitting/discarded work, new collection, diagnostic predictions,
baseline plus physical/reused endpoint calls, storage, cumulative time and memory
before execution. Read reused utilities as unavailable, keep final/probe data
sealed and preserve the earlier spent comparisons. A failed or negative result
is valid; do not retune the palette, policies, objective, seed or metrics.

Full goal and R2/R4/R5/R7/P7/P9/science/release parents remain open. Historical
plans/log are frozen; live status and handoff are the owning issue and tracker.

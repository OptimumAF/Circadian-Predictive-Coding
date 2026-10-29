# Fresh A4 reference-policy workflow — software qualification

[#201 / R7.2-MG.F](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/201)
adds explicit reference-selected learning to the ordinary fresh A4 command.
Source/procedure/result **self-review**, not independent acceptance. Full goal
and R2/R4/R5/R7/P7/P9/scientific/release parents remain open.

## Working behavior

```powershell
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --context-linear-reference periodic --ready-feedback --describe
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --context-linear-reference periodic --ready-feedback --output fresh-reference.json
```

The checkout route uses the existing fresh TRAIN collector, arrived bank,
AttributionStream and checkpoint evaluator. No new learner, controller, runner,
dependency or tuning option. Supported original policies: no_learning, periodic,
budget_aware and drift. Ready feedback is explicit; the bank is inherent to this
fresh A4 route. Other recipes, source files, prequential/retention/fit flags
refuse before delegation. Describe performs no IO. Output is exclusive, bounded
8 MiB, including failed/partial results with exit2. Defaults keep their fixed-six
behavior; ordinary A/B reference state exports remain empty.

Actual partial unscored prefixes, including unfitted/odd states, now cross the
existing stream/report boundary only for explicit A4 reference mode. Original
snapshot conversion accepts None cue heads and retains native validation. The
workflow requires a complete seven-opportunity window and original FIFO receipt
times, committed rows, pending count, observed decisions and actual checkpoint
states before readouts. It never reconstructs a model from a metric or count.
Window completion is separate from complete training.

Games follow all training and retain four fixed A3/A4 blue/red TRAIN episodes,
seed17/max16/raw K7 firstargmax. Logical tick2/4/6 is separate from actual updates.
An identical complete state can reuse a previously complete readout, labeled
with its original tick. Three bounded phases/cache entries, existing paid-prefix
identity/value checks, no cross-invocation reuse or unlimited success chain.
Recovery is null for reference cues without acquired-old loss; late acquisition
is not recovery. The original default summary remains unchanged.

Original cumulative150s/512MiB zero-update/replay readout owner starts before
preparation; the inner stream keeps its six-update90s owner. Policy is bound
before collection. Changed limits, even equal-valued replacement evidence or
boolean-zero counters, stop more work. Failure reports retain initial limits
and actually observed replacement limits separately; they do not admit or renew
the replacement. Collection errors and paid prior game records remain visible.
No final/probe/heldout/promotion/tuning input reaches these paths.

## Executed checks

Final **74 distinct cases pass on each platform**, no skipped/error/final-failed
cases. Windows final evidence reuses unaffected passing cases and reruns affected
cases after repairs; it is not a claim that the initial full cohort passed.

| Software qualification | Windows cumulative | Cached WSL Linux cumulative |
| --- | ---: | ---: |
| Executed case instances, including original failures/repeats | 95 | 76 |
| Body seconds after imports | 16.2647932 | 36.0369231 |
| Sampled body RSS bytes | 114180096 | 100073472 |
| Literal constructors / snapshots | 471 / 5063 | 367 / 3906 |
| Scripted fits / rows | 150 / 1953 | 121 / 1593 |
| Scripted predictions / rows | 281 / 3339 | 221 / 2637 |
| Scripted readouts | 36 | 27 |
| Fake collection environments / teacher steps | 64 / 568 | 44 / 388 |

All original structural/RSS/time ceilings held. Linux parent commands took
31.2541296s and23.7215490s; sampled bodies are not process-tree/import peaks.
Every genuine solve, optimizer/parameter-dependent forward, simulator step,
new scientific source/model reader, probe and final update was **zero**. Literal
snapshot/fixture arithmetic and scripted goals test composition, not native
learning. Accepted #199 geometry/loop and #200 native control evidence remain
separate; no control refit or new comparative claim.

Exact commands (all successful final invocations exit0):

```powershell
.venv/Scripts/python.exe -B artifacts/runs/a4-reference-workflow-20261026/qualify.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/a4-reference-workflow-20261026/qualify.py windows-2 tests/test_a4_reference_workflow.py -k 'describe_before or execute_actual_cli or irrelevant_options or other_modes_exclusive or require_explicit or freeze_original_policy'
.venv/Scripts/python.exe -B artifacts/runs/a4-reference-workflow-20261026/qualify.py windows-3 tests/test_a4_reference_workflow.py -k 'use_actual_partial or incomplete_or_changed or failed_collection'
.venv/Scripts/python.exe -B artifacts/runs/a4-reference-workflow-20261026/qualify.py windows-4 tests/test_a4_reference_workflow.py -k 'prior_success and (budget_bool or budget_limits)'
wsl -d Ubuntu -- env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 timeout 110s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/a4-reference-workflow-20261026/qualify.py linux-1
wsl -d Ubuntu -- env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 timeout 60s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/a4-reference-workflow-20261026/qualify.py linux-2 tests/test_a4_reference_workflow.py -k 'prior_success and (budget_bool or budget_limits)'
```

The first Windows invocation initially failed imports before tests, then, after
fixing the ordinary legacy fixture path, executed64cases:59pass/5fail. These
five describe/execute failures exposed argparse's mutual exclusion of the new
combination. Moving only the fresh A4 flag outside that group, with strict early
mode refusal, fixed it. Subsequent15/14/2 affected cases pass. An early issue
comment's73pass count was corrected to authoritative59pass/5fail. Linux first
74 and final2 cases pass. Each prior failed attempt remains recorded.

Two prospectively recorded engineering amendments covered the final added
unknown-work/boolean-count/collection tests and observed-budget reporting fix.
They left original cumulative work/RSS/time/static caps unchanged. They were
software repairs, not scientific retries. The import failure cost1.4247747parent
seconds/zero executed work. Two orchestration JavaScript syntax errors performed
no calls/edits; two guessed old runner reads found no file; ordinary runner used.
One documentation patch anchor failed with zero edits, then the smaller exact
guide patch applied. No old failure/negative result was rewritten.

Scoped statics on the final four Python files:

```powershell
.venv/Scripts/python.exe -m ruff format --check scripts/run_minigrid_linear_fit_control.py scripts/minigrid_a4_adaptation.py tests/test_a4_reference_workflow.py artifacts/runs/a4-reference-workflow-20261026/qualify.py
.venv/Scripts/python.exe -m ruff check scripts/run_minigrid_linear_fit_control.py scripts/minigrid_a4_adaptation.py tests/test_a4_reference_workflow.py artifacts/runs/a4-reference-workflow-20261026/qualify.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent scripts/run_minigrid_linear_fit_control.py scripts/minigrid_a4_adaptation.py tests/test_a4_reference_workflow.py artifacts/runs/a4-reference-workflow-20261026/qualify.py
.venv/Scripts/python.exe -m mypy --platform linux --explicit-package-bases --follow-imports=silent scripts/run_minigrid_linear_fit_control.py scripts/minigrid_a4_adaptation.py tests/test_a4_reference_workflow.py artifacts/runs/a4-reference-workflow-20261026/qualify.py
```

All final checks pass. First Ruff found one unused test import; both initial
mypy targets found two annotation errors. Explicit annotations/import removal
fixed these without ignores. All three static cohorts, including failures,
cost38.1684266s <120s. Conservative cohort-or-later touched cache111016596B <160MiB;
inherited full caches597728709B(mypy)/390803B(Ruff) remain separate.

`artifacts/runs/a4-reference-workflow-20261026/` holds frozenPROTOCOL, runner,
import failure, exact logs/JUnit/JSON/parent commands, statics and
`final-metadata.json` with final case-set equality, source pins, costs/failures.
Stage80,144B before metadata/report closeout, below12MiB. No native output or
scientific artifact was reopened. Skipped unchanged mathematical/learner/
simulator suites, installed wheel/cleanclone/fullrepo/release/native public
workflow qualification, GPU/pretrained/browser/CI/sweeps. No Git mutations,
installs, agents or unrelated change removal; dirty HEAD064fb328 preserved.

## Remaining outcome and next action

Software and scripted end-to-end behavior pass. This does not establish a new
native public-command outcome, controller utility, CPC advantage, generalization
or release readiness. Earlier accepted native direct-stream outcomes, including
all partial/negative controls, remain their own historical evidence.

Next: freeze **one ordinary fresh drift CLI engineering acceptance trace** with
the existing source/defaults/seed/metrics and strict original nonrenewable
budgets, then validate the complete source→selected updates→partial-state
handoff→target-free games path. Its purpose is the newly connected public entry
point, not a rerun of the baseline comparison or a new scientific win. Reuse
accepted math/controls, no sweep/retuning, and preserve any failure without retry.
After that, reassess the full roadmap's external utility/generalization gaps;
do not automatically add another capture/provenance/checkpoint increment.

Live scope/handoff belongs to issue201/tracker18. Historical plans and development
log remain frozen under current AGENTS; broader unfinished criteria are preserved.

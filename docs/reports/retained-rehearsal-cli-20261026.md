# Retained rehearsal through the ordinary affine command

[R2.1-A4.J.CLI / issue228](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/228)
delivers command access to the existing fresh private rehearsal workflow.
Previous and current goal turns are PROGRESS. This increment has source,
procedure and result **self-review**, not independent acceptance. Full goal,
parents21/23/33 and227's native joint-retention question remain open.

## Working behavior

```powershell
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --context-linear-retained-rehearsal diagnostics --describe
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --context-linear-retained-rehearsal games --describe
```

Both modes pass the original fixed drift reference and retained_rehearsal=True
to the existing descriptor/workflow. Diagnostics sets native_games=False; games
sets True. Describe performs no source reads, collection, models or games.
For separately budgeted execution replace --describe with --output and a new
filename. Diagnostics still runs ordinary source games; games additionally
reads the six fixed private F/A3/A4 tasks per arm after both arms finish fitting.
No new CLI seed, dose, budget, horizon or feedback selection is exposed.

The mode requires --example-affine-a4. References, controllers, legacy modes,
source paths and fit overrides are refused before work. Describe plus output
is refused. The original defaults and controller/reference dispatch remain.
No new runner, model, optimizer, learner, data loader or dependency.

The existing workflow keeps original owners, clocks, private forks, paid fees,
final isolation and bounded partials. Both arms end6/108/head27 after two fits
totaling54rows per arm. The command preserves the entire returned report, including
failed/unknown work, and performs no retry. Source summary metrics describe
source A3/A4 games, not private F retention; the private comparison/readout fields
remain separate. No private learner is deployed or promoted.

Output uses the controller's original compact finite8MiB UTF8+LF writer,
extracted into one small shared helper. Existing destinations are refused before
work; exclusive creation also refuses races after work. Complete reports exit0,
including complete zero-goal results. Failed reports retain known partials and
exit2. Serialization errors raise with no retry, overwrite or truncated JSON.

Changed structure:

```text
scripts/run_minigrid_linear_fit_control.py   existing parser/main only
scripts/minigrid_a4_adaptation.py            new handler/shared writer; controller uses writer
tests/test_retained_rehearsal_cli.py         argument, composition and failure behavior
artifacts/runs/retained-rehearsal-cli-20261026/
  PROTOCOL.md, qualify.py, supervise.py      frozen finite software checks
  *-parent.json, *.xml, *.txt, statics.json  executed receipts
  combined.patch, review.json, software.json
```

The actual AST diff changes only main and handle_controller_cli, adding
handle_rehearsal_cli and _emit_fresh_cli_result. All fresh descriptor/learning/
stream/comparator/evaluator functions are unchanged. Six relevant learner,
solver, comparator, readout, simulator-boundary and budget source pins match
the accepted227 code. Exact inherited dirty snapshots and scoped patch retained;
no unrelated changes or Git mutations.

## Checks and accounting

38focused cases pass on Windows and cached Linux:27new command cases and11
existing controller/default/output regressions. Actual command→fresh source→
collector→both private arms→optional shared game loop→JSON runs for both modes
on literal numerical/environment ports. One further case fails the second
private readout, preserves the complete first six games and closed partial
second prefix, charges attempted2/returned1/unknown work, writes failed output
and exits2. RNG/source/actor/original zero-work owner checks pass in scoped tests.
These fake outcomes establish software behavior, not learned native usefulness.

Windows1's first describe assertion failed because it compared JSON lists with
Python descriptor tuples. Only the test expectation changed to a JSON
roundtrip. That failed attempt did no model/collection work and is retained.
Windows2 then38PASS; Linux1 all38PASS. Windows39instances=38pass/1fail;
Linux38pass; zero skipped/error cases. No acceptance criteria weakened.

| All literal attempts | Windows | Linux |
|---|---:|---:|
| Body seconds | 13.2295826001 | 20.6310196100 |
| Finite parent seconds | 17.0418139999 | 31.1440119001 |
| Sampled peak RSS bytes, after imports | 113,590,272 | 108,597,248 |
| Literal owners / snapshots | 194/2,021 | 194/2,021 |
| Scripted fits / row exposures | 24/486 | 24/486 |
| Scripted predictions / rows | 129/1,281 | 129/1,281 |
| Source/private readout calls | 10 | 10 |
| Fake private environments / attempted steps | 20/39 | 20/39 |
| Literal collections / closed records / teacher steps | 6/18/54 | 6/18/54 |

These counters include all failed calls and the late game exception. The
three full-workflow fixtures take the ordinary packaged TRAIN read path once
each; no historical scientific model/result files are read. Native constructor/
predict/fit/LAPACK ports are guarded or replaced with explicit literals.
Genuine fits, learned forwards, LAPACK, physics, teacher collection, random model
draws, final labels and optional simulator/Torch imports are zero. All software
caps remain below180sbody/220sparent/512MiB and frozen count/storage allowances.

Both fresh module subprocess --describe commands exit0 (1.3632365/1.3567273s).
Their descriptors match existing definitions/historical_source_reads=0. No
child-process/whole-tree RSS claim. Separate in-process runpy __main__ smoke
cost1.0097830s total, sampled peak51,671,040B before command imports. One warning
on the repeated in-process invocation (module already imported by the first
descriptor) is retained. Fresh module subprocesses pass without that warning;
no suppression or production workaround. These metadata costs are additional
to the table and remain within software caps; no native execution.

Scoped five-file Ruffformat/check/default+Linux-target mypy all exit0.
All observed static calls including formatting total18.3397812s, below120s.
Touched cache111,325,236B, inherited/current total598,435,616B separately;
current touched file sizes are not growth or cumulative IO. No ignores added.

Exact commands and outcomes reside in parent/static/description receipts:

```text
.venv/Scripts/python.exe -B artifacts/runs/retained-rehearsal-cli-20261026/supervise.py windows-2 .venv/Scripts/python.exe -B artifacts/runs/retained-rehearsal-cli-20261026/qualify.py windows-2
.venv/Scripts/python.exe -B artifacts/runs/retained-rehearsal-cli-20261026/supervise.py linux-1 wsl -d Ubuntu -- env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 200s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/retained-rehearsal-cli-20261026/qualify.py linux-1
.venv/Scripts/python.exe -m ruff format --check scripts/run_minigrid_linear_fit_control.py scripts/minigrid_a4_adaptation.py tests/test_retained_rehearsal_cli.py artifacts/runs/retained-rehearsal-cli-20261026/qualify.py artifacts/runs/retained-rehearsal-cli-20261026/supervise.py
.venv/Scripts/python.exe -m ruff check scripts/run_minigrid_linear_fit_control.py scripts/minigrid_a4_adaptation.py tests/test_retained_rehearsal_cli.py artifacts/runs/retained-rehearsal-cli-20261026/qualify.py artifacts/runs/retained-rehearsal-cli-20261026/supervise.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent scripts/run_minigrid_linear_fit_control.py scripts/minigrid_a4_adaptation.py tests/test_retained_rehearsal_cli.py artifacts/runs/retained-rehearsal-cli-20261026/qualify.py artifacts/runs/retained-rehearsal-cli-20261026/supervise.py
.venv/Scripts/python.exe -m mypy --platform linux --explicit-package-bases --follow-imports=silent scripts/run_minigrid_linear_fit_control.py scripts/minigrid_a4_adaptation.py tests/test_retained_rehearsal_cli.py artifacts/runs/retained-rehearsal-cli-20261026/qualify.py artifacts/runs/retained-rehearsal-cli-20261026/supervise.py
```

No broad repository suite, current-wheel installation, UI, CI/GPU, independent
agent review, native model/game execution or comparative rerun was performed.
Those were outside the changed command boundaries. All earlier scientific
negatives and spent runs remain;227's raw capture was never reopened.
Stage before final report/receipts194,622B, below12MiB; final receipt records
final storage/log/metadata sizes. README and workflow usage document costs,
private isolation and the unknown native outcome. Historical plans/log stay
frozen; GitHub owns live status per AGENTS.

## Remaining goal and next action

228's scoped command acceptance is complete after self-review. No native joint
retention, CPC, generalization, matched-compute, efficiency or release benefit
is established. The existing227 allowance remains failed/spent, with no retry.
The known224 mutual-forgetting result remains unchanged. Existing learnability,
margin diagnostics, BP/PC/neutral/active controls and broader roadmap gaps are
preserved; software compatibility does not close them.

Next precise action: qualify the updated wheel offline from an empty directory,
using the existing cached install workflow, and verify the registered command's
two new descriptions plus default/controller descriptions without checkout or
research artifacts. Why this: the last installed deployment qualification185
predates the new rehearsal/controller/probe command changes; current source
tests do not prove installed users can access them. Scope/budget that release
check in its own issue before execution; no native training or renewed227
allowance. This is an installation gate for usable functionality, not another
capture/provenance or numerical cohort increment. Full goal stays ACTIVE.

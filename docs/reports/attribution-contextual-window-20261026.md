# Existing fitted chooser in the finite TRAIN window

[#203 / R7.4-MG.A](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/203)
delivers scoped software integration. Source/procedure/result self-review, not
independent acceptance. Full goal, R7 utility/generalization and broader parents
remain open. Earlier native controls, negative utilities and spent failures stay
unchanged; no native model/source/result artifact was opened or rerun.

## Delivered behavior

`CounterfactualController.application_snapshot()` now exposes an original frozen
fit only after its four complete controller_train returns. It retains the actual
request/return/measures, prospective objective, estimates, policy and original
record identities in bounded storage: four returns and one frozen pin. Returned
snapshots are detached. Existing `snapshot()` fields and chooser/fitting math
remain unchanged. Cost measures retain their original time/RSS/violation proxies;
actual native work still requires its external ledger.

`AttributionStream.run_contextual_window(exact_owner)` reuses the existing finite
arrival/diagnostic/FIFO/wake/checkpoint loop. It accepts neither a bare fitted
dataclass nor arbitrary chooser callback. Original base and current full payloads
must match the actual learner and actor version, even with a constant digest.
Each binding callback uses the existing candidate lease, original clock and
wall/RSS budget, and checks the last owned native checkpoint before/afterward.
Frozen fit evidence, source/actor, credit, labels, roles and paid progress remain
checked. The chooser sees causal TRAIN scalars only; application never fits,
relabels ticket fits, reads final/probe data or grants promotion/renewal.

Complete/partial windows, correct-oldest waiting, bounded detached reports,
sealing and failure/unknown-work charges remain. Fit records and application
binding reads/diagnostics/updates remain separate. Fixed-reference entry/defaults
and original counterfactual snapshot outputs remain compatible. No new
algorithm, framework, dependency or hash system. [Usage](../attribution-contextual-window.md).

## Executed evidence

**99 distinct case IDs pass per platform**, with matching final sets, no skipped
or final failed cases. Unaffected passing cases were reused; the original full
cohort is not retroactively described as containing the later tests.

| Software cost, all attempts | Windows | Cached WSL Linux |
| --- | ---: | ---: |
| Case instances, failures/repetitions included | 119 | 105 |
| Body seconds after imports | 8.0163015 | 39.0315350 |
| Sampled body RSS bytes | 92889088 | 93245440 |
| Literal constructors / snapshots | 389 /3468 | 323 /2804 |
| Scripted updates / rows | 98 /98 | 86 /86 |
| Scripted predictions / rows | 98 /98 | 74 /74 |
| Counterfactual run attempts, including refused calls | 281 | 225 |
| Binding-read attempts | 411 | 313 |

Genuine LSE/SGD/parameter-dependent forwards/native physics/new scientific
readers/probe/final work were zero. Trial measures are scalar fixtures, not
native empirical utility. Existing #202's useful conventional native result and
older matched BP/PC/neutral/active evidence remain their original scoped results.

The actual real-owner→four-trial-freeze→real-stream tests cover wait/wake choices,
label delays0/1/2, hard credit, queued correct packets, native weights/optimizer/
RNG payload differences even under constant digest, source version, changed and
equal-replaced fit records, future fit clock, final/probe refusal, detached
reports, source/actor/caller RNG preservation, busy/reentrant calls, lost/late
returns, immutable budgets, changed diagnostic-time policy, changed callback
clock/candidate, and fixed-reference/counterfactual regressions.

Exact executed pytest entry commands:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/attribution-contextual-window-20261026/qualify.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/attribution-contextual-window-20261026/qualify.py windows-2 tests/test_attribution_contextual_window.py -k 'apply_actual or changes_during or future_on or binding_reads_to'
.venv/Scripts/python.exe -B artifacts/runs/attribution-contextual-window-20261026/qualify.py windows-3 tests/test_attribution_contextual_window.py -k 'apply_actual or callback_state_changes'
.venv/Scripts/python.exe -B artifacts/runs/attribution-contextual-window-20261026/qualify.py windows-4 tests/test_attribution_contextual_window.py -k 'apply_actual or callback_state_changes'
wsl -d Ubuntu -- env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 timeout 110s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/attribution-contextual-window-20261026/qualify.py linux-1
wsl -d Ubuntu -- env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 timeout 90s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/attribution-contextual-window-20261026/qualify.py linux-2 tests/test_attribution_contextual_window.py -k 'apply_actual or callback_state_changes'
```

Windows91 initial and12 added/affected cases passed. During source review, two
more clock/candidate regression cases motivated the existing-lease check. Its
first8-case cohort failed before any application updates/predictions because it
referenced nonexistent `runtime._clock`. Correcting to the actual inbox clock
made the exact8-case repair pass. Linux97 then8 affected cases passed. This
failure remains in windows-3 log/XML/JSON; it was never erased or accepted.

Before the repair, Windows had249 of its original256 scalar counterfactual
attempts charged and needed32 more for the exact8 cases. The prospective issue
amendment raised only that engineering ceiling to320 (actual281), preserving
original256 and all failed attempts. Linux stayed256 (actual225); all other work,
RSS/time/static limits stayed unchanged. No native/scientific allowance changed.
This used the single original permitted affected repair, not repeated retries.

Scoped final formatting, Ruff and both mypy targets pass:

```powershell
.venv/Scripts/python.exe -m ruff format src/app/coupled_attribution.py src/app/counterfactual_controller.py tests/test_attribution_contextual_window.py artifacts/runs/attribution-contextual-window-20261026/qualify.py
.venv/Scripts/python.exe -m ruff check src/app/coupled_attribution.py src/app/counterfactual_controller.py tests/test_attribution_contextual_window.py artifacts/runs/attribution-contextual-window-20261026/qualify.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent src/app/coupled_attribution.py src/app/counterfactual_controller.py tests/test_attribution_contextual_window.py artifacts/runs/attribution-contextual-window-20261026/qualify.py
.venv/Scripts/python.exe -m mypy --platform linux --explicit-package-bases --follow-imports=silent src/app/coupled_attribution.py src/app/counterfactual_controller.py tests/test_attribution_contextual_window.py artifacts/runs/attribution-contextual-window-20261026/qualify.py
```

Initial mypy targets each found four heterogeneous fixture-dictionary annotation
errors; explicit `dict[str, Any]` fixed these without ignores. All static attempts
cost61.8778533s <120s, including the initial three-file format parent. Conservative
cohort-or-later touched caches111103044B <160MiB; inherited whole mypy597773765B /
Ruff391347B remain separate. Linux parents36.0904920s and19.1605340s remained under
their original120s bounds. Sampled bodies exclude import/process-tree peaks.

One initial test-filename guess found no file; actual sibling was discovered
with rg. One orchestration JavaScript syntax error made no tool calls/edits.
Both were corrected without an audit/loader framework.
One closeout documentation anchor also matched an obsolete paragraph; it made
zero edits and was replaced by a small exact heading patch. Skipped unchanged math,
native-model/simulator suites, native controller/trial studies, public CLI,
wheel/cleanclone/fullrepo/release, GPU/pretrained/browser/remoteCI/sweeps. No agents,
installs, Git mutations, deletes or unrelated change removal. HEAD064fb328 plus
inherited dirty work remains. Historical plans/log are frozen under AGENTS.

`artifacts/runs/attribution-contextual-window-20261026/` contains PROTOCOL,
qualifier, all logs/XML/JSON/parent commands/statics, final-metadata with exact
case sets/source pins/failures/costs, and RESULTS. This is scoped software
correctness, not native learned scheduling, utility/generalization/CPC advantage
or repository/release readiness. Live handoff belongs to issue203/tracker18.

## Exact next implementation

Compose the existing four-trial owner with real MiniGrid TRAIN contexts and
closed-loop task measures through the existing stream/evaluator. Use a current
owned A3 base so clean and mistake contexts come from actual arrived predictions;
preserve the full base state, setup costs, per-trial budgets, discarded work and
frozen objective. Establish before execution whether the existing current-only
prefix/readout path can carry the warm base without changing native counters or
relabeling clocks. Qualify that composition with literal ownership/cost/isolation
checks before a separate small native protocol. No invented fit estimates,
final/probe-derived context selection, utility-weight tuning, baseline sweep or
extra capture/provenance/audit sequel. A policy that prefers waiting or loses to
the existing heuristic remains a valid result.

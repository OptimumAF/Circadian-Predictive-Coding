# R7.2-MG.CLI / R4.8 — fixed TRAIN reference command

[Issue178](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/178)
exposes the existing fixed-window API through the existing affine command.
Implementation and scripted CLI/runtime behavior qualify; native task usefulness,
independent acceptance, installation and release readiness do not follow.

## Delivered behavior

`scripts/run_minigrid_linear_fit_control.py` accepts an optional exact
`ReferencePolicy` on the current-only A/B stream route and the mutually exclusive
`--context-linear-reference {no_learning,periodic,budget_aware}` CLI option.
The descriptor is pure and declares the original90second/512MiB/six-update
ceiling,54arriving rows, seven opportunities and zero scores/games. It reports
fit upper bounds rather than promising that every update executes.

The new branch uses the existing `AttributionStream.run_reference_window`.
Original labels/FIFO/actor/source/budget checks and failure charges stay intact.
It seals without full-training `finish`, implicit drain, scores, weight exports,
endpoint handoff or games. Policy and incompatible recipe checks precede payload
and model work. Existing always-train, catalogue, recurring and A4 routes remain;
their inherited changes were preserved.

The original fixed defaults produce0/3/2updates and6/3/4pending packets in the
scripted no-learning/periodic/budget-aware cases. CLI status describes window
completion; `stream.training_complete` separately reports whether every packet
trained. There is no source-label or metric-based policy/seed/dose selection.

Late-stop scope is explicit: a committed receipt survives in the inner failed
window report even when strict final outer capture cannot reconcile the loop
counter. Outer `committed_receipts` then remains null with a capture error and
failed status. Lost returns remain attempted/unknown with committed counter0;
they cannot retry. Source/policy/capture changes are refused, not silently fixed.

## Executed verification

Stage: `artifacts/runs/affine-reference-cli-20261025/`. All exact command arrays,
stdout/XML/counter/parent receipts and self-review remain local.

**38distinct checks qualify per platform:**32new API/CLI/runtime cases and6old
pure description/dispatch cases. Actual parser→wrapper→delayed runtime→JSON
paths execute with owned scripted learners and ordinary literal TRAIN episodes.
They perform no genuine model initialization, least-squares solve, gradient,
forward, teacher, physics, replay, probe or final work. Literal NumPy arithmetic
and fake default-path predictions are counted separately.

| Cohort | Windows body / parent seconds | Linux body / parent seconds |
| --- | ---: | ---: |
| Initial30 | 1.3249763 / 2.1391436;29PASS1FAIL | 6.049039473 / 12.1706245;30PASS |
| One affected late-stop correction | .8946517 / 2.2188786;1PASS | Reused corrected full cohort |
| Extra2 + legacy6 | 1.2073144 / 2.1512830;8PASS | 6.530220585 / 12.9663020;8PASS |

Initial sampled body RSS88,129,536B/88,125,440B; affected Windows87,576,576B;
extra90,046,464B/88,465,408B. Sampling starts after imports; these are not
import-time or process-tree peak claims. Windows cumulative37scripted owners/
18updates162rows; Linux33owners/17updates153rows. Each platform16scripted
prediction calls/144rows and one `finish`, exclusively in the unchanged default
regression. New reference routes call neither. Forbidden genuine work count0.

The initial failure was a test expectation: it required the outer final receipt
capture to succeed after a late stop. The existing strict counter refusal was
correct; the inner report retained the actual receipt and charge. The test was
corrected to verify both scopes. No production repair, weakened counter check,
or new numerical allowance was needed. The failed cohort is preserved. The
single affected repair was prospectively recorded; unchanged passing cases were
reused. No other software/static/runtime failures occurred in this increment.

Scoped four-file Ruff, formatting and default/Linux-target mypy pass; source
and tests received no semantic changes after their passing cohorts. Exact args
are in `statics-final.json`, including `--explicit-package-bases` for the test
package imports. Self-review inspected the descriptor, admission, terminal
branch, failure/final-export handling and parser/dispatch/caps along with the
inherited A4 diff. It is not independent review. Final metadata records scoped
whitespace, new-file checks, cumulative static costs, cache and stage sizes.

Representative exact commands:

```console
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/affine-reference-cli-20261025/qualify.py windows-1
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/affine-reference-cli-20261025/qualify.py windows-late-repair tests/test_affine_reference_cli.py::test_should_save_failed_cli_prefix_and_preserve_original_attempt_costs[late_wall]
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/affine-reference-cli-20261025/describe.py windows-describe-periodic periodic
```

Full extra targets and cached WSL equivalents/CPU1/timeouts are in their parent
JSON receipts. The description helper executes the actual public module
`__main__` through stdlib `runpy` with the documented CLI arguments; it does not
replace the parser or descriptor. Three modes ran once on each platform with
RSS sampled before optional imports. Windows combined body1.4479054s/maxRSS
45,309,952B; Linux3.322679190s/maxRSS49,852,416B. Counted UTF-8 descriptor payloads
are785/782/786B; on-disk sizes in metadata include platform newline translation.
No TRAIN source or model was read. These validate source-checkout command delivery,
not installed-package behavior.

## Scope and next action

[Usage](../attribution-reference-window.md). No native CLI fitting study or
gameplay was launched. #177's prior actual0/6/3fit windows remain accepted at
that scope; its duplicate-ID game failure, spent allowance and missing final
RNG/broad preservation/game comparison remain unchanged. Its repaired native
path is unexecuted. #175/#154 UI and #174 old-wheel limits also remain open.

Full suite, clean clone/current-wheel installation, browser, independent review,
neural/CPC/robot/simulator and scientific/release checks were skipped because
this change only exposes the qualified TRAIN path. No new dependencies,
downloads/install/GPU/CI/Git mutations/subagents or raw historical readers.
All negative and unfinished scientific requirements stay preserved. Historical
plans/log are frozen under AGENTS.md; GitHub issues own live handoff.

Next implementation target: bind an exact existing `PromotableActor` to the
public MiniGrid prediction boundary so actual published-policy behavior can be
observed. Existing MiniGrid readouts freeze checkpoints; the actor already has
locked public predictions and complete generation/version snapshots. Reuse the
existing episode loop and promotion controller, pin each episode's serving state
and refuse mid-episode changes before actions. Start with bounded literal
ownership/version tests, then freeze any genuinely new live-serving protocol.
Do not retry the spent fixed-reference comparison or add another audit runner.

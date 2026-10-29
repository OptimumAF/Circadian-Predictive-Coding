# R4.2-A4.D.CLI / P7 — measured state-comparison cost

[Issue184](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/184).
Previous185 was concrete progress. This increment delivers a measured validation
repair and fresh resource qualification; full goal and broader parents remain
open. Single-agent source/procedure/result SELF-REVIEW, not independent acceptance.

## Delivered behavior

`src/app/coupled_attribution.py::_same_value` handles exact builtin scalar leaves
before reflective tensor/dataclass/container dispatch. It performs the same
equality checks and recursive traversal. There is no state caching, identity
shortcut, reduced evidence checking or changed snapshot/receipt/resource/clock
contract. Unsupported subclasses remain refused; bool/int/float distinctions,
NaN inequality, arrays and nested mutation checks remain. All inherited reference
window work in this dirty module stays intact. No solver/objective/configuration,
dependency, model protocol, CLI default or scientific metric changed.

Why this: the original184 Windows cohort passed27assertions but spent143.5350452s
against120s. Before editing, ONE standard cProfile of its existing positive
literal command case identified24,018,692recursive comparator calls,34.955s self;
24,015,563is_dataclass calls spent10.021s self. Snapshot reconstruction was a
smaller cost. The old ordering sent primitive coefficient/history leaves through
reflective dispatch. Moving the identical scalar branch avoids that overhead.

Profile:1PASS/26deselected,78.644207400sbody/82.236381600parent/RSS99291136B,
801466B rawprofile.180,023,799totalcalls; cumulative sampler-thread/context totals
exceed elapsed wall and must not be summed. Profile instrumentation makes this
unsuitable as an unprofiled before/after throughput denominator. No profile retry.

## Executed checks

| Final-source operation | Result | Body seconds | Parent seconds | Sampled RSS bytes |
| --- | --- | ---: | ---: | ---: |
| Positive literal command |1PASS/26deselected |11.601381900 |13.064342100 |96821248 |
| Original full Windows cohort |27PASS |85.268868800 |86.629974100 |97873920 |
| SAME full Linux cohort |27PASS |85.402357809 |98.296086900 |101519360 |
| Equality behavior Windows |16PASS |not separately sampled |1.518204400 |not sampled |
| Equality behavior Linux |16PASS |not separately sampled |8.259751000 |not sampled |

No assertion failures, errors or skips. New equality cases cover exact primitive
values/types, NaN, unsupported equal/identical subclasses, dataclass/class values,
nested mutation on successive reads, list/tuple distinction, and NumPy shape,
dtype and changed values. Original full cohort includes lost start/commit/fit/
game/final-capture returns, corrupted/replaced preceding evidence, guard refusal,
RNG-failure reporting and private/published defaults. Forced failure cases retain
their paid prefixes; these are tests, not new genuine model work.

Both full cohorts have EXACTLY the same three work dictionaries as the original
failed Windows cohort:102empty metadata owners/14,681snapshot calls,42literal
fits540rows,213literal predictions789rows,71literal environments140steps,
18collectionslots162literalteachersteps. Forbidden genuine solver/forward/
physics/modelRNG/final/probe work0. Profile and positive each add10owners1820
snapshots4literalfit54rows24predict88rows8env16steps2collection18teacher on Windows.
Cumulative Windows122owners18,321snapshots50literalfit648rows261predict965rows/
87env172steps22collection198teacher; Linux102/14,681/42/540/213/789/71/140/18/162.
All prospectively amended cumulative caps pass. RSS sampler is after imports,
not exhaustive allocation/import/process-tree peak. Snapshot-slot bounds do not
measure all copying/equality traffic or FLOPs.

Same27-case Windows elapsed fell from the old143.535s to85.269s in these separate
runs, with unchanged work counters and120s criterion. This is scoped engineering
evidence, not a repeated statistical performance benchmark or learning-efficiency
claim. The original120s failure remains FAILED-SPENT and its files unchanged.
New changed-source qualification passes original120sbody/150sparent/512MiB limits
on both platforms; no budget was retroactively raised.

Exact commands at root:

```console
.venv/Scripts/python.exe -X utf8 -B -m cProfile -o artifacts/runs/deployed-a4-validation-performance-20261026/before.prof artifacts/runs/deployed-a4-cli-20261025/qualify.py perf-before -k real_command_to_two_publications_and_eight_actor_games
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/deployed-a4-cli-20261025/qualify.py perf-after -k real_command_to_two_publications_and_eight_actor_games
.venv/Scripts/python.exe -B artifacts/runs/deployed-a4-cli-20261025/checks.py windows perf-full-windows
.venv/Scripts/python.exe -B artifacts/runs/deployed-a4-cli-20261025/checks.py linux perf-full-linux
.venv/Scripts/python.exe -B -m pytest -q -ra --import-mode=importlib tests/test_owned_state_equality.py
```

Linux equality uses cached `/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python`
through WSL/env/timeout25s, parent30s. Both equality commands save JUnit XML;
expanded argv/environment/outcomes are in `checks-first.json`/`equality-linux.json`.
Scoped Ruff format/check/format--check and default+Linux-target mypy
`--explicit-package-bases --follow-imports=silent` cover the source/newtest,
allPASS,9.856474000sstatic subtotal. No further source edits after behavior checks.

Raw evidence: `artifacts/runs/deployed-a4-validation-performance-20261026/`
PROTOCOL/REVIEW/profile/pstats/parents/equality/statics/finalmetadata; original
qualifier stores only NEW `perf-before`, `perf-after`, `perf-full-windows` and
`perf-full-linux` prefixes in184stage. Old results are not rewritten. Two Windows
wildcard path searches and two guessed file reads failed during orientation;
ordinary `rg -g` and existing literal source paths resolved them, no mutation or
productive work. No test/static/model failure or qualification retry this turn.

Fullsuite, real Torch/numerical/native games, fresh descriptors on unchanged
no-work paths, cleanclone, installedwheel rebuild, browser, external independent
review, scientific/heldout/comparative/release checks skipped.183 native API and
185 installed wheel remain evidence for their original bytes/assumptions; this
does not claim a new native command cohort or updated installed wheel.182 RNG
gap,177/89spent comparisons, all neural/controller/CPC negatives remain unchanged.
No install/download/GPU/CI/Gitmutation/delete/reset/agents. Historical plans/log
remain frozen; GitHub is the live handoff.

## Critical path and one exact next implementation action

Do not start another checkpoint/publication/package/performance increment.
Useful A4 acquisition/retention and current-only loss/relearning already exist
in171/172; R2.1 probes and original matched BP/PC/neutral+activeCPC are implemented.
Their evidence is reused, not reconstructed. The existing A4 joint recipe uses
90fit exposures versus54for current-only, so it does not isolate replay from
extra row exposure. Current `run_context_linear_stream` and catalogue sources
have no explicit repeated-current exposure-matched A4 control.

Next: freeze and add an opt-in repeated-current transport control through the
EXISTING `run_context_linear_stream`/catalogue composition, preserving existing
A4 modes. Predefine per-wake current-row multiplicities `(1,1,2,2,2,2)` on the
same six A3→A4→A3 arrivals, yielding6fits90rows with explicit repeated-source
provenance versus joint6/90. Begin with literal role/arrival/ownership/exposure
tests and no solver. This addresses the observed confound before any future
small reviewed comparison; it matches fit count/row geometry, not measured
wall/FLOPs, and grants no new scientific allowance. Freeze all seeds/data/metrics/
directions/budgets before numerical work; preserve original unequal-work runs.
No selection from final data or baseline/CPC tuning to manufacture a win. Parent
retention/recovery/generalization/science/release claims remain unfinished.

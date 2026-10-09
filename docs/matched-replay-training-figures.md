# Matched replay: saved training records and schedule relations

## Completed scope

P9.5b27b validates both complete training bodies against all five original
schedules. Every policy, seed, boundary, applied method, ordered sample ID,
recorded example/update/inference count, manifest relation, role hash and
clock is retained. Outcome semantics and detailed guard/chemistry evidence
remain unfinished for P9.5b27c. P9.5b27, P9.5b and P9.5 stay unchecked.

The [inventory and schedule guide](matched-replay-schedule-figures.md)
records completed P9.5b27a. This increment independently freezes all nine
originals again; it does not borrow that guide's numbers or current defaults.

Why this increment: verify whether the recorded applied work actually agrees
with the saved planned work before interpreting the outcome bodies.
Saved records alone do not establish physical execution or equal compute.

## Complete original identities

All nine originals were pinned against the complete frozen publication
register before parsing. Full source copies are under `next-inputs/`.
Full publication/coverage/prior terminal/accounting metadata is under
`metadata/`. Whole originals, complete configs, nulls, booleans, negative
values and empty collections are preserved in `view.json`, the typed
table and `inventory.json`.

| Original path | Bytes | SHA-256 |
|---|---:|---|
| `data/continual_matched_replay_outcomes_v9_resolved.json` | 155300 | `2eb5b0937c992116ee18350dd2a858fb16522131ebc00018fbe5fecab149fe86` |
| `data/continual_matched_replay_outcomes_v9_resolved_repeat.json` | 155300 | `2eb5b0937c992116ee18350dd2a858fb16522131ebc00018fbe5fecab149fe86` |
| `data/continual_matched_replay_schedule_v9_final.json` | 41951 | `0498da71919e289202fe6caa18aec7dce5b75f57f176752d653ec4186641929c` |
| `data/continual_matched_replay_schedule_v9_repeat.json` | 41951 | `0498da71919e289202fe6caa18aec7dce5b75f57f176752d653ec4186641929c` |
| `data/continual_matched_replay_schedule_v9_resolved.json` | 63509 | `84d8241033c41a841456950de220cf8fa9e1904bff33888622a2e5bd661a29fa` |
| `data/continual_matched_replay_schedule_v9_resolved_repeat.json` | 63509 | `84d8241033c41a841456950de220cf8fa9e1904bff33888622a2e5bd661a29fa` |
| `data/continual_matched_replay_schedule_v9_smoke.json` | 41951 | `0498da71919e289202fe6caa18aec7dce5b75f57f176752d653ec4186641929c` |
| `data/continual_matched_replay_training_v9_resolved.json` | 54824 | `0e39f07428f5e9ea6e494abb4a2f4867ea77a63f9f4b87413fc5f47a83ece833` |
| `data/continual_matched_replay_training_v9_resolved_repeat.json` | 54824 | `0e39f07428f5e9ea6e494abb4a2f4867ea77a63f9f4b87413fc5f47a83ece833` |

The registered family is `matched-replay-v9`, classified
`corrected_development_or_fixture`, with incomplete original execution
environment. Its full frozen document identity and source binding are
retained, without asserting equality with current documentation.
There are no registered existing figure paths. Empty missing-file lists
do not establish complete environment or experimental provenance.

Four complete byte groups remain: three base schedules, two resolved
schedules, two training files and two outcomes. Both training files are
exactly byte-identical. This does not prove separate execution or
independent replication.

All nine bodies contain 10,299 typed leaves: 140 nulls, 270 booleans and
18 negative numeric values. Complete typed body checks include empty
collections; no configuration field was inserted or renamed.

## Recorded applied work

Both training files use `continual_matched_replay_training_v9`. Each
contains FIFO and seeded-reservoir cases for model seeds 17 and 19.
FIFO policy seed is null; reservoir policy seed is 53.
All rows have four ordered boundaries A1, A2, B1, B2 and three methods.

| Recorded field per boundary | BP | PC | CPC |
|---|---:|---:|---:|
| Applied examples | 2 | 2 | 2 |
| Applied optimizer updates | 2 | 2 | 2 |
| Applied inference iterations | 0 | 4 | 6 |
| Sum of examples over four boundaries | 8 | 8 | 8 |
| Sum of optimizer updates over four boundaries | 8 | 8 | 8 |
| Sum of inference iterations over four boundaries | 0 | 16 | 24 |

All method work records use the identical two selected IDs in the same
order within their boundary. Each boundary's selected list equals the last
two positions of its four unique retained IDs. Both source files preserve
every ID and all repeated exposures; totals do not deduplicate them.
Each `sleep_outcome` string is literally `accepted`; it is a record
claim, not independently proven guard execution.

The 32 copied training boundaries and 96 copied method work records across
the two bodies represent 16 shared case/boundary positions. No repeated
copy is counted as an additional independent experiment.
Equal recorded examples and updates do **not** establish matched compute:
the inference counts differ. No baseline, seed, policy, metric, score or
stopping condition was altered.

## Independent cross-schedule relations

Both complete training bodies were compared to every one of the five
whole bound schedule bodies, not just the resolved schedule.

Across the two training files, independent readback checked:

- 40 case pairs and 160 boundary pairs;
- 480 method pairs and 480 ordered sample-list relations;
- 1,440 numeric applied-versus-planned work relations;
- 160 phase training-role hash relations;
- 16 complete resolved-manifest pairs, with typed equality over every
  original config field and empty collection.

The training row `resolved_manifest` retains the original schedule
protocol `continual_matched_replay_schedule_v9`. This is the source
record's protocol, not a replacement training protocol.
Base schedules lack the manifest; no absent config was backfilled.
Policy, model seed, manifest digest, phase, epoch, retained order,
selected order, method identity and all three work counts match each
corresponding schedule record.

Train-role hashes equal the corresponding schedule's phase role hash.
Guard-role hashes have their exact recorded 64-hex form. Both role-hash
maps agree across policies for the same model seed. The training bodies
contain no physical role arrays or guard decision ledger. Hash encoding,
guard correctness, label access and physical isolation are unproved.

## Circadian clocks and explicit gaps

| Saved clock, all four cases in each body | Literal |
|---|---:|
| replay_updates | 8 |
| sleep_events | 4 |
| wake_batches | 4 |
| wake_batches_since_sleep | 0 |
| wake_examples | 44 |

Recoverable record relations: replay updates equal the sum of recorded
CPC optimizer updates, sleep events equal the four accepted boundary
strings, and wake batches equal the number of recorded boundaries.
All counters are nonnegative integers; the zero counter remains zero.

`wake_examples=44` is preserved literally. Its physical-sample derivation
cannot be established from the training hashes. The saved
`wake_batches_since_sleep=0` is consistent with the final accepted string,
but physical reset chronology is not established. No role counts, array
bytes, timings or missing runtime data were synthesized. Detailed guard,
chemistry, scoring, exposure and resource claims in the outcome originals
remain pending. The parent is not complete.

## Figures and visual review

All seven full PNGs were inspected. All 28 panels, 200 numeric labels,
zero inference/counter rows, headers, units and full footers are readable;
no observed overlap or clipping. Independent readback checked every
label/value/source binding, axis range, SVG bar geometry and decoded PNG
bar/zero pixels. Both whole training files are bound to every page.

- [Boundary examples](../artifacts/runs/p95-matched-replay-training-20261006/applied-examples.png)
- [Boundary optimizer updates](../artifacts/runs/p95-matched-replay-training-20261006/applied-optimizer_updates.png)
- [Boundary inference](../artifacts/runs/p95-matched-replay-training-20261006/applied-inference_iterations.png)
- [Total examples](../artifacts/runs/p95-matched-replay-training-20261006/totals-examples.png)
- [Total optimizer updates](../artifacts/runs/p95-matched-replay-training-20261006/totals-optimizer_updates.png)
- [Total inference](../artifacts/runs/p95-matched-replay-training-20261006/totals-inference_iterations.png)
- [Circadian clocks](../artifacts/runs/p95-matched-replay-training-20261006/circadian-clocks.png)

Every page also has an SVG. These are saved work records and arithmetic
sums, not fresh scores, uncertainty intervals or execution measurements.
Outcomes are wholly preserved but not semantically accepted in this scope.

## Commands, outcomes and evidence

Stage: `artifacts/runs/p95-matched-replay-training-20261006/`.
Exactly six ignored helpers: run.py, prepare.py, render.py, audit.py,
validate_static.py, finish.py. Final accounting is inline Python; no
seventh helper. Existing Pillow only, no dependency change.

From the repository run
`.venv/Scripts/python.exe -X utf8 <stage>/run.py <arguments>`.
Each exact argv/cwd/status/stdout/stderr/charged duration is retained in
`command-NNN.json/.stdout/.stderr`.

| Receipt | Arguments after run.py | Outcome |
|---|---|---|
| 001 | `<stage>/prepare.py` | Exit 0; whole checkout/instructions/metadata/nine originals frozen |
| 002 | inline `-c` inventory | Exit 0; all roots, full training/config fields inventoried |
| 003 | `<stage>/render.py` | Exit 0; whole view/table/inventory/cross-schedule/seven PNG/SVG pairs |
| 004 | `<stage>/audit.py` | Exit 1; expected plot dictionary field order differed |
| 005 | `<stage>/audit.py` | Exit 0 after auditor correction; 312 arithmetic, 1,440 numeric cross-relations, 480 ordered lists, 200 labels, 28 refusal controls |
| 006 | `-m ruff format` all six owned helper paths | Exit 0; two formatted, four unchanged |
| 007 | `<stage>/validate_static.py` | Exit 0; Ruff check/format check/py_compile/full formatter AST equality |
| 008 | `<stage>/finish.py prepare` | Scoped acceptance/additive docs only after all gates |
| 009 | `<stage>/finish.py close` | Full checkout/source/task preservation gate |
| 010 | `--executable git diff --check --` eight scoped doc paths | Scoped whitespace gate |
| 011 | inline `-c` final accounting | Full terminal/metadata/source/output/acceptance/helper AST/task/budget gate |

The failed auditor candidate is retained as
`retained-candidates/command-004-audit.txt`; its receipt and charge remain.
Only expected panel object field order changed in the auditor. No original,
canonical value, plotted value or acceptance criterion was changed.
The complete repaired six-helper sources were saved before formatting;
the static receipt validates that version, not the failed version.

Fourteen refusal controls per training body reject duplicate/nonfinite JSON,
unknown root, missing boundary/method, boolean work, wrong selected order,
changed work, wrong sleep outcome, boolean clock, wrong train-role hash,
changed complete manifest field, wrong manifest digest and replay clock.
Diagnostic mutations were discarded and all canonical relations rechecked.

Closing results are authoritative in their exact receipts,
`terminal.json` and `final-accounting.json`; completion requires all
closing gates. Evidence also includes full metadata/source bodies,
`view.json`, `all-typed-leaves.json`, `inventory.json`,
`cross-schedule.json`, `visual-review.json`, `readback.json`,
`static-validation.json`, `acceptance.json`, `coverage-delta.json`
and reversible before/after documentation with inverse edits.

## Reconciliation, budget and skipped gates

The entry exactly matched the previous b27a terminal: 1,030 nonignored
files, 25 package versions, 438 task rows, HEAD
`182077545d12d880e918f73cbf142c2279c211da`.
Only b27b's checkbox changes; all other existing task rows remain
identical. One new unchecked b27c row and this guide are added.
All unrelated checkout bytes, sources, HEAD and packages remain unchanged.

Prospective engineering scope: 600 aggregate seconds, hard 60 per captured
child, 64 MiB owned; fixed 160 manual/discovery/visual/closing reserve plus
every command, including the failed audit. Final accounting reserves its
own complete 60-second cap. This is not a whole-session wall-time or
process-RSS measurement. Prior science 350.7925872/360 and runtime
168.7993043/180 remain spent without reset.

Full pytest, mypy, native/Torch, CI, clean-clone and original
scientific-reader gates skipped for ignored helpers/additive docs.
Their original acceptance gates remain required. No model, dataset,
device, browser, download, sweep, experiment, CI dispatch or publication.
Public APIs, algorithms, architecture, dependencies and existing source
functionality are unchanged.

P9.5/P9.5b/P9.5b27, G0/R0.3/full R3.1 and missing-source requirements
remain open. Physical execution, sample derivation, role isolation,
guard behavior, independent replication, compute fairness and original
environment remain unproved. Owning-with repair P6.7d2b2j6c stays
human-deferred. No criterion was weakened or unfinished work removed.

## Exact next action

P9.5b27c: freeze complete matched-replay-v9 publication/coverage metadata and all nine original files under a fresh small engineering scope before parsing. Independently validate both whole outcome bodies, every method/seed/policy metric and aggregate, role IDs/hashes, replay exposure/work, boundary and sleep telemetry, guard/chemistry/resource/failure/missing/unknown statements against bound schedule/training records. Reconcile all original parent criteria without dropping unverifiable claims. Keep P9.5b27 unchecked until its complete saved-family acceptance passes. No inferred physical execution/isolation/replication/compute fairness/scientific admission, source borrowing, current-default backfill or resumption of the human-deferred repair.

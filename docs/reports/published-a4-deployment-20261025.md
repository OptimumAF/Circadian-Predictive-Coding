# R3.CG / R4.2-A4.D — current-actor candidates and new-task deployment

[Issue183](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/183).
New public `ServingPromotionController.start_candidate` starts from the just-
published actor under original lifetime resources. This advances useful learned
deployment; no checkpoint or provenance audit replaces task behavior.
[Usage](../promotable-minigrid.md#start-the-next-candidate-from-a-published-actor).

## Implementation and scope

New `src/app/published_candidate.py` owns the bounded ordinary TRAIN transition;
existing controller delegates. A latest original publication receipt permits one
preparation attempt. Original old candidate must equal the current actor, ledger
complete/consumed, gate explicitly paused, capacity remaining. Managed/probe/
replay/action/consent-guard owners, pending work/tickets and private advance are
refused. A failed builder consumes the preparation attempt and leaves the old
owner current. The trusted configured builder must return an independent exact
complete-state model; no fit/promotion is performed by starting the owner.

Before retirement the helper rechecks original budget/clock/sampler/progress/
reader/config, sharing resource/limits/state, receipt, old cursor/private state
and complete actor snapshot. Final nonblocking actor lease checks the slot,
then plain retirement/publication assignments follow without native callbacks.
Old receipt total freezes; new capacity subtracts consumed identities; same
budget/gate/admission quota and elapsed time persist. No unpublished state is
rebased, no allocation/training/scientific allowance is reset. The old owner is
not retained by the new owner, so no unbounded success/model chain is introduced.
Caller-retained old histories are bounded by consumed original identity capacity.

`GenerationInboxCursor` format3 adds an explicit positive prior-update offset
without changing original InboxCursor format1/2 fields. Lifetime receipt numbers
continue from that offset. Inbox materialization/checkpoint retains it; original
actor-base and stale-publication checks remain. An offset is accounting rather
than authentication of preceding history. Ordinary new inboxes with an already
charged budget explicitly expose their offset rather than emitting an invalid
zero-based exclusive cursor. Native math/algorithms/default CLI remain unchanged.

Source files: core/inbox_cursor, app/experience_inbox, actor_shadow's private
handoff keyword versions, serving_promotion's public delegate, new helper and
tests/test_published_candidate.py. Existing inherited dirty work is preserved.
Default managed schemas remain unchanged; invoking this operation on a managed
owner is refused before adding its optional single-use marker.

## Verification and retained failures

45distinct checks qualify/platform:31new +14 explicit original cursor/header,
checkpoint and stale-publication/rollback regressions. Real app runtime, shared
public actor, guard/controller and cursor execute on scripted scalar native ports.
Two guarded publications change served values after private training while old
serving remains stable. Checkpoint roundtrip preserves offset; old history stays
readable and frozen after more cumulative work. Malformed offset/receipt labels,
copied/corrupt publication, pending/private/stopped/roles/actions/managed/consent,
callback control/old receipt/actor edits, failed builder single-use and busy final
actor lease are covered. These fixtures are not native learning or game evidence.

Windows initial42:27PASS15FAIL. Production compared identity-only receipt against
its saved deep copy, incorrectly refusing valid starts. Original object identity
plus complete field equality repaired it; all15affectedPASS. Five header-only
cases now construct a literal cursor directly rather than unnecessary models.
Then3pure receipt cases +1positive full transition PASS. Linux final45PASS.
No skips/foreign collection or genuine numerical/physics/model RNG work. Default
format field list unchanged; existing changed-base checkpoint/promotion refusal
retained. Source/procedure/result SELF-REVIEW, not independent acceptance.

Scoped final9-file Ruff/format/default+Linuxmypy PASS. Initial static type-narrowing
and literal annotation failures retained/fixed; one no-op patch context mismatch
changed nothing. Stage contains exact commands/environment/stdout/XML/work and
parent receipts, protocol, review and native evidence. No new tool framework.

| Software cohort | Body seconds | Parent seconds | Sampled RSS bytes |
| --- | ---: | ---: | ---: |
|Windows initial42 (failed) |1.807172400 |2.660452000 |85188608 |
|Windows affected15 |1.130909500 |1.871550300 |89931776 |
|Windows final4 |1.151788000 |1.905645700 |90546176 |
|Linux final45 |6.469049862 |12.359342800 |87461888 |

Cumulative Windows4.089869900sbody/6.437648000sparent,279literalowners453snapshots/
57scripttrain57rows176scriptpredict176rows; Linux6.469049862sbody/12.359342800sparent,
179literalowners307snapshots40scripttrain40rows106scriptpredict106rows. No native
physics/model draws/teacher/final/probe/historical scientific data in software
checks. All original cumulative caps pass; four extra self-review checks were
prospectively declared within them. Statics14.608784300s/touchedcache117356766B
(mtime-selected, not growth, inherited globals separate); stage2159730B before
metadata, under24MiB/output8MiB/log256KiB/meta4MiB. Final-metadata.json records
versions, final source hashes, all measured scopes and whitespace checks. RSS
is body after imports, not process-tree/import peak. Native costs are separate.

```console
.venv/Scripts/python.exe -B artifacts/runs/published-a4-deployment-20261025/checks.py windows windows-1
.venv/Scripts/python.exe -B artifacts/runs/published-a4-deployment-20261025/checks.py windows windows-repair -k "start_from_current or capture_and_checkpoint_generation or callback_changes or false_generation_offsets or busy_actor"
.venv/Scripts/python.exe -B artifacts/runs/published-a4-deployment-20261025/checks.py windows windows-final -k "generation_receipt_renumbering or start_from_current"
.venv/Scripts/python.exe -B artifacts/runs/published-a4-deployment-20261025/checks.py linux linux-final
.venv/Scripts/python.exe -B artifacts/runs/published-a4-deployment-20261025/checks.py statics statics-launch
.venv/Scripts/python.exe -B artifacts/runs/published-a4-deployment-20261025/checks.py native native
```

Fullsuite/cleanclone/currentwheel/unchanged math, baseline and simulator cohorts,
external review/heldout/statistical multiseed/release checks skipped. One agent,
no install/download/GPU/browser/CI/Git mutations/reset/delete. Historical plans/
log frozen perAGENTS; issue tracking is authoritative. #182 remains open with
FAILED-SPENT native serialization/RNG gap, never backfilled or retried here.

## Native result

The FIRST/ONLY prospective case COMPLETE, exit0 under NATIVE_PROTOCOL.md and
pre-launch REVIEW.md. Source/procedure/result SELF-REVIEW, not independent.
This is a distinct4fit54row/two-publication deployment case; no182 allowance
or its missing witness was renewed. Same final source as focused qualification.

| Actual served actor | Learned old A3 | Fresh A4 |
| --- | ---: | ---: |
|c0/gen1/2fits18rows |2/2 |0/2 |
|c1/gen2/4fits54rows |2/2 |2/2 |

Eight actual public actor games used seed17/max16/rawK7firstargmax and shared
prediction leases. A4 feedback was collected only AFTER currentactor start.
Four fits expose54rows:36arrival rows+18explicit repeated A3 rows from36unique
original rows. Eighteen teacher steps are charged separately. B18sentinel guard
rows remain disjoint and are not a learned-old/A4 guard cohort; C18unused.
Two unchanged Bsentinel guard evaluations accepted. No outcome changed the
recipe, guard, metric, seed, update schedule or publication protocol.

Original native4-update120s and readout0-update0-replay150s512MiB budgets/gate/
clock start before data/models. Old owner retires at2charges, new generation
starts from actualc0 with offset2/capacity2, then receipts3/4 under basec0 and
candidatec1. Both guarded publications change actor generations normally; no
unpublished A4 state is rebased. Original source/candidate/actor/control/paid
records validate before more work; old ledger total stays2 as live total reaches4.
Original actor preserves A3 while new candidate updates and then serves both
tasks after promotion. This is cooperative sequencing, not measured concurrent
throughput or hard native preemption.

Final in-process Python/NumPy RNG witnesses BOTH unchanged, recorded BEFORE
fallible serialization. JSON-normalized comparison passed; original native report
and scalar summary exported exclusively (2060150B report). No historical scientific
JSON or old source/result/RNG reconstruction. No retry or favorable selection.

Observed attempts/returns:4fits54rows4LAPACK,94native predicts158rows INCLUDING
8guard predictions72rows and86publicactor predictions86rows;10environments104
physicssteps10closes INCLUDING18teachersteps. All attempts returned. Ten empty
owners5768model snapshots4262gate observations; snapshot-record upper24,802,400
slots excludes copies/equality/allocations. One strict bundled94705B TRAIN read/
54decoded rows; source-generation sunk cost not remeasured. Catalogues, state
validation, guards, games and serialization included in measured time/RSS.
Body43.453743500s/parent44.260663200s/RSS79175680B after imports, including optional
runtime preflight, not import/process-tree peak. All prospective caps pass.

This supports limited known-TRAIN deployed acquisition and old-task retention.
No loss/regain recurrence recovery was tested. No heldout/generalization,
multiseed/statistical/matchedcompute/efficiency/CPC benefit or release claim.
All prior negative/failure/inconclusive evidence, including182, remains unchanged.

## Remaining goal / exact next action

Scoped API/accounting/software/native deployment criteria complete; full goal,
R3/R4/R7 and broader science/release parents remain open. Existing live A4 CLI
still deliberately stops at private learning, so normal users do not yet have
this guarded deployment workflow through that command. Next integrate the
qualified currentactor start and second publication as an explicit existing live
A4 CLI mode, reusing the collector/catalogue/runtime/serving loop and this native
deployment evidence. Preserve private-mode defaults and182 gap. Verify literal
CLI/partial failure/describe/output behavior; do not launch another native cohort
just for packaging/dispatch or start a new capture/audit increment.

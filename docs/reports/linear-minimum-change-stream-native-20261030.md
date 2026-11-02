# Single-head ordinary minimum-change: acquisition and recovery, incomplete joint retention

R2.1-LIN.SHARED.N / [#352](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/352).
The sole genuine ordinary run completed all24 fixed task slots under its original
clock. It learns both contexts, retains some tasks and recovers a lost A task.
Joint retention is incomplete. **The original observer coverage gate failed for
two cost categories**, so full scoped acceptance remains partial. No retry.
The full repository/scientific/release goal remains active.

## Observed task behavior

One single60-coefficient D9/K6 head, no retained rows, replay or context-head
partitioning. Same original A4/B4/A4 TRAIN packets, candidate checkpoints0/4/8/12
and task slots. These are fixed known development templates, not held-out tasks.

| Ordinary task | Initial0 | After A4 | After B8 | After A returns12 |
| --- | --- | --- | --- | --- |
| v1 invoice | fail | pass | fail | pass |
| v1 password | fail | pass | pass | pass |
| v2 invoice | fail | fail | pass | fail |
| v2 parcel | fail | fail | pass | pass |

Both A tasks are newly acquired. During B, invoice is lost while password is
retained. Both B tasks change from failure before B to success after B. On A
return, invoice genuinely recovers; password never needed recovery, so its
recovery value stays null. Returning A loses B invoice while retaining B parcel.
Final ordinary3/4 is not full joint retention or reliable adaptation.

All24 episodes complete, including failures. Final stale_once tasks pass for
both A tasks and B parcel; B invoice fails wrong_workflow. All four revoked
tasks stop permission_denied without workflow success. Earlier untrained tasks
fail max_steps; post-training incorrect ordinary tasks fail wrong_workflow.
The entire original result and signed summaries are retained, not selected
positive endpoints. Frozen task readouts use the stream candidate; the source
learner and deployed actor remain unchanged. This is not deployed adaptation.

The historical #282 cold/shared-head report recorded loss of both B tasks on
return; this observation retains B parcel. That is a descriptive difference
between saved development observations, not independent replication, matched
total computation, method superiority, generalization or a CPC advantage.
No old control was rerun. Accepted separate-head positives remain a different
capacity/routing protocol, not evidence this single head retained all tasks.

## Execution, original authority and costs

Frozen local protocol and observer:
`artifacts/runs/linear-minimum-change-stream-native-20261030/PROTOCOL.md`.
The observer called the existing public CLI once:

```text
--method linear_fit --context-return --train-update minimum_change --output <stage>/native.json
```

Producer terminal exit0. Body5.050620926s, sampled peak53,112,832B; Windows/WSL
host11.3462227s, no host expiry. Existing Linux Python3.12.3/NumPy1.26.4, requested
OMP/MKL/OPENBLAS/NUMEXPR threads1. Physical D:/Dev/Repositories/Circadian and
D:/Dev/_Temp (WSL /mnt/d/Dev/_Temp) used throughout.

Original source12-update/30s/512MiB authority remains: started_at8.956438728,
last_clock9.848567483,12 committed wake updates and0 replay exposures. Source,
actor, NumPy RNG and Python RNG unchanged flags all true; original causal
observed+1 arrival receipts and single-head minimum-change diagnostic verified.
Final/probe/replay/external-action counts and stream final-score forwards are0.
Original source/actor/consent/input/receipt/prefix guards were not bypassed.

Direct observed work:12 actual NumPy least-squares solves/72rows;1 initial cold
solve6rows plus11 warm incremental solves66rows.12 minimum-change owner ports
72rows,0 cold owner ports.34 scheduled training products204rows: one cold
residual product plus three per warm fit. These are schedules, not hardware
FLOPs.24 task episodes,92 typed calls,116 total dispatcher calls,2 teacher
builders;3 owner initializations,2 restores,2 forks.1370 snapshot API returns
include16,380 nominal coefficient values from fitted snapshots. That last
quantity is not all physical allocations/copies. Live fitted coefficients480B,
retained checkpoint coefficients0+480+480+480=1440B; Python/metadata additional.

### Original counter coverage failure — preserved

The observer's affine hook watched `_predict_design`, but typed inference calls
`predict_categorical_linear_logits`, which performs its own affine product.
The teacher hook watched rehearsal `_ticket_trace`, but the ordinary
`training_cases` builder generates its traces directly. Their original zeros
are unused-API counts, **not zero prediction or teacher work**.

Frozen producer/raw receipts remain untouched. `native-work.json`'s
budget_verified flag covers its sampled wall/RSS/import/storage checks and
hooked counters; it does not establish complete semantic counter coverage.
The prospective requirement to count those actual operations before work was
not satisfied. That gate stays FAILED/unverified; it is not repaired by a
post-result rerun, renamed workload, counter edit or retrospective positive flag.

Available guarded owner/source records reconcile68 fitted affine attempts and
returns on68rows,24 unfitted zero-logit reads,92 typed calls/returns,4080 nominal
affine terms,484 token visits. All are within the frozen numeric limits. These
are original guarded readout API quantities, not a second independent kernel
meter. The raw dispatcher116 minus recorded task92 gives24 teacher dispatches;
the fixed three-action builder implies8 teacher traces. Those are explicitly
source-derived quantities, not measured `_ticket_trace` calls. No global
probability API is used by this candidate-local typed normalization path.
Future experiments must bind the actual logits and direct teacher ports before
work; this spent case cannot acquire missing historical independent counters.

## Checks, review and retained failures

Accepted #351 software gates are reused unchanged: Windows40 distinct via39
unchanged+1 targeted repair, Linux-final40; no mathematical/baseline/model test
cohort repeated. Relevant eleven current source files were frozen as exact byte
copies and compared before/after the native case; no production edit occurred.

Stdlib-only preflight passes0.43326688s/19,755,008B, zero project/numeric imports,
and freezes exact phase/task/checkpoint/24slot values. Initial observer statics
had four heterogeneous-record type errors, repaired using a scalar stage size;
failed receipt retained. Three-file Ruff/format/Linux-target mypy then pass
1.9220784s; producer source-completeness/typed-row binding checks pass.980369s.
Launcher parsed successfully. No model work during those checks.

Independent prospective REVIEW-PRE accepts procedure within its read scope:
2 read commands1.1545762s, no imports/tests/models/network. It did not establish
actual outcomes and did not catch the later-observed unused-hook gap. Lead
self-review is separate. REVIEW-POST independently ACCEPTS the covered actual
task results and interpretation, including the failed counter-coverage limitation.
It explicitly does not accept complete original cost coverage; #352 remains
PARTIAL/SPENT. The combined PRE/POST review used four read-only commands totaling
1.8928083s, no imports/tests/models/network/children. The finite exception ended;
no independent numerical replication or full scientific/release acceptance.

After the sole terminal run, the coverage defect was disclosed before saved
analysis. The original analyzer preimage is retained. Its metadata-only extension
validates original guarded readout totals and labels the failed counter gate;
it changes no task metric or model. One patch anchor failed after formatting;
the simpler exact-anchor edit succeeded, with no source/numerical change.
One later usage-document patch also missed its anchor without changing files;
the actual paragraph was read and corrected. Scoped analyzer Ruff/format/Linux-mypy pass. Saved analysis passes
0.027473618s/13,844,480B, without project/numeric imports. All source/actor/RNG/
budget/receipt/slot checks pass; this does not upgrade the failed counter gate.

An ancillary Get-Content -TotalCount4 on minified native.json read its entire
single line; no extra rollout/fit/model reconstruction occurred. This and the
saved analyzer are separate raw reads; no sole-raw-read claim. An ancillary
critical-path source search named one absent script and returned that read error;
the existing generic rehearsal file was then inspected directly. All failed
captures and original #348/#344 FAILED/SPENT history remain.

## Commands, skipped work and next action

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/linear-minimum-change-stream-native-20261030/preflight.py
& artifacts/runs/linear-minimum-change-stream-native-20261030/launch.ps1
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/linear-minimum-change-stream-native-20261030/analyze.py
```

Exact launch argv/exits/hosttime: execution.json. Original raw/native-work/logs/
attempt/SPENT, preflight, source copies, statics, analysis and finite review
receipts remain local in the stage; no artifact publication, commit or push.
The same session31267 was polled to terminal, never restarted. No new package,
dependency, download, deletion/global change, broad suite, browser, GPU/physics,
fresh platform replicate, final/probe, old model/control refit or sweep occurred.
RSS is sampled process memory, not hard instantaneous/process-tree allocation.

Retain this result as useful mixed/negative evidence: single-head incremental
updates do not solve joint retention. No numerical retry to repair observer
coverage or manufacture a fully positive result. Existing separate-context
control is already useful under its explicit capacity/routing qualifications.

Next substantive implementation ([planned #353](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/353)): expose bounded TRAIN episode replay
for this single-head learner through the existing `StreamRehearsal` and ordinary
runner, rather than another checkpoint/export/counter case. Inspection confirms
that generic consolidation already delegates `learner.train_batch`, while CLI
and linear admission currently require replay=none. Reuse original causal TRAIN
episode/retention/receipt/separate replay-budget ports, preserve cold/default/
separate-head behavior and account all extra exposures. Start focused transport,
role/isolation/paid-failure integration checks before any separately frozen
outcome study. This is a new learner capability, not a promise that replay fixes
the remaining task. No new numerical allowance is granted by this report.

Canonical #352 PARTIAL/SPENT handoff: https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/352#issuecomment-6539044177.
Three criteria evidenced; original counter-coverage criterion2 remains unchecked. Parent receipts:18/6539050212,21/6539050375,23/6539050537,24/6539050736. Next #353 is planned only; no new model allowance executed. Full goal active.

Closeout metadata note: adding GitHub receipts initially failed JSON serialization because PowerShell parent keys were numeric. The prior handoff remained intact (2003B); explicit string keys produced verified JSON (2311B). A final document patch anchor also failed without changing files; this append records it. Original native/raw/work/protocol evidence and the failed coverage gate are unchanged.

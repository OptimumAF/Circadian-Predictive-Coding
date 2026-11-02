# Ordinary shared-head TRAIN replay: one fixed comparison

#363 / R5.3-LIN.SHARED.C, 2026-10-31. The sole native two-arm case completed on
its first launch. Both shared affine candidates acquired A and B. Current-only
lost v1 invoice during B, recovered it on A return, and lost v2 invoice on that
return. Joint retained both task pairs at the later checkpoints. This is useful
ordinary candidate learning evidence for these fixed development templates;
it is not circadian, statistical, generalization, efficiency or release evidence.

Canonical scope/budgets: [issue363](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/363).
Local evidence: `artifacts/runs/ordinary-shared-joint-comparison-20261031/`.
Base d9119e80926a139a2f38460564d7eab1b6d2a2b0 plus preserved inherited dirty work.
No production code changed. Existing #362 software/math and historical matched
BP/PC/neutral+active CPC evidence are reused within their scopes, not rerun.

## Actual outcomes

All ordinary tasks initially failed. Checkpoints are source wake updates; native
fit counts are0,4,10,16 and fitted exposures0,24,72,120. Fixed four tasks per point:

|Task|Current-only 0/4/8/12|Joint 0/4/8/12|
|---|---|---|
|v1 invoice|F/T/F/T|F/T/T/T|
|v1 password|F/T/T/T|F/T/T/T|
|v2 invoice|F/F/T/F|F/F/T/T|
|v2 parcel|F/F/T/T|F/F/T/T|

A totals: control0→2→1→2, joint0→2→2→2. B totals: control0→0→2→1,
joint0→0→2→2. Final ordinary success3/4 versus4/4. Joint-minus-control is+1A
after B and+1B after return; all other cohort/point differences0. Both demonstrate
new acquisition from prior failure. Control v1 invoice alone meets the predefined
loss/recovery condition; joint has no observed acquired-A loss, so its recovery
values are null. B-retention-after-return is false only for control v2 invoice.
Checkpoint changes do not measure exact acquisition latency or causal mechanism.

All48fixed episodes completed with known returned work. Final stale_once:
control3/4success/recovery (v2 invoice wrong_workflow), joint4/4. All four
revoke_commit episodes per arm reach staged state and stop permission_denied,
without a committed tool. Their success=false is preserved, not recast as task
success. All have0external actions/0task training. These sampled scenarios do not
establish general safety, and this learned candidate was not deployed/promoted.

## Procedure, isolation and cost

Actual family is context_return_queue_rehearsal_v1, not the private-probe family.
The existing ordinary CLI executes minimum_change with coverage and explicit
replay-joint current_only then joint. Two independently born original sources
use identical D9/K6 TRAIN data, initial state, coverage/current-packet/selected-ID
signatures and limits. First complete paid result, file identity/content pins,
source hashes and counters validate before second source construction. Each
source retains its original12update/30s/512MiB authority and original replay
72row/30s/512MiB authority/clock/consent/receipts; actual replay48rows. Source and
original actor stay unfitted; neither source clock is renewed. All observations
are development outer_selection; zero final/probe labels, probe updates,
promotion, outcome-based replay selection or model random draws.

Exactly16fits120rows per arm: wake12/72 plus replay4/48, preceded by two skipped
replay events. Joint fits current6+retained6; equal-row control repeats current6.
All attempted/returned counts agree. Both arms each use cold1/6rows plus
minimum15/114rows; total32LS240rows, minimum owner ports32/240rows, cold owner
ports0. Scheduled design products92/696row visits, not total FLOPs or equal SVD
conditioning/total computation. Probability and generic predict ports0.

Other total observed work:48task episodes,184typed decisions,136typed-affine
rows/calls,376dispatches (192teacher+184task),4direct teacher builders16traces
plus48trajectory teacher traces; generic owners14init/28restore/12fork,
contextual owners0. Snapshot API2828calls/35,880coefficient values. Input-hash
visits17,280B. Pickle18,976calls/37,047,324output-byte visits (control18,540,490B,
joint18,506,834B); these are not exact physical copy/heap counts. Each fitted
model480coefficient bytes, retained fitted checkpoints1440bytes/arm. Saved-file
integrity stat/hash/parse work is additional, bounded by four files/8MiBstage,
included in whole time/RSS; TRAIN-input hash counter excludes it. No claim of
complete OS I/O/copy/FLOP measurement.

Linux Python3.12/NumPy1.26.4, CPU, OMP/MKL/OpenBLAS/NUMEXPR1. Native whole body
3.132986807s/57,786,368B sampled peak, original90s/512MiB case limits; host
6.4685696s/exit0/not expired (105s limit). Observed arm wrappers1.043236152s and
1.042423479s; original source report .858346321s/.856836327s. Timing is descriptive
and includes validation, not a speed comparison. Original starts/last clocks and
all receipts are saved verbatim in reports/work. No retries or carry-forward
allowance; native.attempted/SPENT.txt retained. Saved analysis .024555117s,
GNU-time wall .16s/19,928KiB peak/exit0 (30s/128MiB bound), no model imports/work.
Dedicated cache initially absent/0; observed16,413,515B below64MiB. Final storage
receipt records retained stage size below8MiB; no evidence deleted.

## Checks, review and preserved failures

Commands from physical D: with TEMP/TMP D:\Dev\_Temp (WSL /mnt/d/Dev/_Temp):

- `.venv/Scripts/python.exe -B artifacts/runs/ordinary-shared-joint-comparison-20261031/check.py windows D:\Dev\_Temp\ordinary-shared-363-win-1`:35PASS,.6671354s/50,298,880B.
- WSL native Python same checker `linux /mnt/d/Dev/_Temp/ordinary-shared-363-linux-1`:35PASS,4.097547934s/37,920,768B.
- Final same checker `windows-repair ...win-2` / `linux-repair ...linux-2`:38PASS each,.6837905s/49,995,776B and4.492595412s/38,019,072B. Zero-model import guard has0violations; no fitting fixture.
- Five files producer/case_checks/analysis/check/test: `python -B -m ruff check`, `ruff format --check`, `mypy --explicit-package-bases --follow-imports=silent`, same mypy `--platform linux`: final allPASS. Initial16type diagnostics repaired with explicit metadata typing/SimpleNamespace/setattr; initial receipt preserved. Total recorded statics17.445825s; RSS not sampled for static tools.
- WSL native Python `producer.py --preflight`: initial .107644707s/31,272,960B; prospectively budgeted repair .105462292s/31,477,760B. Both source-only/no project/numeric imports; original retained as preflight-initial.json, replacement binds28named source+3driver files. Not a transitive authenticity claim.
- `& artifacts/runs/ordinary-shared-joint-comparison-20261031/launch.ps1`: ONE native process, exit0. Exact WSL/environment/timeout command in execution.json; producer calls existing CLI main for both arms.
- `timeout --signal=KILL 30s /usr/bin/time -v <existing-native-python> -B artifacts/runs/ordinary-shared-joint-comparison-20261031/analysis.py`: saved-only readout exit0; analysis.log retains memory/time receipt.

Independent PRE initially CHANGES REQUIRED: in-memory validation did not bind
saved reports. Repaired exact saved CLI/raw equality and bounded identity/content
pins plus three corruption/missing/identical-byte-replacement tests. No model
ran before repair. Amendment6542603997 preserves original preflight/budgets.
PRE GO after6readonly calls/1.0283811s, no model/test replication. Original finding
and repair remain in REVIEW-PRE.md. Independent POST ACCEPT after4readonly calls/
3.6560553s: all28+3hashes/four saved pins/raw equality/matched signatures, all48
slots/eight staged denied revocations and costs inspected. REVIEW-POST.md records
scope/exact receipts; no numerical replication or statistical/release acceptance.
Both finite exceptions ended. Lead source/procedure/result self-review also passes:
unchanged production ports, final observer/gates, failures, original charges and
qualifications reconcile with the records.

Skipped full suite/clean clone/wheel/install/browser/unchanged kernel/baseline/
physics/GPU/remote CI/final data/release checks: this adds a scoped experiment,
not production code. Earlier #354 PARTIAL/SPENT, #358mixed return, #360 private
fresh-ceiling result, #362 failures/P2/receipt/thread unknowns and every earlier
negative remain unchanged. No seed/metric/dose/threshold or baseline tuning.
No commit/push/download/delete/global change. Parents18/21/23/24/full goal open.

## Critical path

This closes a useful ordinary candidate evidence gap; it does not turn that
candidate into a serving learner. Actual CLI serving/adaptive modes still require
separate-context heads and replay none. The next substantial implementation is
planned as [#364 / R4.8-LIN.SHARED](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/364):
expose shared candidate typed serving through existing promotion/guard boundaries,
with development-only gates, original authority, paid refusal/rollback and default
parity. Existing contextual adapter and generic controller/stream ports were
reconciled; its guard state_valid still requires contextual geometry. Next: direct
shared-state slot/geometry/paid-read/refusal/default tests, then the smallest
binding/explicit ordinary opt-in, with fixture budgets before execution. The
planned ticket supplies no numerical allowance. Do not automatically
repeat this experiment, add another inspector/capture layer, or tune until a win.

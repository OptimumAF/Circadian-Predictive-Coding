# R7.5-CTX.LS.V.B — ordinary packet episode retention

Scoped #299 software increment accepted, 2026-10-29. GitHub issues remain the
live tracker; full learned-controller, retention/recovery, model-family and
release parents remain open. Earlier repeated-update/checkpoint evidence and
retired publication history are reused unchanged. No new checkpoint increment,
baseline reconstruction or #287 FAILED/SPENT retry was run.

## Delivered and observed

Following #298's old-task loss, source inspection identified a concrete reuse
dependency: existing dynamic joint rehearsal accepted one episode per wake,
while ordinary updates contain one or two three-step tickets. Optional ordered
`AttributionBatch.packet_episodes` now carries these actual complete TRAIN
episodes. Arrays, labels, roles, class counts, ordering, unique episode IDs and
schema must agree before admission. Original row IDs remain unchanged; arrays
and metadata are copied, and original metadata stays pinned through continuation.
The existing bank compiler still refuses this new metadata on its original exact
prototype path. No callback or new runner grants authority.

`compile_context_packet` certifies supplied ordinary new-v1/old-v1 rows against
authentic sandbox traces. Trace checking is diagnostic work, not free memory
recovery or an unrecorded label source. The helper retains each complete episode
in the existing bounded buffer, while the whole original packet remains one paid
delayed wake. It excludes retained episodes already wholly present in the current
packet by exact contiguous feature/target equality. It uses existing consolidation
and replay charges, including current rows repeated in a joint solve. Fixed-bank
probe semantics and the single-episode default remain unchanged.

Two fresh fixture streams (uninterrupted and paused/resumed after update2) execute
old6 → new3 → old6 through `AttributionStream`. Each returns3 wake receipts and
2 consolidation receipts,5 LS fits/33 fitted rows:15 wake rows plus18 replay rows.
Rehearsal events are skipped/complete/complete with0/9/9 charged exposures;
retained rows selected are6 then3. Both final candidates correctly predict all
supplied old/new TRAIN rows, with original source and actor snapshots unchanged
and retention within the existing30-transition/65536-byte policy. This is a
TRAIN readback on a fresh fixture source, not a closed-loop task outcome, a
deployed-source study, unseen-task generalization or scientific replication.

Replay cap8 refuses the9-row consolidation after2 paid wakes/9 rows; those wake
charges remain and replay records0 because no replay fit was admitted. Corrupting
the original packet metadata after the first wake refuses the next wake without
more work. Existing owner-local fake-learner continuation tests and the existing
isolated fixed-TRAIN/probe joint test also pass on the combined tree. No final
evaluation enters training, promotion, tuning or packet selection.

`current_only` still refuses6 current +3 retained rows because9 is not divisible
by a complete6-row packet. A pure test preserves that criterion. This increment
does not supply a new fair matched comparison or ACK consolidation proof.

## Commands, costs and acceptance

All commands run physically in `D:\Dev\Repositories\Circadian`, with process-local
TEMP/TMP=`D:\Dev\_Temp`; Linux TEMP/TMP/TMPDIR=`/mnt/d/Dev/_Temp`. Artifacts:
`artifacts/runs/context-packet-rehearsal-20261029/`. `source-hashes.json` was written
BEFORE combined checks and includes six source/test/protocol files; lead and
independent reviewer verified current hashes match. Receipts are exclusive files.

Pure gate:
```powershell
.venv/Scripts/python.exe -B artifacts/runs/context-packet-rehearsal-20261029/check.py artifacts/runs/context-packet-rehearsal-20261029/windows-pure-work.json tests/test_context_packet_rehearsal.py -k 'not execute_ordinary and not keep_completed and not changed_original'
```
16PASS,0fits/0rows,2.637833000000683seconds,70,762,496B peak. Seven authentic trace
checks/21 sandbox dispatches. One further pure complete-packet refusal test was
then added and included in both full cohorts. No failed numerical attempt.

Combined Windows:
```powershell
.venv/Scripts/python.exe -B artifacts/runs/context-packet-rehearsal-20261029/check.py artifacts/runs/context-packet-rehearsal-20261029/windows-combined-work.json tests/test_context_packet_rehearsal.py 'tests/test_context_linear_joint_rehearsal.py::test_should_keep_probe_inputs_out_of_fixed_train_bank[joint]' 'tests/test_context_linear_joint_rehearsal.py::test_should_compose_owned_same_row_count_and_refuse_cross_context' tests/test_rehearsal_continuation.py
```
54PASS,3.916868000000250seconds,79,036,416B peak.

Combined Linux (same selection plus `--capture=sys`):
```powershell
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-packet-rehearsal-20261029/check.py artifacts/runs/context-packet-rehearsal-20261029/linux-combined-work.json tests/test_context_packet_rehearsal.py 'tests/test_context_linear_joint_rehearsal.py::test_should_keep_probe_inputs_out_of_fixed_train_bank[joint]' 'tests/test_context_linear_joint_rehearsal.py::test_should_compose_owned_same_row_count_and_refuse_cross_context' tests/test_rehearsal_continuation.py --capture=sys
```
54PASS,16.685346570seconds,75,735,040B peak. Each complete cohort:17 native LS fits,
105 fitted rows,17lstsq calls;282 nested training-port calls/582 rows (wrappers,
native and fake ports, not distinct physical solves);4 probability calls/18 rows;
84 routes/345 rows;7 ticket-trace checks,2 training_cases calls,45 sandbox dispatches;
0 closed-loop readout episodes and0 Torch. Observer records0 separate affine
helper calls; that is not a claim of zero matrix computation inside predictions.

Total all three receipts:34fits/210rows/34lstsq,23.240047570001seconds, max79,036,416B;
564 nested train-port calls/1164 rows; pure cohort adds0.
Total traces21, training_cases calls4, dispatches111, probability8/36, routes168/690.
All prospective40-solve/240-row,120seconds/cohort and512MiB caps held. No scientific
cohort was attempted; numerical acceptance comprises bounded software fixtures.

Five changed source/test files:
```powershell
$checked = @('src/app/coupled_attribution.py','src/app/stream_rehearsal.py','src/app/context_joint_rehearsal.py','scripts/context_catalog_replay.py','tests/test_context_packet_rehearsal.py')
.venv/Scripts/python.exe -B -m ruff check @checked
.venv/Scripts/python.exe -B -m ruff format --check @checked
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent @checked
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux @checked
```
Ruff/format and both type configurations pass. An initial mypy invocation caught
two optional-episode narrowing errors before native work; the guard now captures
an explicit nonoptional episode tuple. A test API mistake (snapshot requested
after stream finish) was corrected by inspection before any numerical cohort.
No criterion weakened; failed checks and original artifacts remain.

Final four static commands took10.480212099996seconds in `statics.json`; static
peak RSS was not instrumented. Final lead receipt/hash/count/whitespace checks
pass; stage is6 files/9823 bytes after static receipts, no additional model work.

Lead self-reviewed the source, complete combined tree, role/ownership boundaries,
test procedure and receipts. Independent finite reviewer completed three source/
protocol reads BEFORE launch and one final receipt/hash read; no blocker, accepted
scoped evidence. Four readonly calls~1.0651167seconds command wall, no imports,
tests, models, writes or network; finite exception ended. This was independent
source/evidence assessment, not independent experimental replication.

Skipped other numerical joint controls, full baseline/math suites, raw-backed/
GPU/remoteCI/clean-clone/current-wheel/release checks, and full learned-controller
comparisons outside scope. #279 wheel predates this change. No new dependency,
install, download, delete, commit, push or global setting change. Inherited dirty
and ignored work, protocols, budgets, negatives and #287 spent study are preserved.

## Critical path and exact next action

This removes the demonstrated packet geometry prerequisite without inventing a
memory runner. The real next outcome is retention on the ordinary owned source
while continuing to serve. Compose this existing packet-capable joint rehearsal
at the ordinary periodic PRIVATE serving update boundary, using original arrived
packets and existing resource admission, with explicit consolidation and repeated
row costs. Inspect and reuse the existing runtime/action executor before adding
anything. Source preparation, actor publication and ACK histories must retain
their current guarantees; ACK still refuses unsupported consolidation histories.

Do not immediately repeat #298 or revive #287. Any small future comparison needs
a new frozen/reviewed fair exposure protocol that resolves complete-packet6+3
geometry without weakened controls or partial-packet padding. Reuse existing
matched BP/PC/neutralCPC/activeCPC where applicable and keep fresh-task probes
isolated from deployed/benchmark learners. This software result does not
establish an adaptation advantage or satisfy broader scientific/release parents.

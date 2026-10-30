# R7.5-CTX.LS.V.C — retained private learning alongside serving

Scoped #300 implementation and native engineering observation accepted,
2026-10-29. GitHub remains live tracking; full repository/research/release
requirements remain open. This follows accepted #299 packet geometry and #298's
negative ordinary-packet retention finding, without rerunning those comparisons
or #287 FAILED/SPENT. Original retired publication/repeated-update evidence is
unchanged; this advances actual learner behavior, not another capture increment.

## Working behavior and implementation

Public workflow:
```powershell
python -B -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --separate-context-heads --paired-serving retained_joint --output NEW.json
```

Ordinary owned LS source12/72 feeds the existing six-packet TRAIN recipe A:
old,new,new,old,old,new, delay1. `context_head_live` compiles exact original packets,
creates existing bounded `StreamRehearsal`, and uses existing periodic private
serving schedule. Final readout on the detached private candidate follows closure
of all serving workers. No promotion, ACK, probe/final data or new learner/runner.

`ResourceSharedRuntime` optionally binds the original packet/helper pair. The
existing gate reserves2 attempts before an actual wake, holding that lease in
`ExitStack` until runtime wake and at most one joint consolidation finish.
Reservations stay paid even when replay skips/fails. Empty polls reserve nothing;
paused/serving/resource/quota deferrals perform no fit/retention. An admitted
two-attempt bundle is not preempted between calls; serving reads its separate
original actor. Existing single-attempt/default paths keep prior behavior.

`_record_tick` declares replay consent only for this explicit ordinary TRAIN path;
default consent stays training-only. Actual native origin must exactly match
sample ID, features, targets, TRAIN role and training/replay permissions. Original
helper/packets/budget/clock, sharing-counter deltas and one current candidate stay
pinned. Detached prior receipt values plus raw owner identities refuse corruption
and replacement. Candidate advancement requires observed returned wake receipts
and exact committed fit/row deltas. No model history chain or deferred replay queue.
ACK retains its existing refusal of unsupported consolidation histories.

Affected production files: `src/app/resource_sharing.py`,
`src/app/coupled_attribution.py`, `scripts/context_catalog_replay.py`,
`scripts/run_bp_ticket_serving.py`, `scripts/context_head_live.py`,
`scripts/recurring_ticket_workflow_example.py`. New focused tests:
`tests/test_context_retained_serving.py`; [usage/protocol](../context-retained-serving.md).

## Native observation — exactly one invocation

Artifacts: `artifacts/runs/context-retained-serving-20261029/actual.json`,
`native-work.json`, `native.attempted`, `native.py`. Existing Linux3.12.3/NumPy1.26.4
producer, physical D checkout and D temporary paths; no installation/download.

- Terminal0, root/child complete, all6 TRAIN task readouts PASS,3 steps each:
  v1 invoice/password/legacy and v2 invoice/parcel/legacy; counts new2/2 old4/4.
  Observer emitted the sole endpoint after23 fits/144 rows; no later fit.
- Child6 ordinary wakes/27 rows +5 joint fits/45 rows =11 solves/72 rows.
  Initial replay skips; following five events each commit one9-row joint solve.
  Source12/72 +child11/72 =23 physical solves/144 fitted rows,23lstsq calls.
  Final candidate counters23/144. Replay rows include the repeated current rows.
- Original source/served actor/root ownership/owned-source check remain true;
  NumPy and Python RNG unchanged; actor generation0/versioncontext-ls-12.
  Serving stays available on the original actor; the learned candidate is private.
  Original live timeline succeeds15/24 across ordinary/failure scenarios; its
  failures remain recorded and are not replaced by the candidate's6/6 readout.
- All6 label packets consumed, no pending labels. Lead verified actual
  `action_snapshot.progress`:24 decisions,6 wake admissions,11 native attempts/
  completions,5 replay commits,45 replay rows, unknown-work flags false.
  Actual `sharing_snapshot`:12 paid reservations/limit12, one serving-active
  deferral, final idle gate. Reservation count is not returned fit count.
- 11.799593122seconds,64,090,112B sampled peak,25 teacher traces,54 task episodes
  (source24, serving24, terminal6),269 sandbox dispatches,194 typed calls,
  388 delegated head calls,158 affine calls/158 rows,6 probability calls/27 rows,
  268 routes/506 rows and36 unfitted reads.0 Torch/optimizer/latent/model-random
  work; no probe/final exposure or external actions. Original source budget30s
  from birth, observed last-minus-start10.641248220s; never renewed.
- Stage14 files/1,386,847B after native receipts; actual report1,365,377B, both
  under8MiB. Tests/static/native scripts and all failed receipts retained locally.

These are known TRAIN sandbox tasks under one fixed order, with extra replay and
diagnostic costs. This is useful private closed-loop behavior, not a matched
compute comparison, circadian advantage, unseen-task generalization, sustained
retention/recovery trajectory, deployed learned policy or release qualification.
Earlier #298 negatives remain; no baseline tuning or favorable run selection.

## Commands, checks, failures and resources

All project commands run from `D:\Dev\Repositories\Circadian`, process-local
TEMP/TMP=`D:\Dev\_Temp`; WSL TEMP/TMP/TMPDIR=`/mnt/d/Dev/_Temp`.

Final focused command (exact executable/argv also in each work receipt):
```powershell
.venv/Scripts/python.exe -B artifacts/runs/context-retained-serving-20261029/check.py artifacts/runs/context-retained-serving-20261029/windows-final-work.json tests/test_context_retained_serving.py tests/test_resource_sharing.py tests/test_adaptation_action.py
```
Windows affected checks98PASS, then an accidentally included unrelated Torch
native test failed at the observer boundary: overall command EXIT1,6.183082900s,
253,132,800B. Torch module imported; `_components` was blocked at function entry,
before model initialization/random/numerical work. Do not describe this invocation
as passing or Torch-free. Its paid18LS114rows are retained, not repeated.

Explicit pure compatibility selection:
```powershell
.venv/Scripts/python.exe -B artifacts/runs/context-retained-serving-20261029/check.py artifacts/runs/context-retained-serving-20261029/windows-adaptation-work.json tests/test_adaptation_action.py --deselect=tests/test_adaptation_action.py::test_native_no_update_and_bounded_wake_preserve_original_authorities
```
45PASS,0fits,1.186960100s,66,772,992B; no Torch import.

Linux final command uses the same three test files, explicit same native-test
deselection and `--capture=sys`:98PASS/EXIT0,18LS114rows,16.682773990s,83,578,880B.
```powershell
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-retained-serving-20261029/check.py artifacts/runs/context-retained-serving-20261029/linux-final-work.json tests/test_context_retained_serving.py tests/test_resource_sharing.py tests/test_adaptation_action.py --deselect=tests/test_adaptation_action.py::test_native_no_update_and_bounded_wake_preserve_original_authorities --capture=sys
```

Initial pure attempts:6PASS1FAIL2.831772200s75,165,696B and12PASS1FAIL2.519134700s
74,452,992B, both0 fits. First consent refusal triggered a secondary observer
exception; failure notifications now retain the primary refusal without redoing
origin verification. Second fixture expected raise instead of the workflow's
returned error report; assertion corrected, no production criterion weakened.
Third pure gate13PASS2.277664300s73,728,000B,0fits. Initial static checks caught
optional/generic state annotations and unused imports; fixed before native case.

Preliminary Windows/Linux51PASS each consumed36LS228rows, respectively5.294714800s/
81,264,640B and19.140974550s/84,938,752B. They preceded current-candidate/receipt
guards and do NOT qualify those later changes. Lead identified the candidate
proof gap; reviewer found aliased receipt pins. Both corrected before the native
case; final tests include corruption of the already-paid first receipt, with no
additional fits. Frozen original60-solve/400-row engineering allowance/receipts
remain. Prospective amendment6509971472 admitted final two cohorts plus original
one native case, total96solves/640rows. All fixture work72/456 plus native23/144
=95LS600fitted rows/95lstsq, within amended limits. No scientific allowance reset.
Each platform's injected joint-call refusal preserves2 paid wakes/9 fitted rows,
2 two-attempt reservations and9 charged replay rows despite no joint matrix fit.
Thus replay-exposure charges also include18 refused/unfitted fixture rows; fitted
row totals alone are not a complete cost account. No failed charges are refunded.
Per-cohort120s512MiB/native60s512MiB/source30s/task/storage limits unchanged.

Seven changed source/test files passed Ruff, format, default and Linux-platform
mypy; exact argv/output in `statics.json`, four commands4.206721300seconds,
static peakRSS uninstrumented. Lead checked complete source/procedure/receipts,
actual action/sharing fields, role/ownership flags, task transcripts and whitespace.
No further numerical work after the single native terminal result.

Independent finite reviewer:3 source/protocol calls completed BEFORE native
launch, fourth final evidence projection accepted scoped result. Read3 output
partially truncated; no extra read consumed. Final projection used absent
`adaptation_progress`/`sharing` keys, so final decision/reservation scalar fields
were NOT independently verified; lead checked the actual fields above. Other
native work/outcome/ownership/task/budget fields were inspected. Four readonly
calls~1.3seconds toolwall, no imports/tests/models/writes/network; exception ended.
This was independent source/evidence assessment, not experimental replication.

Skipped forbidden Torch native test after guarded failure, other unchanged
baseline/math/raw-backed/native/GPU/remoteCI suites, clean-clone/current-wheel and
release qualification. #279 wheel predates current work. No install, deletion,
commit, push or global setting changes; inherited dirty/ignored work preserved.

## Exact next implementation action

Connect this actual retained candidate to the EXISTING guarded publication path
and subsequent shared serving. Inspect `ServingPromotionController` and existing
drain checks; validate both6 wake receipts and5 joint consolidations/paid counters
before promotion, then demonstrate the served learned policy on the known new
and old task requests with exact rollback. Reuse this private result; do not
automatically repeat it, add more capture infrastructure, broaden ACK or rerun
#287. Any new execution needs a small prospective protocol; any comparative study
must resolve fair complete-packet exposure and include existing matched controls.
Broader model-family/R2 fresh-task/long-duration/scientific/release parents remain
open. This next outcome makes the learned behavior available to users, rather
than adding another private checkpoint or inference about unseen generalization.

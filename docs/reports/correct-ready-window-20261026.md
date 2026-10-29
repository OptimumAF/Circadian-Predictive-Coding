# R7.1-MG.A - proved-correct packet progress

Issue213 scoped SOFTWARE complete after source/procedure/result SELF-REVIEW,
not independent acceptance. Fullgoal/broad R2/R4/R5/R7/P7/P9/science/release OPEN.
No native experiment was authorized or run;212partial learning/negative utility
and format exception/209FAILED-SPENT remain unchanged. Existing86 runtime port
reused; its unfinished original native criterion is not completed by this work.

Observable behavior: acknowledge_correct=True lets a selected wait on healthy
ready TRAIN request a fresh proof under existing ActorShadowRuntime.acknowledge_ready.
After successful read-only proof, oldest packet is consumed with no optimizer
update, so a later arrived mistake is eligible on a subsequent opportunity.
DefaultFalse keeps prior waiting/schema/learning behavior. At most one wake OR
acknowledgement per ordinary tick, same clocks/credits/delays, no tail drain.

Fresh independent fork is restored from current pinned candidate inside the
existing exclusive lease, predicts only original source features, then compares
arrived labels. Exact packet/features/targets/versions/source/actor/clock/budgets
checked. Post-prediction source/actor equality is checked before acknowledgement
commit. Earlier0mistake scalar cannot authorize consumption; disagreement
returns no receipt, invalid/nonfinite/lost/candidate/source/actor/clock mutation
fails closed. Ordinary TRAIN/readyfeedback only; no managed/probe/sleep/replay/
renewal/capacity relaxation. Existing runtime/inbox ports are not edited.

Processed packet cursor differs from optimizer count. Applied observers get the
actual packet when fits are noncontiguous (e.g.0,2), native checkpoints advance
only on real fits. Version-bound ack receipts are pinned by value AND identity;
changed/equal-replaced prior receipts refuse more work and detached knownprefix
survives. Ledger container identity is pinned; prior history is also checked
inside fresh proof BEFORE another consumption commits. Lost return after actual ack preserves consumed receipt/zerooptimizer
charge/unknownreturn and refuses continuation. Every proof fork/predict/row/
attempt/return/unknown appears separately in acknowledgement.proof.

Reporting keeps legacy training_complete=allpacketsactuallyfitted; new
consumption_complete=allpacketsfittedORacknowledged. all_packets_fitted,
processed_packets, loop_processed_packets, unfitted_packets, pending_unapplied,
applied_packet_indices, acknowledged_experiences and known_acknowledgements
distinguish actual work and failed callbacks. Acknowledgements are never native
fits/replay/new budget authority; original appliedreceipt/budget conservation
remains. Capture failures stay failed without clearing live ledger corruption.

End-to-end existing run_controller_learning accepts acknowledge_correct=True
ONLY with retain_arrived=True and explicit validation_recipe. Original four-trial
fitter/warmup unchanged, flag applies equally to five validation windows, every
window still precedes endpoint games. Strict evaluator reference_applied_cues
checks exact committed-cue prefix length=nativeupdates-2, exact integer0/1cues,
9/18head geometry and original total-row/failedwork/zero-budget/source guards.
It is distinct from original four-arrival reference_catalogue_cues; no padding
or fictitious future fits. Inconsistent cue count/geometry/bools refuse before
physical IO. Only actual applied receipts determine cue metadata.

Literal integration: periodic/drift/fitted each fit packets0,2 (blue,red),
acknowledge1,3, end native4/54 with old/new2/2; two joint36rows18prior and two
extra fresh36rowproofs perarm. Noop/reserved-budget arms remain0fits/0acks/
pending4. Same fourtrialfit/originalsourceactor/budget limits. This shows pipeline
correctness with scripted games, NOT actual native improvement/generalization/
recovery/controller/CPC/efficiency or release readiness.212native1/2new limitation
remains the only actual observation for that unacknowledged schedule.

Changed existing sources: src/app/coupled_attribution.py; scripts/
minigrid_controller_learning.py; scripts/minigrid_controller_validation.py;
src/adapters/minigrid_policy_evaluation.py. No new learner/scheduler/CLI/dependency.
New tests:
```text
tests/test_reference_correct_ack.py  real stream/lease/receipt/proof fault cases
tests/test_minigrid_correct_ack.py    full literal fit/freeze/validation/readout
```
Why this: use existing stateful stream/lease, keep packet progress distinct rather
than force healthy optimizer work or add another orchestration framework.
Source-review.json proves other stream methods/constructor/budgetguards, other
learning methods/fitter prefix, other evaluator functions AST unchanged.
Original controller/objective/nativelearner/catalogue/runtime physics pins match
212. Default69MiniGrid and affected core reference/contextual/ready regressions
are included, math/baseline archive not rerun.

197distinct final cases qualify per platform: Windows192replacement suite plus
6fault/4completion/4ownership/4finalpositive affected =210passing instances;
Linux195initial+4ownership+4finalpositive =203instances, latest197allpass0skips.
Newfeature initial functional checks allpass; source review adds proof-only
postcheck and explicit consumption field, affected cases rerun under24caserepair.
Initial static3typing errors fixed (ackpins/testbox annotations/auxiliaryloopvar);
final8file Ruffformat/check/default+Linuxmypy PASS. Static54.4736706s plus
sourceproof.6206521s/touchedcache111026176B not growth(global inherited separate).

IMPORTANT failed tooling attempt: first Windows accounting aggregated absent
ctx.WORK.games key, failed first guarded fixture and poisoned teardown.192cases
collected/384XMLerror entries, no useful validation, parent21.5049949s. Exact
literal counters UNOBSERVED; do not invent0 or count those errors as passes.
Qualifier missing-key lookup repaired/maxfail1 added, not a new framework.
Prospective issuecomment6426582891 records original FULL fixture-cap reservation
for this failedattempt (not measurement), then2x cumulative literal envelope
solely to retain that conservative debt. New per-invocation limits, original
180scumulativebody210sparent512MiB/time/stage/native0 limits unchanged. One
replacement<=224case qualification plus <=24affectedproductchecks. Initial
460359Bfailedlog exceeded256KiB; preserve it as sole explicitly amended exception,
newlogs256KiB. Original failure/capbreach never retroactively passes.

Observed successful Windows work1496literalctors19558snap353scriptedfits3450rows
694predict7905rows111scripted4game243countercalls, plus reserved UNOBSERVED debt
4096ctors100000snap1024fits16384rows4096predict49152rows1024games1024countercalls.
Known+cautiousreserved remains within amended2x original envelope. Cumulative
body80.9396191s(includes21.5049949failed parent conservatively)/parent89.0459138s/
max sampledRSS120303616B. Linux1394ctors17977snap335fits3178rows630predict7111rows
103game239countercalls; body137.528857694s/parent172.0308925/RSS107659264B.
All successful per-invocation original caps held, no further allowance. Sampled
body RSS is after imports, not whole process-tree qualification. Genuine native
fit/predict/LAPACK/physics/oldscientificrawreader/probe/final/promotion0; optional
native imports forbidden. No installs/downloads/GPU/agents/CI/Gitmutation/reset.

Exact argv/outcomes in artifacts/runs/correct-ready-window-20261026/*-parent.json:
- .venv/Scripts/python.exe -B <stage>/supervise.py windows-1 .venv/Scripts/python.exe -B <stage>/qualify.py windows-1 (failed accounting)
- windows-2 same qualifier replacement full suite192PASS
- windows-3 tests/test_reference_correct_ack.py::test_should_fail_closed_on_bad_or_lost_fresh_proof (6PASS)
- windows-4 four explicit core/native completion IDs (4PASS; receipt fullargv)
- windows-5 ownership4PASS, windows-6 final core/MiniGridpositive4PASS; linux-2/-3 same4+4affectedPASS (full argv receipts). Total Windows18/Linux8 affected checks stay under24.
- linux-1 supervisor wsl -d Ubuntu -- env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout200s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B <stage>/qualify.py linux-1
Scope8file Ruffformat/check and mypy --explicit-package-bases --follow-imports=silent, also --platform linux.
Logs/XML/qualification/protocol/amendment/source review/statics/failuredebt/
parent/issue receipts retained. No raw scientific artifact was opened.

Fullsuite/cleanclone/currentwheel/CLIextension/browser/GUI/independent/native/
neural/robotics/comparative/scientific/final/probe/release checks skipped.
Historicalplans/log frozen perAGENTS, liveissue/tracker holds status; unrelated
dirty work and all negative/spent evidence preserved. No repository-wide claim.

ONE next action: freeze ONE small new native acknowledgement case through this
existing API, retain212unacknowledged outcome evidence, unchanged data/seed/
policy/objective/original150sreadout envelope, explicit extra proof work/ack
receipts/processed-vs-fit/exposure/game/capture caps. Future producer must use
allow_nan=False BEFORE freezing, not repeat212format exception. No automatic
scientific allowance, baseline tuning or209/212retry. Only after protocol/runtime/
source/procedure gates execute the newbehavior; actual benefit remains unproven.

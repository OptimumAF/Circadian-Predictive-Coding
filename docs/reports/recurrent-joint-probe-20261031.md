# Isolated fresh F and A return from joint recurrent TRAIN

Task: [#385 / R2.1-RNN.J](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/385).
Software and prospective native procedure have scoped independent PRE acceptance.
The ONE native process completed within budget with a NEGATIVE adaptation
result: neither F game succeeds, all five initially successful old tasks are
lost, and A return regains none. Scoped independent actual-result POST accepted.
Full repository/scientific/release goals remain OPEN/ACTIVE.

## Implemented behavior

The existing joint example now has explicit `--fresh-task-probe`. New
`scripts/recurrent_joint_probe.py` composes the existing F collector,
`run_fresh_task_probe`, A-return templates and qualified `ProbeRollouts`.
Existing evaluator/observer admit `joint_public_cue=True` with source128 only,
native144 after F16, native160 after A16 and exact per-head clocks. Both endpoint
plans contain F-blue/red probe_test and all A/B/C-blue/red TRAIN games. Source2
retains its original two-game/40-second/default mode; generic TRAIN monitors and
return still require the two separate complete A episodes. No new learner,
math, optimizer, controller, framework or dependency.

The caller protects original cold source/serving actor, live benchmark candidate
128, selected checkpoint, three RNGs, two baseline readouts, TRAIN receipts and
original limits/births/charges throughout. Probe and return use the same separate
learner with full optimizer continuation. Existing qualified endpoint witnesses
protect the two update budgets and prior paid reports. New joint readout requires
each episode's preservation proof, Boolean success, finite return, exact native
counts, fixed role/order and cleanup before admitting A return. Negative False/0
is valid. Corrupt or equal-replaced prior records refuse more work.

Final evaluation is unavailable. Probe_test labels remain diagnostic, not TRAIN,
promotion, replay, tuning, seed/dose/metric selection or deployed updates.
All probe/return inputs keep their roles; a full-episode reset is never replaced
by concatenating histories.
Use [the command guide](../recurrent-joint-example.md) for invocation and limits.

## Evidence reused and scientific scope

#384 joint native baseline already showed52/54 TRAIN actions and5/6 goals with
A-blue failure; both its original failed case and prospectively corrected case
remain charged. #383 LS control41/54 and original online work deferred, #373
simpler joint-affine F2/2 plus allold6/6, R2.1 and four-family matched controls,
sequential old-task failures and other original negative results are retained.
None is retrained here for a new comparison. Source2 failed136 is not resumed,
reconstructed, refunded or retrospectively made complete.

This new fixed question uses a different upfront joint TRAIN history and a
caller-owned live endpoint, not saved JSON weights. Fixed native H8/K7/raw148,
seed17/lr.03/momentum.9/meanCE and joint128+F16+A16 remain unchanged. Native F
has no pre-probe game control, so postF competence is not a native gain estimate.
Old retention uses actual initially successful tasks; initially failed A-blue
cannot be counted as forgotten or recovered without observed competence.
Observed regain requires success→loss→success, without causal attribution.
F retention after A return is separate. This is known-domain diagnostic evidence,
not reliable generalization, equal computation, CPC advantage or release readiness.

## Executed software checks

Physical cwd `D:\Dev\Repositories\Circadian`; TEMP/TMP/TMPDIR=`D:\Dev\_Temp`.
No native dependency imports, models, real source payload or physics in fixtures.
MetaPath blocks Torch/MiniGrid/Gym/pygame; pytest plugin autoload is disabled.

```powershell
& ./artifacts/runs/recurrent-joint-probe-20261031/run-step.ps1 -Step checks -Attempt first -Suite joint
& ./artifacts/runs/recurrent-joint-probe-20261031/run-step.ps1 -Step checks -Attempt reader -Suite reader
& ./artifacts/runs/recurrent-joint-probe-20261031/run-step.ps1 -Step checks -Attempt final -Suite final
```

|Run and exact node selection|PASS|FAIL|Body seconds|RSS bytes|Host seconds/PID|
|---|---:|---:|---:|---:|---|
|First:19new probe +20existing joint|36|3|19.952618400013307|98254848|20.0877194 /85940|
|Reader:19new probe +entire26existing source2 game cases|45|0|32.2299411999993|101822464|32.3699395 /12892|
|Final:23new probe +20existing joint +3source2 cases below|46|0|39.972372799995355|101318656|40.1101857 /28656|

The three final source2 cases are the two parameter variants of
`test_should_bind_only_declared_paid_state_and_target_free_plan` and
`test_should_compose_real_generic_probe_observer_and_return_without_deployed_updates`.
They are explicitly selected in `check_once.py`. The four new corrupted-F cases
explain19→23 new cases; changing the old-source selection26→3 explains45→46 total.
Across current node IDs there are69distinct cases. Cumulative130 executions =
127PASS/3FAIL/0skip across3runs92.15493240000797 seconds. Each run stayed below
64cases; total below192 and120seconds. All fixture slots are SPENT.

The initial three failures used TRAIN-only `_batch` in the new F fixture;
generic role validation refused every probe update. The fixture was corrected
to existing role-preserving `_packet`; no production role check was relaxed.
The finite independent review then found that aggregate preservation could
mask a missing per-episode proof. Joint-only checks and four actual generic
F16 corruption cases now refuse ALL A updates for false proof, integer success,
NaN return or wrong native count. Negative values and source2 defaults pass.

Metered scripted totals:3579constructors/3074updates27666rows/256teacher calls
2304rows/199live fake predictions. All genuine native work ZERO. Scoped static
errors retained: initial4fixture-alias Ruff errors and9annotations/type errors
(3.3909667s), later4meter-method assignments +1tuple annotation (.7763737s).
Final7-file Ruff/format and6-file mypy/PS parse all0/.8032205s; intervening
corrected checks0/.7320559s;6AST parses0/.0951377s. Total measured statics
5.7977545 seconds. No additional test run after the final combined tree passed.

Static commands used `.venv/Scripts/python.exe -m ruff format --no-cache`,
`-m ruff check --no-cache` on the example, helper, evaluator, observer, new tests,
checker and native meter; `-m mypy --follow-imports=silent --cache-dir
artifacts/runs/recurrent-joint-probe-20261031/mypy-cache` on the six excluding
the dynamic native meter. The meter was source-reviewed/linted/AST parsed,
not type-checked. Scoped tracked `git diff --check` returned0. Full suite,
Linux, clean-clone, packaging/install, browser, release and unchanged native
math/baseline reruns were skipped; previous scoped evidence is reused.

Exact final static invocations (the formatter applied the final formatting):

```powershell
./.venv/Scripts/python.exe -m ruff format --no-cache scripts/recurrent_joint_example.py scripts/recurrent_joint_probe.py src/adapters/minigrid_probe_rollout.py src/adapters/minigrid_policy_evaluation.py tests/test_recurrent_joint_probe.py artifacts/runs/recurrent-joint-probe-20261031/check_once.py artifacts/runs/recurrent-joint-probe-20261031/native_observe.py
./.venv/Scripts/python.exe -m ruff check --no-cache scripts/recurrent_joint_example.py scripts/recurrent_joint_probe.py src/adapters/minigrid_probe_rollout.py src/adapters/minigrid_policy_evaluation.py tests/test_recurrent_joint_probe.py artifacts/runs/recurrent-joint-probe-20261031/check_once.py artifacts/runs/recurrent-joint-probe-20261031/native_observe.py
./.venv/Scripts/python.exe -m mypy --follow-imports=silent --cache-dir artifacts/runs/recurrent-joint-probe-20261031/mypy-cache scripts/recurrent_joint_example.py scripts/recurrent_joint_probe.py src/adapters/minigrid_probe_rollout.py src/adapters/minigrid_policy_evaluation.py tests/test_recurrent_joint_probe.py artifacts/runs/recurrent-joint-probe-20261031/check_once.py
```

## Frozen native procedure and actual evidence

Protocol: `artifacts/runs/recurrent-joint-probe-20261031/NATIVE_PROTOCOL.md`,
frozen AFTER final software gates, BEFORE data/models/outcomes. ONE public CLI
process, original TRAIN bundle read once, then four fixed F teacher sources,
128warm+16probe+16return updates. Benchmark128 never receives probe updates.
Baseline6 learned games plus8 after each phase =22, separately from4source
teacher trajectories. All roles, metrics, seeds and directions frozen.

```powershell
& ./artifacts/runs/recurrent-joint-probe-20261031/run-step.ps1 -Step native
```

Whole application240s/768MiB, original TRAIN180s128updates and continuous
readonly200s/0update0replay. F16/A16 each180s768MiB, both born once before F;
selected endpoint80s0update0replay each. Original source2 still40s. Original
deadlines/counters/births remain enforced; no renewal or clock forgiveness.
Observer270s/768MiB, hidden owned host285s. Output2MiB/log256KiB/work128KiB/
whole stage64MiB including cache32MiB/all failures. No automatic retry or
successor after inability. All project outputs are physically on D:.

Planned caps:250native staged constructors/750private randn/326000values/
658optimizers,160CE/backward/SGD1440rows,44teacher predictions396rows,
22native sessions/352live predictions,556forwards2188unrolledrows. Up to
26environments/resets/seeds/64teacher decisions416physics steps52closes;
complete requires26distinct owners closed. Snapshot100000/clones1M/3Bscalars;
autograd.grad/LS/replay/promotion/final ZERO. Staging is discarded private
initialization, not250independent model seeds; named counters are not all FLOPs,
allocations, copies or IO. Original failed cases/costs remain separate.

The owned process PID80336 reached terminal0/host118.2153608seconds without
expiry. Observer117.78955260000657 seconds, application115.73482289997628 seconds,
peak369,688,576 bytes. All named attempts returned and every cap held. The same
process handle was polled through completion; there was no retry or second run.

## Complete negative native result

|Known task|Baseline joint128|After F16|After A16 return|
|---|---|---|---|
|A blue|Fail,16steps,0return|Fail,16steps,0return|Fail,16steps,0return|
|A red|Goal,9steps,.49375return|Fail,16steps,0return|Fail,16steps,0return|
|B blue|Goal,9steps,.49375return|Fail,16steps,0return|Fail,16steps,0return|
|B red|Goal,9steps,.49375return|Fail,16steps,0return|Fail,16steps,0return|
|C_train blue|Goal,9steps,.49375return|Fail,16steps,0return|Fail,16steps,0return|
|C_train red|Goal,9steps,.49375return|Fail,16steps,0return|Fail,16steps,0return|
|F blue,probe_test|Not run before probe|Fail,16steps,0return|Fail,16steps,0return|
|F red,probe_test|Not run before probe|Fail,16steps,0return|Fail,16steps,0return|

All22learned games completed with per-episode unchanged-state proof and cleanup.
The five baseline goals terminated; the other17games truncated at16. All26
distinct owners, including the4source collectors, closed. All4teacher source
games succeeded in9steps/.49375, with36teacher decisions/steps. Native317
predictions/steps are distinct from those36teacher steps. All16post-adaptation
games failed, including both new tasks and all old tasks at both endpoints.

Retention of initially successful old tasks is0/5 afterF and0/5 afterA. Old A-red
has success→loss→loss; A-blue never has an observed success. No old recovery,
F competence, or F retention premise is established. This is a complete negative
outcome, not a failed experiment, incomplete game, timeout or null preservation.
The benchmark itself remains128 and unchanged; these losses are only on its
disposable diagnostic fork. Source, serving actor, benchmark and Torch/NumPy/
Python RNG preservation flags are all true.

Teacher improvements do not contradict that native failure:

|Phase/relative updates|F-blue CE / correct|F-red CE / correct|
|---|---|---|
|F0|1.283837 /6of9|1.228605 /6of9|
|F4|.838632 /6of9|.754037 /6of9|
|F8|.922319 /5of9|.646288 /7of9|
|F16|.608722 /6of9|.493936 /7of9|
|A0,paid again|.608722 /6of9|.493936 /7of9|
|A4|.632913 /6of9|.520305 /7of9|
|A8|.464349 /7of9|.645874 /7of9|
|A16|.686579 /6of9|.514314 /7of9|

Fresh teacher accuracy12/18→13/18→13/18; CE improves under F updates, but none
of the F sequences becomes exact. Old A teacher accuracy16/18→14/18→16/18;
final A-blue/red both8/9, CE.4148297002709749/.34089340428606657. The initial
A-blue/red CE was.39054208126886075/.3408221393924048 and accuracy7/9,9/9.
Joint128 same-row allold correctness remains52/54. Generic score DTOs contain
CE/accuracy, not a separate exact_sequence_accuracy field; the predeclared
nine-row hard-label exactness is derivable from returned per-decision correct
flags (equivalently9of9 accuracy). F exactness is0/2 throughout; A exactness
1/2 beforeF and0/2 afterF/afterA. Original decision records remain in actual.json;
no metric, denominator, seed, dose or direction was changed to favor the result.

## Actual costs and original authority

|Observed named work|Count|
|---|---:|
|Native batch /CE/backward/SGD|160 /160/160/160|
|Training rows,warm /F /A|1440,1152 /144 /144|
|Staged native constructors|250|
|Private randn calls /values|750 /326000|
|Optimizer constructions|658|
|Teacher predictions /rows|44 /396|
|Native sessions /live predictions|22 /317|
|Native forwards /unrolled rows|521 /2153|
|Snapshot ports|68206|
|Tensor clones /cloned scalars|37345 /9694226|
|Environments/resets/localGymseeds/closes|26 /26 /26 /26|
|Teacher decisions /teacher physics|36 /36|
|Total physics|353|
|Original TRAIN bundle reads /bytes|1 /94705|

All named attempts returned. Source public constructor1 initializes two heads;
the250staged constructors include all discarded train/teacher/live initializations
and are not independent seeds. Derived zero biases3750values are not random
draws. `public_train` train_episode port0 is honest: existing stream/generic
call train_batch directly. Autograd.grad/LS/lstsq/replay/promotion/final work0.
No assertion that these named ports count every FLOP/allocation/copy/IO.

Native clocks: warm128/1152/perhead64/576, Fendpoint144/1296/perhead72/648,
Aendpoint160/1440/perhead80/720. The original128TRAIN receipts remain a0/c0,
observation0..127/arrival-and-apply1..128. F and A budgets each charge16updates/
144rows with zero replay. Budget clocks are separately preserved:

|Authority|Birth|Last observed clock|Charged updates/replay|
|---|---:|---:|---:|
|Original TRAIN|247053.4385451|247056.9641447|128 /0|
|Continuous original readonly|247053.5548351|247167.9064403|0 /0|
|F probe|247065.8971329|247167.8016094|16 /0|
|A return|247065.8971403|247167.8016383|16 /0|

F/A budgets were born once before F; last observations include both endpoint
readouts rather than restarting clocks per phase. Endpoint readout elapsed
43.770890800020425/55.23400860000402seconds, both<80, exact0updates0replay and
timing/binding proofs true. They include guards/copies/physics/inference, not
an isolated inference timing. The continuous original readonly lifetime is
114.3516052seconds<200. No boundary, birth, counter or permanent cost was renewed.

Reserved F training/test arrays193104bytes; two old A monitors and two return
templates add21456bytes each, total236016numeric bytes, plus42912bytes for four
collected source arrays. Copies, encoded trajectories, Python/native state,
snapshot guards and reports are additional and included in whole observed RSS.
Full report1,133,438bytes, native work2306/log1833/stderr0; original TRAIN SHA
remains the frozen hash in NATIVE_PROTOCOL.md. A saved-only scalar projection
read that new report once,0model calls/.1582645seconds/analysis-process peak
133050368bytes; it did not reread historical artifacts or select another model.

## Review and next outcome

Lead combined source/procedure/result self-review distinguishes the passed
plumbing and complete negative task outcome. Independent finite source/procedure/
actual-result POST ACCEPT used4reads/.7590326seconds; no numerical replication.
The earlier broad review
query was truncated, so a targeted second read completed the missed rollout/
failure semantics. A verbose teacher excerpt was elided from displayed POST
output; the native rosters/costs/clocks/report and exactness interpretation were
inspected. No extra read, native computation or new model authority. No old source payload,
earlier failed capture or model execution was performed by the reviewer.
The finite exception ended; no allowance carries to #386. All six #385 scoped
criteria are evidenced, including complete execution with honest negative
interpretation. This does not complete reliable adaptation or broader parents.

Final source-binding/storage self-review (`c8d78b`, matching pre-native `3a91cf`):

|File|SHA256|
|---|---|
|scripts/recurrent_joint_example.py|5be90fefa3c5a7eb58542eb5f5df5c72378346d04b26d07d5c3d4fae14ec18e3|
|scripts/recurrent_joint_probe.py|bcbc16f82aaea7984eccb9b2f54998880e2605f0b89ee655988ce3d230b8a4cd|
|src/adapters/minigrid_policy_evaluation.py|3ae52cbae20803c367e47e29bfe7a56c745dbca7c9d5c641b002f0a1dfc4ad9e|
|src/adapters/minigrid_probe_rollout.py|037b8303b848ac0c8d8439b3603c1621827ac2c6c805b4f2117dd7d53b8319ca|
|tests/test_recurrent_joint_probe.py|871d39c1a73d5a05dff96f358bc4691151b608860d1e77d40103a7547a2ea29b|

Stage39files18,941,679bytes<64MiB; cache17,760,485bytes<32MiB. All failures,
attempts and temporary test captures retained. Scoped tracked diff check0 with
existing LF/CRLF advisories. HEADd9119e8 unchanged, new scripts/tests/docs remain
local untracked; no commit/push/publication or clean-clone claim. Inherited
tracked/untracked/ignored work, old failures and unrelated source edits preserved.
GitHub closure receipt: [canonical #385 evidence](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/385#issuecomment-6551421310).
Issue385 closed with all6scoped criteria evidenced; parent18/21/23 remain open,
comments6551438873/6551439097/6551439299. #386 all6criteria unchecked/no allowance.
Final archival consistency is lead self-review; no extra independent batch or
model execution after the accepted result. The full goal stays ACTIVE.

The negative result makes conventional retention controls relevant, but does not
justify a new CPC algorithm or outcome-tuned F dose. Existing raw recurrent
recent/coverage replay and generic StreamRehearsal already exist; existing
old-A probe rehearsal is an invariant-history/native-family recipe, not this
raw148joint128 probe. Reconcile those actual seams and matched extra-F exposure
before any new comparison or mechanism. Simpler accepted joint-affine retention
remains a valid preferable control. No new model allowance is carried forward.
The substantive successor [#386 / R2.1-RNN.J.R](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/386)
is PLANNED with all criteria unchecked: reuse the existing fixed-bank/generic
probe components for a conventional rehearsal versus matched extra-F reference,
not a new replay algorithm. Exact membership, opportunity matching and native
clock admission must be reconciled from current source and frozen before work.
Original F16 remains the preserved unmatched-exposure reference; it is not
rerun as an extra arm. Next implementation action: inspect
`StreamRehearsal(fixed_training_episodes=..., wake_role='probe_train')` and raw
recurrent replay phase/count contracts, choose the minimal paired composition,
then freeze its modelZERO protocol. No new fixture or model allowance exists.
The initial planned R5.3-RNN.J label was corrected before implementation:
actual roadmap parent24 makes R5.3 specifically CLM contrastive head consolidation,
not raw recurrent replay. #386 retains all six criteria/unfinished work under
the appropriate R2.1 parent; no failed experiment or spent budget was renamed.
Reliable R2.1/R4.2 acquisition/retention/recovery, heldout/recurrentRL/R4.7/R4.8,
Linux/science/release and the full repository goal remain OPEN/ACTIVE.

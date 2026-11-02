# Joint native recurrent BC: software qualified, five of six TRAIN games succeed

Scope: [#384 / R4.2-RNN.JBC](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/384).
This checkout now has a usable `scripts.recurrent_joint_example` command through
the existing learner, stream and game evaluator. Final focused checks pass.
The corrected native case completed and received scoped independent actual-result
acceptance. The first native case failed and remains permanently charged.
No repository-wide, clean-clone, packaging or scientific-benefit acceptance.

The prior repeated-update increment and R2.1/matched BP/PC/neutral/active controls
were already accepted at their recorded scope and were not reconstructed or
rerun. #383's frozen-feature joint LS diagnosis remains negative (41/54 correct);
its online integration is deferred. The accepted simpler joint-affine control
is reused. Existing sequential readout acquisition/return and ten old-task
retention failures remain unchanged. #382 visual and #378 Linux gates stay open.

## Behavior and fixed recipe

New files are `scripts/recurrent_joint_example.py`,
`tests/test_recurrent_joint_example.py` and [usage](../recurrent-joint-example.md).
README links the command. No native learner, optimizer, evaluator, stream,
snapshot policy, ordinary default or dependency changed for this increment.
The relevant evaluator already has inherited checkout edits; those are preserved.
HEAD remains `d9119e80926a139a2f38460564d7eab1b6d2a2b0`; this is a dirty local
source qualification, not committed publication or the older installed wheel.

All original A-blue/red, B-blue/red, C_train-blue/red TRAIN demonstrations are
available upfront. Exactly 128 complete nine-row episode presentations repeat
that order by index i modulo 6: A44/B42/C42 presentations, 396/378/378 rows.
There are 54 unique TRAIN
rows and 1,152 exposures; each cue receives 64 updates and 576 rows. New arrival
IDs and label delay 1 use ordinary existing `AttributionStream` receipts.
Neither repeated presentation nor upfront availability implies independent data,
retained-history replay consent or a chronological adaptation study.

Original raw148/H8/K7, private seed17, learning rate .03, momentum .9, mean CE,
full-episode hidden reset and ordinary SGD remain unchanged. No class weights,
LS fit, replay, new controller, early stopping, dose tuning or seed selection.
Teacher readouts are fixed at updates 0 and 128 on all six TRAIN packets;
native games run only at 128, all seven first-argmax actions, seed17/max16.
The evaluator's `recurring_training=True` flag selects six-context readouts;
the outer definition explicitly says `joint_train_recurrent_bc128_v1`. This
is not the original A32/B32/C32/A32 training sequence.

Original source and serving actor remain cold. Candidate native clocks reach
128 updates/1,152 rows and 64/576 per head. The caller checks original owner/
policy/state, input values, three RNGs, limits, clock identities, births,
permanent counters and receipt prefixes before and after callbacks. Prior
readout roots, containers and values are sealed with existing paid evidence
guards before every next update. The TRAIN stream finishes before postfit
games. Only two successful readout records are retained. Replaced or corrupt
prior evidence refuses work; no unlimited success chain or renewed budget.
Final evaluation, probe data, tuning and promotion have no path into this run.

## Executed outcome

Teacher scores are on the same TRAIN rows, separately from target-free games.

|Task|CE before|CE after|Correct before|Correct after|Postfit game|Return|Steps|
|---|---:|---:|---:|---:|---|---:|---:|
|A blue|1.978060|.390542|0/9|7/9|Fail, truncated|0|16|
|A red|1.951843|.340822|0/9|9/9|Goal|.49375|9|
|B blue|1.918841|.343346|2/9|9/9|Goal|.49375|9|
|B red|1.926203|.350333|1/9|9/9|Goal|.49375|9|
|C_train blue|1.901253|.351757|1/9|9/9|Goal|.49375|9|
|C_train red|1.901579|.362806|2/9|9/9|Goal|.49375|9|

Total correct: 6/54 before, 52/54 after. Exact teacher sequences: 0/6 to 5/6.
Games: 5/6 goals; A-blue's negative result remains. All six episodes completed
with legal recorded action indices, full trajectories, unchanged learned-state
proofs, and six distinct closed environment owners. The five goals terminated;
A-blue truncated at 16. No prefit games were run, so native improvement is not
estimated. This is partial conventional joint competence, not full joint fit,
causal retention, recovery, held-out transfer, equal compute or CPC advantage.

## Failure and prospective correction

Original stage: `artifacts/runs/recurrent-joint-bc-20261031/`. Its ONE native
case is FAILED/SPENT: terminal1 after all128 paid updates/1,152 rows and six
prefit score packets/54 rows. Final readout refused before scoring or games:
`recurring memory readout requires its unchanged native0 source`. The caller
incorrectly passed a reader restored to128. All six final games are `not_run`,
not failures; aggregate final preservation is null, not true. This is an
integration failure, not negative learning evidence. Both lead and initial
independent PRE review missed this existing evaluator contract. Fake evaluator
checks were too permissive. No weights were saved, so the JSON cannot resume
the missing trained endpoint.

The failed caller/tests were copied before correction; their SHA256s remain in
the successor protocol. The minimal caller fix removes the redundant reader
fork/restore and passes the original cold source plus the separate trained
checkpoint. The existing evaluator creates its own scorer fork and restores
that checkpoint. Its native0-source check was not relaxed. Existing scripted
readout checks now require a cold0/0 source at both0/128.

Before any further model work, issue comments6550911044/6550940464 and
`artifacts/runs/recurrent-joint-bc-source-fix-20261031/PROTOCOL.md` froze a separate
ONE-case allowance, same seed/data/dose/metrics/math/caps. Its observer is
byte-identical to the first; only the launcher stage path changes. Cumulative
two-case limits were explicit. This corrected-code rerun used a new paid process;
it did not restore saved weights, refund work, or renew old clocks. The original
spent command and failed outcome remain unchanged.
No postfit performance had been observed when the correction was frozen.
No further automatic case is admitted. All original failures/costs are kept.

## Commands, checks and costs

All commands ran at physical `D:\Dev\Repositories\Circadian`, with TEMP/TMP/
TMPDIR=`D:\Dev\_Temp`, no bytecode or pytest plugin auto-loading. No new
dependency, download, GPU, WSL, browser, remote CI, publication or deletion.

```powershell
& ./artifacts/runs/recurrent-joint-bc-20261031/run-step.ps1 -Step checks
& ./artifacts/runs/recurrent-joint-bc-20261031/run-step.ps1 -Step checks -Attempt final
& ./artifacts/runs/recurrent-joint-bc-20261031/run-step.ps1 -Step native
& ./artifacts/runs/recurrent-joint-bc-20261031/run-step.ps1 -Step checks -Attempt source-fix
& ./artifacts/runs/recurrent-joint-bc-source-fix-20261031/run-step.ps1 -Step native
```

The first two scripted runs passed18/20 cases, body5.026286300009815/
6.548782300000312 seconds, peak87,826,432/87,429,120 bytes, host5.1686175/
6.6757376 seconds, PIDs58172/65712. Final combined caller/fixture run passed20,
body6.62945090001449 seconds/86,642,688 bytes, host6.7710864/PID44436.
Cumulative58 passing executions,20 distinct cases,3 runs/18.204519500024617
seconds,803 scripted constructors/771 updates6,939 rows/16 readouts. All
engineering fixture slots are SPENT. Genuine Torch/model/solver/physics/source
payload ZERO in fixtures; imports of native dependencies were blocked.

Behavior checks cover exact episode schedule/IDs/exposures, role/order/partial
input refusal before model admission, real stream128 scripted updates, source/
actor/RNG preservation, preceding equal-replaced evidence, permanent counter
mutation, paid-output refusal, failed postfit work retention, missing/NaN prefit
metrics before training, and negative/invalid native outcome qualification.
No failed or skipped tests in these runs. The initial type check found two
missing annotations and returned1/3.2417348s; fixed and preserved. A prior issue
comment's '.3s' was a typo, corrected in the later receipt.

Scoped statics used the existing venv:

```powershell
./.venv/Scripts/python.exe -m ruff check --no-cache scripts/recurrent_joint_example.py tests/test_recurrent_joint_example.py artifacts/runs/recurrent-joint-bc-20261031/check_once.py artifacts/runs/recurrent-joint-bc-source-fix-20261031/native_observe.py
./.venv/Scripts/python.exe -m ruff format --check --no-cache scripts/recurrent_joint_example.py tests/test_recurrent_joint_example.py artifacts/runs/recurrent-joint-bc-20261031/check_once.py artifacts/runs/recurrent-joint-bc-source-fix-20261031/native_observe.py
./.venv/Scripts/python.exe -m mypy --follow-imports=silent --cache-dir artifacts/runs/recurrent-joint-bc-20261031/mypy-cache scripts/recurrent_joint_example.py tests/test_recurrent_joint_example.py artifacts/runs/recurrent-joint-bc-20261031/check_once.py
```

Final four-file Ruff/format, three-file mypy and PowerShell parser all0/
.7523395 seconds. Earlier corrected checks/AST/PS .7922285 seconds and
cumulative-meter helper types .4954969 seconds also0; measured statics total
5.2817997 seconds including initial errors. The thin native observer is
linted/formatted/source-reviewed, not independently type-checked. Full suite,
Linux, clean clone, package build/install, browser and release checks skipped;
unchanged native math/baseline/R2.1 suites reused at their accepted scope.

|Observed work|Original FAILED/SPENT|Corrected complete|Cumulative paid|
|---|---:|---:|---:|
|Native batch/CE/backward/SGD updates|128|128|256|
|Training rows|1152|1152|2304|
|Native constructors including staging|136|154|290|
|Private randn calls / values|408 / 177344|462 / 200816|870 / 378160|
|Optimizer constructions|398|434|832|
|Teacher calls / rows|6 / 54|12 / 108|18 / 162|
|Native forward calls / unrolled rows|134 / 1206|201 / 1321|335 / 2527|
|Live sessions / predictions / physics|0 / 0 / 0|6 / 61 / 61|6 / 61 / 61|
|Snapshot ports|3560|11024|14584|
|Tensor clones / cloned scalars|7161 / 1893864|12673 / 3316722|19834 / 5210586|
|Observer seconds|6.600664700003108|14.524496199999703|21.125160900002811|
|Host seconds|7.0015262|14.9322833|21.9338095|

All named attempts returned and caps held. These counters exclude unknown
native allocations, all FLOPs, some validation/copy/IO work; they are not a
total compute metric. Per-stage temporary constructors are discarded private
initializations, not hundreds of independent model seeds. Derived zero biases
are2040/2310 values, not additional random draws. `public_train`/train_episode
port0 is honest: the ordinary stream calls train_batch, counted directly.
Autograd.grad/LS/solver/replay/probe/final/promotion work all ZERO.

Original observer RSS351,825,920 bytes; corrected observer354,062,336 and
application354,111,488 bytes (independent sampling schedules). Original body
4.658602199982852 seconds; corrected12.512307500001043. Owned hosts PID70540/
79204 returned1/0 without expiry, observed to terminal on their same handles.
Original TRAIN birth245013.4440068→245016.885942/128charges/0replay, readonly
245013.5595482→245016.9295369/0/0 remain permanent. Corrected TRAIN
245258.8579549→245262.3404219/128/0, one continuous readonly
245258.9738262→245270.117492/0/0. New births belong to the separate process.
Original128 chronological receipts and corrected128 receipts retain a0/c0,
observed ticks0..127, arrivals/applies1..128; no final-scoring forwards in TRAIN.

Each native process reads one original94,705-byte TRAIN bundle, SHA256
`5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508`.
That is two paid reads189,410 bytes, not reuse without cost. Corrected output
235,698 bytes/work2224/log1766/stderr0. Original actual50,910 bytes/work2211/
log1853/stderr0. MiniGrid3.1.0/Gymnasium1.2.3/pygame-ce2.5.7 are in actual
environment metadata. All six environments closed in the complete case;
the failed case constructed none.

Source SHA256 after correction:
`eb14c62b0b819bbf21267102d59ad64da366aecd12cdb75f0e8e097785ff437d`;
test `4359716e5d182bceb7584c7e77b80f4404ce219171d8b847f51f4df0d0220d1c`;
both observer copies `e450a3e381e512efcbfe657ab77b8ff4fcf819fcf88e9c0fdaf9f854739f2d40`.
Original protocol's source-only constructor estimate60fake was corrected to
1024 staged fake calls BEFORE fixtures; native148 estimate corrected to154
BEFORE any model, accounting for six session forks. Neither correction erased
an attempt. Cumulative metering preserves every prior SPENT-linked receipt.

## Critical path after this increment

Do not add another checkpoint, provenance, fourth-update, wheel or capture
increment. Native joint CE now supplies partial known-TRAIN competence, so a
substantive next question is whether this learner can adapt to a fresh task
while retaining its already successful tasks and recover old behavior on return.
R2.1 generic disposable probes and public-cue compatibility already exist; do
not recreate that framework or repeat matched baseline work.

Next bounded outcome: use the existing disposable F16/A16 probe on a caller-owned
joint128 checkpoint, with old TRAIN monitors and target-free F/A game readouts.
Keep the five initial native successes and A-blue failure separate when defining
retention/recovery. Inspection found a concrete dependency: the existing native
public-cue probe readout deliberately accepts source_updates2 only; it cannot
yet read a source128 endpoint. Reconcile generic monitor/return envelope limits
and actual current runner before adding narrowly declared source128 support as
part of that outcome, preserving source2 and original failed136 evidence.
Generic TRAIN monitors admit at most two batches, and fixed return requires
exactly two corresponding templates. Preserve that default; use the existing
endpoint readout for bounded full oldABC observations rather than concatenate
recurrent episodes or silently widen the monitor contract. The substantive
successor is [#385 / R2.1-RNN.J](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/385),
PLANNED with all criteria unchecked.
Use a live endpoint callback through the existing stream/composition; saved
JSON is not a model. Freeze/check a small modelZERO interface first, then a
separate local learning protocol. No new model allowance from this handoff.

This is a new diagnostic from a different initial TRAIN history, not a retry of
the failed two-warm-update probe or evidence that dose changes fixed it. Reuse
accepted simple affine/four-family controls without claiming equal compute.
All broader R4.2/R2.1 reliable adaptation, recurrent RL, held-out layouts,
R4.7/R4.8, Linux, science/release and the full goal remain OPEN/ACTIVE.

## Review and scoped disposition

Independent reviewer `/root/recurrent_update_review` inspected source/procedure,
the corrected caller contract, exact final fixture receipt, native results/costs
and this report/usage. Final scoped ACCEPT:4 read batches/.6671733 seconds,
no model execution or numerical replication. Initial PRE missed the cold-source
handoff defect; original failure and corrective PRE/POST remain explicit. One
guessed fixture metadata filename was absent; the correct checks-source-fix.json
and log were reconciled at final POST. Review notes and timings remain in the
original stage. The finite exception ended; no allowance carries into #385.

Lead source/procedure/result self-review is separate. All five #384 scoped
criteria are evidenced: composition, fixed recipe, final checks, prospectively
budgeted executed qualification including original failure, and handoff.
Partial competence is accepted as an observation, not six-task learning success.
The requested concise archival plans/log updates preserve unfinished parents;
GitHub remains the live tracker. Final storage totals and issue receipts are
recorded in the handoff. No additional models, probes or old captures were read.

Final storage self-review (`be8983`): original stage39files17,339,538 bytes,
successor11files253,273 bytes, combined17,592,811 bytes below the prospective
72MiB allowance. Original cache17,223,909 bytes is below32MiB; individual stages
are below64/8MiB. Failed captures/preimages remain retained. Scoped `git diff
--check -- README.md DEVELOPMENT_PLAN.md RESEARCH_ROADMAP.md docs/development-log.md`
returned0; only existing LF/CRLF advisories. Final source/test and failed-preimage
hashes match the recorded qualification. New source/tests/docs remain local
untracked files; inherited tracked/untracked/ignored changes were preserved.

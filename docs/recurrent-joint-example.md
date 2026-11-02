# Joint native recurrent TRAIN example

The exclusive `--reward-training` mode connects the actual joint128 endpoint
to the existing sampled-reward runner, with a distinct momentum-only control:

```powershell
python -m scripts.recurrent_joint_example --reward-training --describe
python -m scripts.recurrent_joint_example --reward-training --output joint-reward.json
```

Budget native execution before running. This mode skips the ordinary prefix
readouts, performs eight TRAIN reward updates and eight zero-loss control
updates on the same observed trajectories in separate candidates, then reads
frozen BC, reward and momentum endpoints on six known tasks each. Full source,
actor, benchmark and optimizer state remain protected. Readout records have
explicit comparison-arm and warm/continuation-objective labels. No final/probe
data or outcome-based selection enters training. The
[complete software integration](reports/joint-recurrent-reward-integration-20261031.md)
has executed literal-port checks. The [single native attempt](reports/joint-recurrent-reward-native-20261031.md)
completed all144 updates but failed its serialization-work cap during readouts.
The comparison and final preservation remain unqualified; its allowance is spent,
with no retry or reward-versus-control ranking.
Existing default, probe, rehearsal and weighted commands retain their behavior.

From the physical checkout with the existing optional Torch/MiniGrid runtime:

    python -m scripts.recurrent_joint_example --describe
    python -m scripts.recurrent_joint_example --output new-joint.json

Describe reads no source data and constructs no model. Output must be a new
file; an existing file refuses before training. Direct temporary outputs to D:.
The current local wheel predates this example; use the checkout command above.

The command composes the existing `AttributionStream`, public-cue recurrent
learner and native evaluator. All six original nine-step TRAIN demonstrations
are explicitly available upfront: A-blue/red, B-blue/red and C_train-blue/red.
Exactly 128 complete-episode presentations cycle through that order. Repeated
presentations have distinct chronological IDs and label delay 1; they are not
independent data or replay of another learner's retained history.

This produces 1,152 row exposures from 54 unique TRAIN rows: A contributes 396,
B 378, and C_train 378. Each cue head receives 64 updates and 576 rows. The
original 148 inputs, hidden width 8, seven actions, seed 17, learning rate .03,
momentum .9 and mean categorical CE remain unchanged. The existing native learner resets hidden state for each
complete episode. No additional optimizer, controller or orchestration exists.

Teacher scores on all six TRAIN tasks are recorded before training and after
update 128. Six target-free native games run only at 128, each with seed 17
and at most 16 steps, using first-argmax over all seven action probabilities.
The evaluator receives
the original cold source and a separate trained checkpoint; it owns its private
scorer. Neither source nor serving actor trains or receives a promoted policy.
Scores, native success/return, and costs are separate observations. No prefit
games means postfit success is not an estimated native-game learning gain.

The fixed joint pool differs from a chronological new-task/old-task/return
experiment. This command cannot establish retention, recovery, unseen transfer,
matched compute, or circadian advantage. Final evaluation and isolated probe
data are unavailable to this recipe; no outcome selects updates or promotions.

One 128-update, 180-second TRAIN budget and one zero-update, zero-replay,
200-second readout budget span the process without renewal. The application
limit is 240 seconds and 768 MiB. The same TRAIN stream finishes before postfit
games. Every next update validates prior paid
readout evidence, receipt prefixes, original source/actor state, RNGs, budget
limits, births and permanent charges. Equal replacement or corruption refuses
more work. Only two readout records are retained; failures keep paid receipts
and partial readouts rather than claiming completion.

The report is bounded to 2 MiB and is evidence, not serialized model weights,
resume authority, or a renewable budget. Native staging constructs temporary
owners; report training counters alone do not measure that cost. A qualification
must account for staging, teacher/live inference, guards/copies, physics, time
and memory. The frozen local qualification and every failed attempt are recorded
in [the evidence report](reports/recurrent-joint-bc-20261031.md).

For changes, extend this existing composition and its focused behavior tests.
Keep source/checkpoint isolation, complete-episode resets, fixed comparisons,
negative outcomes and sealed evaluation intact. Freeze a separate small
protocol before changing the data schedule or running another model case.

## Isolated fresh-task and return diagnostic

The explicit opt-in extends this same command:

    python -m scripts.recurrent_joint_example --fresh-task-probe --describe
    python -m scripts.recurrent_joint_example --fresh-task-probe --output new-probe.json

After the original joint128 training and all six known-TRAIN games finish, the
existing collector produces two F `probe_train` and two F `probe_test` episodes.
All four teacher trajectories, roles, steps, cleanup and costs are recorded.
The generic probe copies the live joint checkpoint onto a separate learner.
It performs the fixed F16 then A16 return, preserving optimizer continuation.
The benchmark stays at128 updates; the disposable native endpoints are144 and160,
with72/80 updates per cue. Logical probe/return counters each run from0 to16.
No saved JSON is treated as weights or authority to resume an experiment.

Teacher metrics on F and two original A monitors are recorded at relative
0/4/8/16 in each phase. The default two-A-monitor contract is unchanged; separate
recurrent episodes are never concatenated into one score sequence. Qualified
native readouts observe F-blue/red `probe_test` and all A/B/C-blue/red `train`
tasks after EACH phase. The mixed readout keeps each episode's role explicit.
Each game uses actual public observations, a new recurrent session, all seven
first-argmax actions, seed17 and at most16steps. No teacher acts during these
learner games. F competence after A return and all old-task failures remain
visible; success on return alone does not prove recovery.

Probe and return each have one original16-update180-second768MiB budget, born
once before probe updates. Each eight-game endpoint has a separate80-second,
zero-update/zero-replay readout budget. The original joint application240-second
and continuous200-second readout deadlines still apply throughout; no clock or
charge is renewed. The source2 probe's original40-second/two-game mode is unchanged.
Each episode's preservation proof, Boolean success, finite return, exact paid
native counts, role/order and cleanup are required before more work. Corrupted
or equal-replaced prior endpoint evidence refuses the next phase. Source,
serving actor, benchmark, random states, original budgets and baseline receipts
remain protected. Only the disposable learner receives F/A updates.

The output's `adaptation` field separates collection, generic probe/return,
native endpoint records and original phase budgets. Inspect nested statuses;
paid partial work remains when a later phase or game fails. Original failures
and other qualified controls are not reconstructed or rerun by this command.
This is a fixed known-domain diagnostic, not final evaluation, deployed learning,
reliable generalization, a matched-compute comparison, or circadian advantage.
Qualification and costs are in [the probe evidence report](reports/recurrent-joint-probe-20261031.md).

## Matched old-TRAIN rehearsal diagnostic

The separate explicit option compares conventional replay with extra F exposure:

    python -m scripts.recurrent_joint_example --compare-rehearsal --describe
    python -m scripts.recurrent_joint_example --compare-rehearsal --output new-comparison.json

Choose either this option or `--fresh-task-probe`. The comparison uses the same
existing stream runner, original joint128 checkpoint and generic isolated probe.
It runs two disposable arms in fixed order: F48 then A16, and F16 plus32 old-TRAIN
updates then A16. Both arms receive48 native updates/432 rows before return and
16 updates/144 rows on return. Each starts from the same live full checkpoint,
including optimizer state. The deployed source, serving actor and benchmark128
remain unchanged. JSON reports cannot restore their weights or renew budgets.

This option explicitly consents to retaining the six original arrived TRAIN
episodes:54 rows and133968 encoded-array bytes. Existing coverage replay selects
two complete old episodes after each F wake, rotating A-blue,A-red,B-blue,B-red,
C-blue,C-red; presentation counts are6,6,5,5,5,5. No F episode enters this bank.
The fixed bank refuses unobserved cursor, membership, owner or array changes.
Only successful checked selection advances its cursor witness. Pause/resume
preserves that witness and previous paid work rather than repeating replay.

Both arms collect the same teacher diagnostics and native F2 plus oldABC6 games
after F and after A. Teacher checkpoints match native update counts:128,140,152,
176, then192 after return. Logical F counts differ (48 versus16); replay counts,
old-information exposure and storage remain separate. Equal native updates and
rows do not mean equal total computation, elapsed time or information access.
Final evaluation is unavailable; probe_test data never become training, replay,
promotion or tuning inputs. Earlier complete arm evidence must remain unchanged
before the next arm can perform work. Failed attempts and charges remain visible.

The selected two-arm mode has application480s, continuous readout420s, F/A300s,
and eight-game endpoint120.0s limits; its report is bounded to4MiB. These limits
are chosen before work starts. The original default and single-probe limits
above remain unchanged. Freeze and review a small local native protocol before
running a new comparison; this command is not a sweep or a claim of benefit.
The qualification record is [the rehearsal report](reports/recurrent-joint-rehearsal-20261031.md).

## Cold-start learnability reference

    python -m scripts.recurrent_joint_example --cold-fresh-task-probe --describe
    python -m scripts.recurrent_joint_example --cold-fresh-task-probe --output new-cold-probe.json

This exclusive option starts from the original seed17 cold full state, with zero
benchmark TRAIN updates. It selects one isolated `cold_F` arm through the same
helper, generic F48/A16 recipe and native endpoint kernel. It performs no replay
and creates no retained old-TRAIN replay bank. The original TRAIN packets remain
available for read-only diagnostics and the fixed A16 training phase. Here A16
is first A acquisition, not retention or recovery of an acquired old task.

Only explicit cold_F/source0 admits the native48/64 endpoints with24/32 updates
per head. Warm joint128 and source2 modes retain their original strict counts.
The zero-update benchmark evidence is sealed before F; source, serving actor,
benchmark, RNGs and original budgets remain protected. Corruption or replacement
of F endpoint evidence prevents A updates. Failed native work remains separately
recorded even when the committed-update budget is zero, and a failed learner
cannot retry. A game failure is a valid result; execution failure stays incomplete.

Limits are application240s, continuous readout200s, phases300s, eight-game
endpoint120.0s and report2MiB, born once. The command never extends a failed
budget. Freeze and review a separate small native protocol before execution.
Compare with the accepted warm F48 result only where its data/math/seed/dose
remain unchanged; different upfront history means different total computation.
Neither teacher fit nor one cold/warm result proves general plasticity loss,
generalization or circadian benefit. Qualification: [cold probe report](reports/recurrent-cold-probe-20261031.md).

## Class-balanced joint TRAIN reference

    python -m scripts.recurrent_joint_example --class-balanced-joint --describe
    python -m scripts.recurrent_joint_example --class-balanced-joint --output new-weighted-joint.json

This exclusive option uses the existing weighted schedule owner and native
class-balanced episode objective on the original joint128 TRAIN order. The
six sources,1152 training rows, seed17, H8 architecture, learning rate and momentum
match the ordinary joint reference. The fixed analytical weights use only each
current complete TRAIN episode; no probe or final labels enter training or
weight selection. Reserved probe-role episodes remain refused.

The original source and serving actor stay cold, while the separate candidate
receives128 weighted updates,64 per cue. Existing read-only evaluation reports
all six TRAIN teacher scores at0/128 and six native games at128. Wrapper and
native learner identities remain pinned through snapshot and clock callbacks;
a later callback cannot replace an earlier checked owner with an equal copy.
Original budgets, births, charges, failed work, RNGs and prior evidence remain
protected. No probe adaptation, replay or publication occurs in this mode.

Application240s/TRAIN180s/readout200s/768MiB and output2MiB limits are selected
before work starts. Freeze/review a small native protocol before qualification.
Equal updates and rows do not match weight computation, guards or elapsed time.
The original sequential weighted failed run is preserved; this joint reference
cannot recover its unknown endpoints. Evidence and limits: [weighted joint report](reports/recurrent-joint-weighted-20261031.md).
